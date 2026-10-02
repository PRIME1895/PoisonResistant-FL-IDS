"""
inference_server.py

FastAPI inference server for the Poison-Resistant Federated IDS project.

Serves the trained global classifier (produced by federated training across
the 30-client setup) as a REST API, so a network flow / node-behaviour
record can be scored for maliciousness in real time, and a client's
aggregated update can be scored for trust before being accepted into the
global model.

Run:
    uvicorn inference_server:app --host 0.0.0.0 --port 8000

Docs:
    http://localhost:8000/docs
"""

import os
import time
import logging
from contextlib import asynccontextmanager
from typing import List

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fl-ids-inference")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_PATH = os.getenv("FLIDS_MODEL_PATH", "models/global_model.pt")
FEATURE_DIM = int(os.getenv("FLIDS_FEATURE_DIM", "41"))  # NSL-KDD numeric feature count
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRUST_THRESHOLD = float(os.getenv("FLIDS_TRUST_THRESHOLD", "0.5"))


# ---------------------------------------------------------------------------
# Model definition
# Mirrors the classifier architecture used during federated training.
# Swap this for the real class from your training code if it differs.
# ---------------------------------------------------------------------------
class IDSClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x)).squeeze(-1)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class FlowRecord(BaseModel):
    features: List[float] = Field(
        ..., description=f"Flat vector of {FEATURE_DIM} preprocessed numeric features."
    )


class BatchRequest(BaseModel):
    records: List[FlowRecord]


class PredictionResponse(BaseModel):
    malicious: bool
    confidence: float
    trust_score: float
    latency_ms: float


class BatchPredictionResponse(BaseModel):
    results: List[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    device: str
    model_loaded: bool


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
state = {"model": None}


def load_model() -> nn.Module:
    model = IDSClassifier(input_dim=FEATURE_DIM).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict)
        logger.info("Loaded global model weights from %s", MODEL_PATH)
    else:
        logger.warning(
            "No checkpoint found at %s — serving an untrained model. "
            "Set FLIDS_MODEL_PATH to your trained weights.",
            MODEL_PATH,
        )
    model.eval()
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = load_model()
    yield
    state["model"] = None


app = FastAPI(
    title="PR-FLIDS Inference Server",
    description="Serves the federated, poison-resistant intrusion-detection model.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
def _score(features: List[float]) -> PredictionResponse:
    if len(features) != FEATURE_DIM:
        raise HTTPException(
            status_code=422,
            detail=f"Expected {FEATURE_DIM} features, got {len(features)}.",
        )

    start = time.perf_counter()
    model = state["model"]
    with torch.no_grad():
        x = torch.tensor(np.array(features, dtype=np.float32)).unsqueeze(0).to(DEVICE)
        confidence = model(x).item()
    latency_ms = (time.perf_counter() - start) * 1000

    return PredictionResponse(
        malicious=confidence >= TRUST_THRESHOLD,
        confidence=round(confidence, 4),
        trust_score=round(1.0 - confidence, 4),
        latency_ms=round(latency_ms, 3),
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        device=DEVICE,
        model_loaded=state["model"] is not None,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(record: FlowRecord) -> PredictionResponse:
    """Score a single flow / node-behaviour record for maliciousness."""
    return _score(record.features)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchRequest) -> BatchPredictionResponse:
    """Score multiple records in one call."""
    results = [_score(r.features) for r in request.records]
    return BatchPredictionResponse(results=results)
