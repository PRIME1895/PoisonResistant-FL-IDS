from __future__ import annotations

"""Client feedback payloads.

This module defines a minimal, serializable structure for *server → client* feedback
in our FL simulation.

In real federated deployments, the server can send each client a small message per
round (e.g., trust score, whether the update was used, and why). Here we persist
that feedback to disk under `runs/<run_id>/client_feedback.json` so it can be used
for analysis or future client-side adaptations (self-audit/quarantine).

We intentionally keep the schema flexible (dict-based) so adding more diagnostics
won't break older runs.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ClientFeedback:
    """Feedback for a single client for a single FL round."""

    client_id: int  # 1-based index matching client_*.csv
    used: bool
    trust: Optional[float] = None
    cosine_similarity: Optional[float] = None
    loss_stability: Optional[float] = None
    cross_layer: Optional[float] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "client_id": int(self.client_id),
            "used": bool(self.used),
        }
        if self.trust is not None:
            out["trust"] = float(self.trust)
        if self.cosine_similarity is not None:
            out["cosine_similarity"] = float(self.cosine_similarity)
        if self.loss_stability is not None:
            out["loss_stability"] = float(self.loss_stability)
        if self.cross_layer is not None:
            out["cross_layer"] = float(self.cross_layer)
        if self.notes is not None:
            out["notes"] = str(self.notes)
        return out


def round_feedback_payload(*, round_num: int, feedback: List[ClientFeedback]) -> Dict[str, Any]:
    """Build a JSON-serializable payload for a single round."""

    return {
        "round": int(round_num),
        "clients": [f.to_dict() for f in feedback],
    }

