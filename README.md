# NSL‑KDD Poisoning‑Resistant Federated IDS (Cross‑Layer Trust + Robust Aggregation)

A practical, demo‑ready **Federated Learning Intrusion Detection System (FL‑IDS)** built on the **NSL‑KDD** dataset.

This repo is designed for a strong viva/demo:
- You can run a clean **baseline FL** experiment.
- You can simulate **label‑flipping poisoning**.
- You can defend using **cross‑layer trust scoring** (cosine similarity + loss stability + cross‑layer consistency) and robust aggregation.
- Every run automatically logs round metrics **locally** into the `runs/` folder.

---

## What problem are we solving?
Traditional IDS training assumes all data is centralized. In real deployments, different organizations or network segments may not be able to share raw traffic logs. Federated Learning (FL) enables collaborative training **without sharing raw data**.

But FL introduces a new threat: **poisoning attacks** (malicious clients send harmful updates). This project demonstrates:
1) the baseline FL‑IDS,
2) how poisoning hurts it, and
3) how poisoning‑resistant aggregation can recover performance.

---

## Tool stack (your stack)
- **Development:** PyCharm Pro
- **Deployment (later phase):** Heroku
- **Monitoring (later phase):** Datadog
- **Diagrams (optional):** ToDiagram / Visme

---

## Repo features
### Dataset & preprocessing
- Robust NSL‑KDD loader (`KDDTrain+.txt`, `KDDTest+.txt`)
- Drops difficulty column when present
- One‑hot encoding: `protocol_type`, `service`, `flag`
- Standard scaling for numeric columns
- Binary labels: `normal → 0`, `attack → 1`

### Federated learning simulation
- Non‑IID split into 5 realistic clients (normal / DoS / probe / rare attacks)
- PyTorch MLP client model
- FedAvg baseline

### Poisoning + defenses
- **Phase 6:** Label‑flipping poisoning on selected clients
- **Phase 7 Option A:** Cosine similarity trust weighting + optional outlier drop
  - Trust score combines:
    - cosine similarity of updates
    - loss stability
    - cross‑layer consistency
- **Phase 7 Option B:** Update clipping + robust aggregation
  - trimmed mean
  - coordinate median

### Phase 8 logging
- Local experiment logging (runs/ folder)

---

## Project structure (high level)
- `main.py` — CLI entry point (verify/train/split-clients/fl-train)
- `nsl_kdd/` — dataset loading + FL simulation (+ local experiment logger)
- `preprocessing/` — feature preprocessing pipeline
- `data/clients/` — generated client CSVs + `manifest.json`
- `tests/` — pytest suite

---

## Dataset files (required)
Place these files in the project root (same folder as `main.py`):
- `KDDTrain+.txt`
- `KDDTest+.txt`

---

## Setup
```powershell
python -m pip install -r requirements.txt
```

### PyTorch note (GPU / global torch)
This repo imports torch only when you run `fl-train`. If you already installed **torch+CUDA globally**, make sure your PyCharm `.venv` is configured to **inherit global site‑packages**.

---

## Quickstart (recommended demo path)
These steps reproduce the clean end‑to‑end pipeline.

### 1) Verify dataset
```powershell
python main.py verify
```

### 2) Train a centralized baseline (binary)
```powershell
python main.py train --binary
```

### 3) Create 5 non‑IID federated clients
Recommended for fast + strong metrics in demos:

```powershell
python main.py split-clients --client-size 2000 --seed 42
```

*(You can increase `--client-size` later for heavier experiments, but 2000 is a great “demo sweet spot”.)*

### 4) Run Federated Learning (baseline FedAvg)
```powershell
python main.py fl-train --rounds 5 --local-epochs 1 --device cpu --seed 42 --aggregation fedavg
```

---

## Phase 6 — Poisoning attack (label flipping)
Flip labels for one or more malicious clients.

Example: poison client 2 by flipping 50% of its labels:

```powershell
python main.py fl-train --rounds 5 --local-epochs 1 --device cpu --seed 42 --malicious-clients 2 --label-flip-rate 0.5 --aggregation fedavg
```

---

## Phase 7 — Poisoning‑resistant defenses

### Option A: Cross‑layer trust weighting (cosine) + outlier drop
Trust score (per client update):
- cosine similarity
- loss stability
- cross‑layer consistency

```powershell
python main.py fl-train --rounds 5 --local-epochs 1 --device cpu --seed 42 --malicious-clients 2 --label-flip-rate 0.5 --aggregation cosine --cosine-drop-k 1 --trust-alpha 1.0 --trust-beta 0.5 --trust-gamma 0.5
```

### Option B: Clipping + robust aggregation
Trimmed mean:
```powershell
python main.py fl-train --rounds 5 --local-epochs 1 --device cpu --seed 42 --malicious-clients 2 --label-flip-rate 0.5 --aggregation trimmed_mean --clip-norm 5 --trim-ratio 0.2
```

Coordinate median:
```powershell
python main.py fl-train --rounds 5 --local-epochs 1 --device cpu --seed 42 --malicious-clients 2 --label-flip-rate 0.5 --aggregation median --clip-norm 5
```

---

## Phase 8 — Local experiment logging (recommended)
Every time you run `fl-train`, this repo automatically saves the generated metrics **locally** (no external services needed).

### Where files are saved
A new folder will be created:
- `runs/<run_id>/run.json` — run config + metadata
- `runs/<run_id>/rounds.json` — per-round metrics (JSON)
- `runs/<run_id>/rounds.csv` — per-round metrics (CSV)

`rounds.csv` is the easiest to open quickly in Excel.

### Example
Run your experiment (any aggregation/poisoning settings):

```powershell
python main.py split-clients --client-size 2000 --seed 42
python main.py fl-train --rounds 1 --local-epochs 1 --device cpu --seed 42 --malicious-clients 2 --label-flip-rate 0.5 --aggregation cosine --cosine-drop-k 1 --trust-alpha 1.0 --trust-beta 0.5 --trust-gamma 0.5
```

Then open:
- `runs/<latest_run_id>/rounds.csv`

---

## What outputs should you expect?
### Files
When you run `split-clients`, you’ll get:
- `data/clients/client_1.csv` … `client_5.csv`
- `data/clients/manifest.json` (family distributions)

### Terminal output
When you run `fl-train`, you’ll see:
- final metrics (accuracy/precision/recall/F1)
- round-by-round history
- for `--aggregation cosine`, extra diagnostics like:
  - `cosine_sim_mean`
  - `trust_mean`
  - `loss_stability_mean`
  - `cross_layer_mean`
  - `dropped_clients`

---

## Evaluation metrics (what to report)
This repo reports standard IDS‑friendly binary classification metrics on `KDDTest+.txt`:
- **Accuracy** — overall correctness
- **Precision** — how many predicted attacks are actually attacks
- **Recall (Detection Rate / TPR)** — how many true attacks are detected (very important for IDS)
- **F1-score** — balance between precision and recall

For your report/viva, you should compare these scenarios:
1) Baseline FL (FedAvg)
2) FL + Poisoning (label flipping)
3) FL + Defense (Option A cosine trust, Option B clipped robust aggregation)

---

## Tests
```powershell
pytest
```

---

## Dataset credit (NSL‑KDD)
This project uses the **NSL‑KDD** dataset, an improved version of the KDD’99 dataset for intrusion detection research.

Please cite:
- M. Tavallaee, E. Bagheri, W. Lu, and A. A. Ghorbani, “A Detailed Analysis of the KDD CUP 99 Data Set,” *Proceedings of the IEEE Symposium on Computational Intelligence for Security and Defense Applications (CISDA)*, 2009.

Dataset reference/download:
- https://www.unb.ca/cic/datasets/nsl.html

