# NSL‑KDD Poisoning‑Resistant Federated IDS (Cross‑Layer Trust + Robust Aggregation)

A practical, demo‑ready **Federated Learning Intrusion Detection System (FL‑IDS)** built on the **NSL‑KDD** dataset.

This repo is designed for a strong viva/demo:
- Run a clean **baseline FL** experiment.
- Simulate **label‑flipping poisoning**.
- Defend using **cross‑layer trust scoring** (cosine similarity + loss stability + cross‑layer consistency) and robust aggregation.
- Every run automatically logs round metrics **locally** into the `runs/` folder.

---

## What problem are we solving?
Traditional IDS training assumes all data is centralized. In real deployments, different organizations or network segments may not be able to share raw traffic logs. Federated Learning (FL) enables collaborative training **without sharing raw data**.

But FL introduces a new threat: **poisoning attacks** (malicious clients send harmful updates). This project demonstrates:
1) the baseline FL‑IDS,
2) how poisoning hurts it, and
3) how poisoning‑resistant aggregation can recover performance.

---

## Research motivation (why this is important)
Existing Federated Learning–based IDS often assume honest participating clients, which is unrealistic in adversarial network environments.

Known gaps this project addresses:
- **Poisoning/backdoor attacks** can severely degrade FL‑IDS performance.
- Many defenses are **generic FL security** and not tailored to IDS needs.
- Prior FL‑IDS defenses often lack **cross‑layer validation**, making it harder to distinguish real traffic anomalies from malicious client updates.
- There is limited work on **trust‑aware aggregation** that dynamically evaluates client reliability during federated training.

---

## Implementation phases (mapped to the repo)
These phases match the methodology-style roadmap used in the project.

### Phase 1 — Dataset setup (NSL‑KDD)
**Input:** `KDDTrain+.txt`, `KDDTest+.txt` (in project root)

What happens:
- Auto-detects delimiter (comma vs whitespace)
- Assigns column names (best-effort)
- Drops the difficulty column when present

CLI:
```powershell
python main.py verify
```

### Phase 2 — Baseline centralized training (sanity check)
What happens:
- Baseline sklearn model (fast sanity check)
- Binary labels: `normal → 0`, `attack → 1`

CLI:
```powershell
python main.py train --binary
```

### Phase 3 — Preprocessing (binary IDS features)
What happens:
- One‑hot encoding: `protocol_type`, `service`, `flag`
- Standard scaling for numeric features
- Binary labels: `normal → 0`, `attack → 1`

> Note: In the FL pipeline this preprocessing is fit once on the union of client data
> (equivalent to global train), then applied consistently to all clients + test.

### Phase 4 — Federated environment simulation (non‑IID clients)
What happens:
- The NSL‑KDD training set is split into **non‑IID clients** with different attack‑family mixtures.

Recommended demo split (2k rows/client works well and runs fast):
```powershell
python main.py split-clients --client-size 2000 --seed 42
```

Outputs:
- `data/clients/client_1.csv` … `client_5.csv`
- `data/clients/manifest.json` (family distributions)

### Phase 5 — Local IDS model (client‑side)
What happens:
- Each client trains a small PyTorch MLP locally (binary classification).
- Only model updates are aggregated (raw client data never leaves the client).

### Phase 6 — Federated Learning experiments (baseline + poisoning)
#### Phase 6A — Clean FedAvg (baseline)
CLI:
```powershell
python main.py fl-train --rounds 5 --local-epochs 1 --device cpu --seed 42 --aggregation fedavg
```

#### Phase 6B — FedAvg + Poisoning (label flipping)
CLI example (poison **client 2**):
```powershell
python main.py fl-train --rounds 5 --local-epochs 1 --device cpu --seed 42 --malicious-clients 2 --label-flip-rate 0.5 --aggregation fedavg
```

**Which clients are poisoned?**
- Poisoning is controlled by `--malicious-clients`, which takes **1-based client indices** corresponding to files in `data/clients/`.
  - `--malicious-clients 2` → poisons `data/clients/client_2.csv`
  - `--malicious-clients 2,5` → poisons `client_2.csv` and `client_5.csv`
- If you want to justify *why* a certain client is chosen (e.g., “high DoS concentration”), cite `data/clients/manifest.json` which stores family distributions per client.

### Phase 7 — Poisoning‑resistant defenses (core contribution)
Two defense options are implemented.

#### Option A (recommended): Trust‑aware aggregation with cross‑layer validation
Trust score combines:
- cosine similarity of client updates
- loss stability
- cross‑layer consistency (network/transport/application feature groups)

CLI example:
```powershell
python main.py fl-train --rounds 5 --local-epochs 1 --device cpu --seed 42 --malicious-clients 2 --label-flip-rate 0.5 --aggregation cosine --cosine-drop-k 1 --trust-alpha 1.0 --trust-beta 0.5 --trust-gamma 0.5
```

#### Option B: Robust aggregation (clipping + robust reducer)
Trimmed mean:
```powershell
python main.py fl-train --rounds 5 --local-epochs 1 --device cpu --seed 42 --malicious-clients 2 --label-flip-rate 0.5 --aggregation trimmed_mean --clip-norm 5 --trim-ratio 0.2
```

Coordinate median:
```powershell
python main.py fl-train --rounds 5 --local-epochs 1 --device cpu --seed 42 --malicious-clients 2 --label-flip-rate 0.5 --aggregation median --clip-norm 5
```

### Phase 8 — Local experiment logging (reproducibility)
Every `fl-train` run stores metrics locally under `runs/<run_id>/`:
- `run.json` — run config + metadata
- `rounds.json` — per-round metrics in JSON
- `rounds.csv` — per-round metrics in CSV (Excel friendly)

This is produced by `nsl_kdd/local_logger.py`.

---

## Repo features
- Dataset loading + preprocessing (binary IDS)
- Non‑IID client simulation
- Baseline FL (FedAvg)
- Poisoning (label flipping)
- Poisoning‑resistant aggregation (trust‑aware + robust)
- Local run logging (`runs/`)

---

## What outputs should you expect?
### Files
When you run `split-clients`, you’ll get:
- `data/clients/client_1.csv` … `client_5.csv`
- `data/clients/manifest.json` (family distributions)

### Local metrics logs
When you run `fl-train`, you’ll get a new folder `runs/<run_id>/` containing:
- `run.json`
- `rounds.json`
- `rounds.csv`

---

## Evaluation metrics (what to report)
This repo reports standard IDS‑friendly binary classification metrics on `KDDTest+.txt`:
- **Accuracy** — overall correctness
- **Precision** — how many predicted attacks are actually attacks
- **Recall (Detection Rate / TPR)** — how many true attacks are detected (very important for IDS)
- **F1-score** — balance between precision and recall

### Extra metrics (recommended for report)
Your advisor’s suggestion is solid: add IDS-specific error rates.
At minimum, report:
- **False Positive Rate (FPR)** = FP / (FP + TN)
- **False Negative Rate (FNR)** = FN / (FN + TP)  *(missed attacks)*

(These aren’t currently printed by the CLI, but they are straightforward to compute from a confusion matrix. See **Future Scope** below.)

---

## Plots (3 research‑grade figures)
Use the local run logs (`runs/<run_id>/rounds.json` or `rounds.csv`) to generate the three key plots:

### Plot 1 — Accuracy vs FL Rounds
Compare:
- Clean FedAvg
- FedAvg + Poisoning
- Defended FL

### Plot 2 — Recall vs FL Rounds (IDS priority)
Recall is critical in IDS because false negatives mean missed attacks.

### Plot 3 — Trust/Defense signal vs FL Rounds (Phase 7 only)
Shows the defense is active and interpretable (e.g., `trust_mean`, `cosine_sim_mean`, or `dropped_clients`).

### Generate plots
After you have 3 runs saved locally (clean/poisoned/defended), run:

```powershell
python main.py plot --clean "runs/<clean_run_id>" --poisoned "runs/<poisoned_run_id>" --defended "runs/<defended_run_id>" --out-dir figures
```

Output:
- `figures/plot1_accuracy_vs_rounds.png`
- `figures/plot2_recall_vs_rounds.png`
- `figures/plot3_defense_signal_vs_rounds.png`

---

## Multi-run comparison (single plot across all scenarios)
If you have the scenario folders under `runs/`:
- `Centralized_baseline/`
- `baseline_FedAvg/`
- `Poisoning_attack(label_flipping)/`
- `Cross-layer_trust_weighting(cosine)+outlier_drop/`
- `trimmed_mean(clipping+robust aggregation)/`
- `coordinate_median(trimming+robust aggregation)/`

You can generate:
- **One single comparison plot** with all runs overlaid
- **One merged metrics file** containing every round’s metrics from every run

Run:
```powershell
python -m nsl_kdd.compare_runs
```

Outputs (already generated into `figures/comparison/`):
- `figures/comparison/comparison_accuracy.png`
- `figures/comparison/comparison_recall.png`
- `figures/comparison/all_round_metrics.json`
- `figures/comparison/all_round_metrics.csv`

---

## Future scope / improvements (based on advisor feedback)
If your ma’am asked for these, here’s how they map to the project.

### 1) Flow diagram (architecture + dataflow)
Yes — you should add a clear flow diagram in your report:
- **NSL‑KDD → preprocessing → client splits → local training → server aggregation → evaluation → plots**

You can draw it in ToDiagram and include:
- (a) data flow (where files are read/written)
- (b) FL flow (client updates → aggregator)

### 2) Compare 5 clients vs 10 clients
You already support this:
- `python main.py split-clients --n-clients 10 --client-size 2000 --seed 42`

Only note: the current code uses a custom non‑IID spec only for 5 clients (default). For 10 clients it will fall back to a generic split unless you add 10‑client specs.

### 3) What happens when attacker ratio is high?
Run the same experiment but poison more clients, e.g.:
- 1/5 attackers (20%)
- 2/5 attackers (40%)
- 3/5 attackers (60%)

Then compare **recall + FPR**. This becomes a nice “robustness vs attacker ratio” table/plot.

### 4) Can the server know which client is attacker? (and “send back” info)
Right now:
- The server **computes per-client trust diagnostics** internally (cosine similarity / cross-layer consistency).
- It uses these to down-weight or drop clients.

A novel extension is:
- Send each client a “trust feedback” score per round.
- Clients can self-audit (or be quarantined by policy) if their trust is consistently low.

### 5) Comparison with existing systems
You already have multiple baselines in `runs/`:
- centralized baseline
- FedAvg
- FedAvg + poisoning
- trust-aware cosine
- trimmed mean
- coordinate median

That’s a strong comparison section.

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
