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

## Research motivation (why this is important)
Existing Federated Learning–based IDS often assume honest participating clients, which is unrealistic in adversarial network environments.

Known gaps this project addresses:
- **Poisoning/backdoor attacks** can severely degrade FL‑IDS performance.
- Many defenses are **generic FL security** and not tailored to IDS needs.
- Prior FL‑IDS defenses often lack **cross‑layer validation**, making it harder to distinguish real traffic anomalies from malicious client updates.
- There is limited work on **trust‑aware aggregation** that dynamically evaluates client reliability during federated training.

This repo is built to demonstrate those issues and implement a practical defense you can explain and measure.

---

## Implementation phases (mapped to the repo)
These phases match the methodology-style roadmap used in the project.

### Phase 1 — Dataset setup (NSL‑KDD)
**Input:** `KDDTrain+.txt`, `KDDTest+.txt` (in project root)

What happens:
- File loading supports common NSL‑KDD formatting.
- Drops the difficulty column when present.

CLI:
```powershell
python main.py verify
```

### Phase 2 — Preprocessing (binary IDS features)
What happens:
- One‑hot encoding: `protocol_type`, `service`, `flag`
- Standard scaling for numeric features
- Binary labels: `normal → 0`, `attack → 1`

CLI:
```powershell
python main.py train --binary
```

### Phase 3 — Federated environment simulation (non‑IID clients)
What happens:
- The NSL‑KDD training set is split into **5 non‑IID clients** with different attack‑family mixtures.

Recommended demo split:
```powershell
python main.py split-clients --client-size 2000 --seed 42
```

Outputs:
- `data/clients/client_1.csv` … `client_5.csv`
- `data/clients/manifest.json`

### Phase 4 — Local IDS model (client‑side)
What happens:
- Each client trains a small PyTorch MLP locally.

### Phase 5 — Baseline Federated Learning (FedAvg)
Goal:
- Establish baseline FL performance before attacks.

CLI:
```powershell
python main.py fl-train --rounds 5 --local-epochs 1 --device cpu --seed 42 --aggregation fedavg
```

### Phase 6 — Poisoning attack (label flipping)
Goal:
- Show the degradation under malicious clients.

CLI example (poison client 2):
```powershell
python main.py fl-train --rounds 5 --local-epochs 1 --device cpu --seed 42 --malicious-clients 2 --label-flip-rate 0.5 --aggregation fedavg
```

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

---

## Repo features
(Quick summary — detailed steps are in **Implementation phases** above.)

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

## Plots (3 research‑grade figures)
Use the local run logs (`runs/<run_id>/rounds.json` or `runs.csv`) to generate the three key plots:

### Plot 1 — Accuracy vs FL Rounds
Shows learning progression and compares:
- Phase 5: Clean FedAvg
- Phase 6: FedAvg + Poisoning
- Phase 7: Poisoning‑Resistant FL

### Plot 2 — Recall vs FL Rounds (IDS priority)
Recall is critical in IDS because false negatives mean missed attacks.
This plot shows poisoning increasing missed attacks and the defense reducing them.

### Plot 3 — Trust/Defense signal vs FL Rounds (Phase 7 only)
Shows the defense is active and interpretable (e.g., `trust_mean`, `cosine_sim_mean`, or `dropped_clients`).

### Generate plots
After you have 3 runs saved locally (clean/poisoned/defended), run:

```powershell
python main.py plot --clean "runs/binary_model(clean)" --poisoned "runs/binary_model(poisoned)" --defended "runs/binary_model(defended)" --out-dir figures
```

Output:
- `figures/plot1_accuracy_vs_rounds.png`
- `figures/plot2_recall_vs_rounds.png`
- `figures/plot3_defense_signal_vs_rounds.png`

---

## Multi-run comparison (all scenarios in one plot)
If you have the 6 scenario folders under `runs/` (from centralized baseline through robust defenses), you can generate:
- **One single comparison plot** with all runs overlaid
- **One merged metrics file** containing every round’s metrics from every run

Run:

```powershell
python -m nsl_kdd.compare_runs
```

Outputs:
- `figures/comparison/comparison_accuracy.png` (all scenarios: accuracy vs rounds)
- `figures/comparison/comparison_recall.png` (all scenarios: recall vs rounds)
- `figures/comparison/all_round_metrics.json`
- `figures/comparison/all_round_metrics.csv`

The merged metrics files contain a `run_name` column so you can filter/group by scenario.

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
