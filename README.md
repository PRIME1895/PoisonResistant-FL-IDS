# NSL‑KDD Federated IDS (Baseline + PyTorch FedAvg)

A practical, demo‑ready pipeline that turns the **NSL‑KDD** intrusion detection dataset into:
- a clean ML baseline, and
- a **federated learning simulation** (non‑IID clients + PyTorch FedAvg)

Use it for coursework, a viva demo, or as the foundation for poisoning/robust aggregation experiments.

---

## What you can do with this repo
- **Load NSL‑KDD reliably** (comma or whitespace delimited)
- **Preprocess features**
  - One‑hot: `protocol_type`, `service`, `flag`
  - Scale numeric features
  - Binary labels: `normal → 0`, `attack → 1`
- **Train a quick baseline** (Logistic Regression)
- **Simulate a federated environment**
  - Split training data into **5 non‑IID clients**
  - Run **FedAvg** using a simple PyTorch MLP
- **Export client datasets** (CSV + manifest) for repeatable experiments

---

## Dataset files (required)
Place these files in the project root (same folder as `main.py`):
- `KDDTrain+.txt`
- `KDDTest+.txt`

---

## Quick demo (recommended path)
Run these in order to get a full end‑to‑end demo:

```powershell
python -m pip install -r requirements.txt
python main.py verify
python main.py train --binary
python main.py split-clients --client-size 5000 --seed 42
python main.py fl-train --rounds 2 --local-epochs 1 --device cpu
```

---

## Commands

### 1) Verify dataset health
Prints delimiter/shape + top labels.

```powershell
python main.py verify
```

### 2) Train baseline IDS (binary)

```powershell
python main.py train --binary
```

### 3) Preprocess only (fit on train → transform test)

```powershell
python -m preprocessing.preprocess
```

### 4) Create non‑IID federated clients
Creates **5 clients** + a `manifest.json` with family counts.

```powershell
python main.py split-clients --client-size 20000 --seed 42
```

**Output folder (default):** `data/clients/`
- `client_1.csv` … `client_5.csv`
- `manifest.json`

Tip (fast demo):

```powershell
python main.py split-clients --client-size 5000 --seed 42
```

### 5) Federated training (PyTorch MLP + FedAvg)
Runs FedAvg on the generated client CSVs and evaluates on `KDDTest+.txt`.

```powershell
python main.py fl-train --rounds 5 --local-epochs 1 --batch-size 256 --lr 0.001 --device cpu --seed 1337
```

If you have a GPU:

```powershell
python main.py fl-train --device cuda
```

---

## PyTorch + CUDA in PyCharm (use global torch inside `.venv`)
If you already have **torch+CUDA installed globally** (system Python), your `.venv` must be configured to **inherit system site‑packages**.

### Option A — Enable in PyCharm
1. **PyCharm** → **Settings** → **Project** → **Python Interpreter**
2. Click ⚙️ → **Show All…**
3. Select your interpreter (`.venv`) → **Edit…**
4. Enable: **Inherit global site‑packages**

### Option B — Recreate the venv with system site-packages

```powershell
rmdir /s /q .venv
py -3.12 -m venv .venv --system-site-packages
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Verify torch is visible:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

---

## Notes
- The loader drops the `difficulty` column when present.
- Federated splitting uses common NSL‑KDD families: `normal`, `dos`, `probe`, `r2l`, `u2r`.
- Some families (especially `u2r`) are rare; small `--client-size` values may under‑sample them.

---

## Troubleshooting
- **`fl-train` says torch is missing**
  - You’re running a Python interpreter that doesn’t have torch installed, OR
  - Your `.venv` is not inheriting global site‑packages.

---

## Tests

```powershell
pytest -q
```

---

## Dataset credit (NSL‑KDD)
This project uses the **NSL‑KDD** dataset, an improved version of the KDD’99 dataset for intrusion detection research.

Please cite:
- M. Tavallaee, E. Bagheri, W. Lu, and A. A. Ghorbani, “A Detailed Analysis of the KDD CUP 99 Data Set,” *Proceedings of the IEEE Symposium on Computational Intelligence for Security and Defense Applications (CISDA)*, 2009.

Dataset reference/download:
- https://www.unb.ca/cic/datasets/nsl.html

