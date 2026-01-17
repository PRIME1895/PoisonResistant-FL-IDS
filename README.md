# PythonProject – NSL-KDD pipeline + Federated (non-IID) client splitter

This repo is a runnable baseline for working with the **NSL‑KDD** dataset:

- Loads `KDDTrain+.txt` / `KDDTest+.txt` (auto-detects comma vs whitespace)
- Cleans labels (handles `normal.` vs `normal`)
- Preprocesses features (one‑hot for categorical + standard scaling for numeric)
- Trains a quick baseline classifier (Logistic Regression)
- Splits the training set into **non‑IID federated clients** (CSV + manifest)
- Runs a **PyTorch FedAvg** simulation (MLP per client)

## Expected dataset files
Place these files in the project root (same folder as `main.py`):

- `KDDTrain+.txt`
- `KDDTest+.txt`

## Quick start (Windows / PowerShell)

```powershell
python -m pip install -r requirements.txt
python main.py verify
python main.py train --binary
```

## (Important) PyTorch + CUDA in PyCharm (use global torch inside .venv)
If you already have **torch+cuda installed globally** (system Python), your project `.venv` must be configured to *inherit system site‑packages*.

### Option A — Enable inherit global site-packages in PyCharm
1. **PyCharm** → **Settings** → **Project** → **Python Interpreter**
2. Click the ⚙️ (gear) → **Show All…**
3. Select your project interpreter (`.venv`) → **Edit…**
4. Enable: **Inherit global site‑packages**

### Option B — Recreate the venv with system site-packages
If you recreate your venv from scratch:

```powershell
# from project root
rmdir /s /q .venv
py -3.12 -m venv .venv --system-site-packages
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Verify torch is visible:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## Commands

### 1) Verify dataset files
Prints delimiter/shape + top labels.

```powershell
python main.py verify
```

### 2) Train a baseline model (binary IDS)
Binary mapping: `normal -> 0`, everything else -> `1`.

```powershell
python main.py train --binary
```

### 3) Run preprocessing only (fit on train, transform test)
This prints processed shapes + a few feature rows.

```powershell
python -m preprocessing.preprocess
```

### 4) Split into 5 non‑IID federated clients
Creates 5 CSV files + a `manifest.json` describing the split and per‑client label family counts.

```powershell
python main.py split-clients --client-size 20000 --seed 42
```

Outputs (default):

- `data/clients/client_1.csv`
- `data/clients/client_2.csv`
- `data/clients/client_3.csv`
- `data/clients/client_4.csv`
- `data/clients/client_5.csv`
- `data/clients/manifest.json`

Tip: for a faster demo run:

```powershell
python main.py split-clients --client-size 5000 --seed 42
```

### 5) Federated training with PyTorch (MLP + FedAvg)
This runs a lightweight FedAvg simulation using the generated `client_*.csv` files and evaluates on `KDDTest+.txt`.

```powershell
# 1) Create clients
python main.py split-clients --client-size 20000 --seed 42

# 2) Run FedAvg on CPU
python main.py fl-train --rounds 5 --local-epochs 1 --batch-size 256 --lr 0.001 --device cpu --seed 1337
```

If you have a GPU + CUDA:

```powershell
python main.py fl-train --device cuda
```

## Notes
- The loader drops the `difficulty` column when present.
- Federated splitting uses common KDD/NSL-KDD family groupings: `normal`, `dos`, `probe`, `r2l`, `u2r`.
- Some families (especially `u2r`) are rare. If you use a small `--client-size`, the “rare attacks” client may still contain few/no `u2r` rows depending on availability.

## Troubleshooting
- If `python main.py fl-train` says torch is missing:
  - you’re running a Python interpreter that doesn’t have torch installed
  - or your `.venv` is not inheriting global site-packages

## Tests

```powershell
pytest -q
```
