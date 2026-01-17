# PythonProject – NSL-KDD pipeline + Federated (non-IID) client splitter

This repo is a runnable baseline for working with the **NSL‑KDD** dataset:

- Loads `KDDTrain+.txt` / `KDDTest+.txt` (auto-detects comma vs whitespace)
- Cleans labels (handles `normal.` vs `normal`)
- Preprocesses features (one‑hot for categorical + standard scaling for numeric)
- Trains a quick baseline classifier (Logistic Regression)
- **Phase 3:** Splits the training set into **5 non‑IID federated clients** (CSV + manifest)

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

### 4) Split into 5 non‑IID federated clients (Phase 3)
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

Tip: for a faster demo run, use a smaller size:

```powershell
python main.py split-clients --client-size 5000 --seed 42
```

## Notes
- The loader drops the `difficulty` column when present.
- Federated splitting uses common KDD/NSL-KDD family groupings: `normal`, `dos`, `probe`, `r2l`, `u2r`.
- Some families (especially `u2r`) are rare. If you use a small `--client-size`, the “rare attacks” client may still contain few/no `u2r` rows depending on availability.

## Tests

```powershell
pytest -q
```
