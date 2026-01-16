# PythonProject – NSL-KDD baseline pipeline

This repo contains a small, runnable baseline that loads the NSL‑KDD files (`KDDTrain+.txt`, `KDDTest+.txt`), preprocesses features (one‑hot for categorical + standard scaling for numeric), and trains a basic classifier.

## Files expected
- `KDDTrain+.txt`
- `KDDTest+.txt`

## Quick start (Windows / PowerShell)
```powershell
python -m pip install -r requirements.txt
python main.py verify
python main.py train --binary
```

### Commands
- `python main.py verify` – checks that the files exist and prints basic stats
- `python main.py train` – trains a baseline model and prints metrics

## Notes
- By default `--binary` maps `normal` → 0 and everything else → 1.
- If your dataset delimiter differs, the loader auto-detects comma vs whitespace.
