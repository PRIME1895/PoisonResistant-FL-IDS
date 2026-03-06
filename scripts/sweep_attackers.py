from __future__ import annotations

# Ensure imports like `import nsl_kdd` work when running as a script.
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

"""Run a small sweep over attacker counts and summarize recall + FPR.

This script is meant for quick comparisons like:
- 30 clients, 5 malicious vs 10 malicious
- optionally higher attacker counts

It uses the existing FL training entry point (nsl_kdd.torch_fl.train_fedavg_binary)
so results are logged under runs/ automatically.

Output:
- prints a table to stdout
- writes figures/sweeps/attackers_summary.csv

Note:
- Malicious clients are selected deterministically (seeded) from 1..n_clients.
"""

import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from nsl_kdd.data import load_nsl_kdd, train_test_paths
from nsl_kdd.torch_fl import FLConfig, train_fedavg_binary


def _pick_malicious(n_clients: int, k: int, *, seed: int) -> tuple[int, ...]:
    if k <= 0:
        return ()
    if k >= n_clients:
        return tuple(range(1, n_clients + 1))
    rng = np.random.default_rng(seed)
    picks = rng.choice(np.arange(1, n_clients + 1), size=int(k), replace=False)
    return tuple(int(x) for x in sorted(picks))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sweep attacker counts and compare recall/FPR")
    p.add_argument("--clients-dir", type=str, default="data/clients", help="Directory with client_*.csv")
    p.add_argument("--attackers", type=str, default="5,10", help="Comma-separated attacker counts")
    p.add_argument("--label-flip-rate", type=float, default=0.3, help="Label flip rate for malicious clients")
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument(
        "--aggregation",
        type=str,
        default="fedavg",
        choices=["fedavg", "cosine", "trimmed_mean", "median"],
        help="Aggregation/defense strategy",
    )
    p.add_argument("--cosine-drop-k", type=int, default=0)
    p.add_argument("--clip-norm", type=float, default=None)
    p.add_argument("--trim-ratio", type=float, default=0.2)
    p.add_argument("--trust-alpha", type=float, default=1.0)
    p.add_argument("--trust-beta", type=float, default=0.0)
    p.add_argument("--trust-gamma", type=float, default=0.5)
    return p


def _parse_int_list(s: str) -> list[int]:
    s = str(s).strip()
    if not s:
        return []
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def main() -> int:
    args = build_parser().parse_args()

    root = Path(__file__).resolve().parents[1]
    clients_path = (root / args.clients_dir).resolve()

    # Ensure output dirs exist early.
    out_dir = (root / "figures" / "sweeps").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    client_files = sorted(clients_path.glob("client_*.csv"))
    if not client_files:
        raise SystemExit(f"No client CSVs found in: {clients_path}. Run split-clients first.")

    client_dfs = [pd.read_csv(p) for p in client_files]
    n_clients = len(client_dfs)

    _, test_path = train_test_paths(root)
    test_df = load_nsl_kdd(test_path)

    attacker_counts = _parse_int_list(args.attackers)
    if not attacker_counts:
        raise SystemExit("--attackers must contain at least one value")

    rows: list[dict[str, object]] = []
    run_dirs: list[str] = []

    for k in attacker_counts:
        mc = _pick_malicious(n_clients, k, seed=int(args.seed) + 100 * k)

        cfg = FLConfig(
            rounds=int(args.rounds),
            local_epochs=int(args.local_epochs),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            device=str(args.device),
            seed=int(args.seed),
            malicious_clients=mc,
            label_flip_rate=float(args.label_flip_rate),
            aggregation=str(args.aggregation),
            cosine_drop_k=int(args.cosine_drop_k),
            clip_norm=args.clip_norm,
            trim_ratio=float(args.trim_ratio),
            trust_alpha=float(args.trust_alpha),
            trust_beta=float(args.trust_beta),
            trust_gamma=float(args.trust_gamma),
        )

        result = train_fedavg_binary(client_dfs, test_df, config=cfg)

        # Capture the local run directory if present in metrics/history.
        # LocalLogger stores it in runs/<run_id>/; the run_id isn't currently returned,
        # so we infer the latest directory after each run.
        try:
            runs_root = (root / "runs").resolve()
            if runs_root.exists():
                latest = max((p for p in runs_root.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime)
                run_dirs.append(str(latest))
        except Exception:
            pass

        metrics = dict(result.metrics)
        rows.append(
            {
                "n_clients": n_clients,
                "n_attackers": int(k),
                "malicious_clients": ",".join(str(x) for x in mc),
                "aggregation": cfg.aggregation,
                "label_flip_rate": cfg.label_flip_rate,
                "rounds": cfg.rounds,
                "recall": float(metrics.get("recall", 0.0)),
                "false_positive_rate": float(metrics.get("false_positive_rate", 0.0)),
                "accuracy": float(metrics.get("accuracy", 0.0)),
                "f1": float(metrics.get("f1", 0.0)),
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values(["n_attackers"]).reset_index(drop=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"attackers_summary_{ts}.csv"
    df.to_csv(out_path, index=False)

    # Pretty print
    cols = ["n_attackers", "recall", "false_positive_rate", "accuracy", "f1", "aggregation", "label_flip_rate"]
    print(df[cols].to_string(index=False))
    print(f"\nWrote sweep summary: {out_path}")
    if run_dirs:
        print("Run folders (most recent per attacker count):")
        for p in run_dirs:
            print(f"- {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

