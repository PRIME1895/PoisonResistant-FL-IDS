from __future__ import annotations

import argparse
import json
from pathlib import Path

from nsl_kdd.data import load_nsl_kdd, train_test_paths
from nsl_kdd.federated import DEFAULT_5_CLIENT_SPECS, family_distribution, split_non_iid
from nsl_kdd.pipeline import train_and_eval


def cmd_verify(project_root: Path) -> int:
    train_path, test_path = train_test_paths(project_root)

    if not train_path.exists():
        print(f"Missing: {train_path}")
        return 2
    if not test_path.exists():
        print(f"Missing: {test_path}")
        return 2

    train_df = load_nsl_kdd(train_path)
    test_df = load_nsl_kdd(test_path)

    print("Found NSL-KDD files:")
    print(f"- {train_path.name}: rows={len(train_df):,}, cols={len(train_df.columns)}")
    print(f"- {test_path.name}:  rows={len(test_df):,}, cols={len(test_df.columns)}")

    if "label" in train_df.columns:
        print("\nTrain label distribution (top 10):")
        print(train_df["label"].value_counts().head(10).to_string())

    return 0


def cmd_train(project_root: Path, *, binary: bool) -> int:
    train_path, test_path = train_test_paths(project_root)
    train_df = load_nsl_kdd(train_path)
    test_df = load_nsl_kdd(test_path)

    result = train_and_eval(train_df, test_df, binary=binary)

    print(f"Trained on {result.n_train:,} rows; evaluated on {result.n_test:,} rows")
    for k, v in result.metrics.items():
        print(f"{k}: {v:.4f}")

    return 0


def cmd_split_clients(project_root: Path, *, out: str, n_clients: int, client_size: int, seed: int) -> int:
    train_path, _ = train_test_paths(project_root)
    train_df = load_nsl_kdd(train_path)

    out_dir = (Path(project_root) / out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = DEFAULT_5_CLIENT_SPECS if n_clients == 5 else None
    clients, manifest = split_non_iid(
        train_df,
        n_clients=n_clients,
        client_size=client_size,
        seed=seed,
        specs=specs,
        keep_family_column=False,
    )

    dist = {}
    for i, cdf in enumerate(clients, start=1):
        name = f"client_{i}"
        client_path = out_dir / f"{name}.csv"
        cdf.to_csv(client_path, index=False)
        dist[name] = {
            "rows": int(len(cdf)),
            "family_counts": family_distribution(cdf),
        }

    manifest_out = {
        **manifest,
        "source": str(train_path.name),
        "output_dir": str(out_dir),
        "clients": dist,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest_out, indent=2), encoding="utf-8")

    print(f"Wrote {len(clients)} clients to: {out_dir}")
    for k, v in dist.items():
        print(f"- {k}: rows={v['rows']}, families={v['family_counts']}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NSL-KDD baseline loader + trainer")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("verify", help="Verify dataset files and print basic stats")

    train_p = sub.add_parser("train", help="Train a baseline classifier and print metrics")
    train_p.add_argument("--binary", action="store_true", help="Binary labels: normal=0, attack=1")

    split_p = sub.add_parser("split-clients", help="Split training set into non-IID federated client CSVs")
    split_p.add_argument("--out", default="data/clients", help="Output dir (default: data/clients)")
    split_p.add_argument("--n-clients", type=int, default=5, help="Number of clients (default: 5)")
    split_p.add_argument("--client-size", type=int, default=20000, help="Rows per client (default: 20000)")
    split_p.add_argument("--seed", type=int, default=1337, help="RNG seed")

    return p


def main() -> int:
    args = build_parser().parse_args()
    project_root = Path(__file__).resolve().parent

    if args.cmd == "verify":
        return cmd_verify(project_root)

    if args.cmd == "train":
        return cmd_train(project_root, binary=bool(args.binary))

    if args.cmd == "split-clients":
        return cmd_split_clients(
            project_root,
            out=str(args.out),
            n_clients=int(args.n_clients),
            client_size=int(args.client_size),
            seed=int(args.seed),
        )

    raise SystemExit("Unknown command")


if __name__ == "__main__":
    raise SystemExit(main())
