from __future__ import annotations

import argparse
from pathlib import Path

from nsl_kdd.data import load_nsl_kdd, train_test_paths
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NSL-KDD baseline loader + trainer")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("verify", help="Verify dataset files and print basic stats")

    train_p = sub.add_parser("train", help="Train a baseline classifier and print metrics")
    train_p.add_argument("--binary", action="store_true", help="Binary labels: normal=0, attack=1")

    return p


def main() -> int:
    args = build_parser().parse_args()
    project_root = Path(__file__).resolve().parent

    if args.cmd == "verify":
        return cmd_verify(project_root)

    if args.cmd == "train":
        return cmd_train(project_root, binary=bool(args.binary))

    raise SystemExit("Unknown command")


if __name__ == "__main__":
    raise SystemExit(main())
