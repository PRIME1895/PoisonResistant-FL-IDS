from __future__ import annotations

from pathlib import Path

from nsl_kdd.data import detect_delimiter, load_nsl_kdd, peek_num_columns


def verify(path: Path) -> None:
    delim = detect_delimiter(path)
    n_cols = peek_num_columns(path, delim)
    print(f"- {path.name}: delimiter={delim}, columns={n_cols}")

    df = load_nsl_kdd(path)
    print(f"  rows={len(df):,}, cols={len(df.columns)}")
    if "label" in df.columns:
        vc = df["label"].value_counts().head(10)
        print("  top labels:")
        for k, v in vc.items():
            print(f"    {k}: {v:,}")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    train = root / "KDDTrain+.txt"
    test = root / "KDDTest+.txt"

    if not train.exists() or not test.exists():
        raise SystemExit(f"Expected dataset files at {train} and {test}")

    print("NSL-KDD file verification")
    verify(train)
    verify(test)


if __name__ == "__main__":
    main()
