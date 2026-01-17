from __future__ import annotations

import argparse
import json
from pathlib import Path

from nsl_kdd.data import load_nsl_kdd
from nsl_kdd.federated import DEFAULT_5_CLIENT_SPECS, family_distribution, split_non_iid


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Split NSL-KDD training set into non-IID federated clients")
    p.add_argument("--out", type=str, default="data/clients", help="Output directory (default: data/clients)")
    p.add_argument("--n-clients", type=int, default=5, help="Number of clients (default: 5)")
    p.add_argument(
        "--client-size",
        type=int,
        default=20000,
        help="Rows per client (default: 20000; set smaller for quick demos)",
    )
    p.add_argument("--seed", type=int, default=1337, help="RNG seed")
    return p


def main() -> int:
    args = build_parser().parse_args()

    root = Path(__file__).resolve().parents[1]
    train_path = root / "KDDTrain+.txt"
    if not train_path.exists():
        raise SystemExit(f"Missing: {train_path}")

    out_dir = (root / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_nsl_kdd(train_path)

    specs = DEFAULT_5_CLIENT_SPECS if args.n_clients == 5 else None
    clients, manifest = split_non_iid(
        train_df,
        n_clients=args.n_clients,
        client_size=args.client_size,
        seed=args.seed,
        specs=specs,
        keep_family_column=False,
    )

    # Write client CSVs and build distributions.
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
        "output_dir": str(out_dir.relative_to(root)),
        "clients": dist,
    }

    (out_dir / "manifest.json").write_text(json.dumps(manifest_out, indent=2), encoding="utf-8")

    print(f"Wrote {len(clients)} client files to: {out_dir}")
    for k, v in dist.items():
        print(f"- {k}: rows={v['rows']}, families={v['family_counts']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
