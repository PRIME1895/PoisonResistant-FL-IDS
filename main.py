from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from nsl_kdd.data import load_nsl_kdd, train_test_paths
from nsl_kdd.federated import DEFAULT_5_CLIENT_SPECS, family_distribution, split_non_iid
from nsl_kdd.pipeline import train_and_eval


def _load_env(project_root: Path) -> None:
    """Best-effort load of .env so local runs can enable Mongo logging easily."""

    try:
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=project_root / ".env", override=False)
    except Exception:
        # Optional convenience only.
        return


def _parse_int_list(csv: str) -> list[int]:
    s = str(csv).strip()
    if not s:
        return []
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


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


def cmd_fl_train(
    project_root: Path,
    *,
    clients_dir: str,
    rounds: int,
    local_epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    seed: int,
    malicious_clients: str,
    label_flip_rate: float,
    aggregation: str,
    cosine_drop_k: int,
    clip_norm: float | None,
    trim_ratio: float,
    trust_alpha: float,
    trust_beta: float,
    trust_gamma: float,
) -> int:
    # Lazy import so `torch` is optional unless you run this command.
    from nsl_kdd.torch_fl import FLConfig, train_fedavg_binary

    clients_path = (project_root / clients_dir).resolve()
    if not clients_path.exists() or not clients_path.is_dir():
        raise SystemExit(f"Clients dir not found: {clients_path}. Run `python main.py split-clients` first.")

    client_files = sorted(clients_path.glob("client_*.csv"))
    if not client_files:
        raise SystemExit(f"No client CSVs found in: {clients_path}")

    client_dfs = [pd.read_csv(p) for p in client_files]

    # Load test set from the canonical NSL-KDD file.
    _, test_path = train_test_paths(project_root)
    test_df = load_nsl_kdd(test_path)

    mc = tuple(_parse_int_list(malicious_clients))

    cfg = FLConfig(
        rounds=rounds,
        local_epochs=local_epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        seed=seed,
        malicious_clients=mc,
        label_flip_rate=float(label_flip_rate),
        aggregation=str(aggregation),
        cosine_drop_k=int(cosine_drop_k),
        clip_norm=clip_norm,
        trim_ratio=float(trim_ratio),
        trust_alpha=float(trust_alpha),
        trust_beta=float(trust_beta),
        trust_gamma=float(trust_gamma),
    )

    result = train_fedavg_binary(client_dfs, test_df, config=cfg)

    print(f"FL config: aggregation={cfg.aggregation}, malicious={list(cfg.malicious_clients)}, flip_rate={cfg.label_flip_rate}")
    if cfg.aggregation == "cosine":
        print(f"- cosine_drop_k={cfg.cosine_drop_k}")
    if cfg.clip_norm is not None:
        print(f"- clip_norm={cfg.clip_norm}")
    if cfg.aggregation in {"trimmed_mean", "median"}:
        print(f"- trim_ratio={cfg.trim_ratio}")

    print("\nFedAvg/Robust-FL final metrics:")
    for k, v in result.metrics.items():
        if k == "round":
            continue
        print(f"{k}: {v:.4f}")

    print("\nRound history:")
    for row in result.history:
        r = int(row.get("round", 0))
        extra = []
        if "cosine_sim_mean" in row:
            extra.append(f"cos={row['cosine_sim_mean']:.3f}")
        if "trust_mean" in row:
            extra.append(f"trust={row['trust_mean']:.3f}")
        if "loss_stability_mean" in row:
            extra.append(f"lossStab={row['loss_stability_mean']:.3f}")
        if "cross_layer_mean" in row:
            extra.append(f"xLayer={row['cross_layer_mean']:.3f}")
        if "dropped_clients" in row:
            extra.append(f"dropped={int(row['dropped_clients'])}")
        extra_s = f" ({', '.join(extra)})" if extra else ""
        print(
            f"- round {r}: acc={row['accuracy']:.4f}, f1={row['f1']:.4f}, prec={row['precision']:.4f}, rec={row['recall']:.4f}{extra_s}"
        )

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

    fl_p = sub.add_parser("fl-train", help="Run a simple PyTorch FL simulation over client CSVs")
    fl_p.add_argument("--clients-dir", default="data/clients", help="Directory containing client_*.csv (default: data/clients)")
    fl_p.add_argument("--rounds", type=int, default=5, help="Federated rounds (default: 5)")
    fl_p.add_argument("--local-epochs", type=int, default=1, help="Local epochs per client per round (default: 1)")
    fl_p.add_argument("--batch-size", type=int, default=256, help="Batch size (default: 256)")
    fl_p.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    fl_p.add_argument("--device", default="cpu", help="Torch device, e.g. cpu or cuda (default: cpu)")
    fl_p.add_argument("--seed", type=int, default=1337, help="RNG seed")

    # Phase 6: poisoning
    fl_p.add_argument(
        "--malicious-clients",
        default="",
        help="Comma-separated 1-based client indices to poison, e.g. '2' or '2,5' (default: none)",
    )
    fl_p.add_argument(
        "--label-flip-rate",
        type=float,
        default=0.0,
        help="Fraction of labels to flip for malicious clients (0..1). Default: 0 (no poisoning).",
    )

    # Phase 7: defenses
    fl_p.add_argument(
        "--aggregation",
        default="fedavg",
        choices=["fedavg", "cosine", "trimmed_mean", "median"],
        help="Aggregation/defense strategy (default: fedavg)",
    )
    fl_p.add_argument(
        "--cosine-drop-k",
        type=int,
        default=0,
        help="(aggregation=cosine) Drop K lowest-similarity client updates per round (default: 0)",
    )
    fl_p.add_argument(
        "--clip-norm",
        type=float,
        default=None,
        help="Clip each client update by this L2 norm before aggregation (default: None)",
    )
    fl_p.add_argument(
        "--trim-ratio",
        type=float,
        default=0.2,
        help="(aggregation=trimmed_mean) Coordinate-wise trim ratio in [0,0.49] (default: 0.2)",
    )

    # Phase 7: trust score weights (only used for aggregation=cosine)
    fl_p.add_argument(
        "--trust-alpha",
        type=float,
        default=1.0,
        help="(aggregation=cosine) Trust weight for cosine similarity term (default: 1.0)",
    )
    fl_p.add_argument(
        "--trust-beta",
        type=float,
        default=0.0,
        help="(aggregation=cosine) Trust weight for loss stability term (default: 0.0)",
    )
    fl_p.add_argument(
        "--trust-gamma",
        type=float,
        default=0.5,
        help="(aggregation=cosine) Trust weight for cross-layer consistency term (default: 0.5)",
    )

    return p


def main() -> int:
    args = build_parser().parse_args()
    project_root = Path(__file__).resolve().parent

    _load_env(project_root)

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

    if args.cmd == "fl-train":
        return cmd_fl_train(
            project_root,
            clients_dir=str(args.clients_dir),
            rounds=int(args.rounds),
            local_epochs=int(args.local_epochs),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            device=str(args.device),
            seed=int(args.seed),
            malicious_clients=str(args.malicious_clients),
            label_flip_rate=float(args.label_flip_rate),
            aggregation=str(args.aggregation),
            cosine_drop_k=int(args.cosine_drop_k),
            clip_norm=(float(args.clip_norm) if args.clip_norm is not None else None),
            trim_ratio=float(args.trim_ratio),
            trust_alpha=float(args.trust_alpha),
            trust_beta=float(args.trust_beta),
            trust_gamma=float(args.trust_gamma),
        )

    raise SystemExit("Unknown command")


if __name__ == "__main__":
    raise SystemExit(main())
