#!/usr/bin/env python
"""Run comprehensive FL-IDS experiments on multiple datasets (NSL-KDD, UNSW, CICIDS).

This script:
1. Loads train/test data for each dataset
2. Splits training data into 30 non-IID federated clients
3. Runs attacker sweeps for different aggregation methods (5vs10 attackers)
4. Generates comparative analysis across datasets

Usage:
    python scripts/run_multi_dataset_experiments.py [--datasets nsl_kdd,unsw,cicids] [--rounds 5]
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
from datetime import datetime
import pandas as pd

from nsl_kdd.multi_dataset import get_dataset
from nsl_kdd.federated import split_non_iid
from nsl_kdd.torch_fl import FLConfig, train_fedavg_binary
from nsl_kdd.local_logger import LocalLogger

import numpy as np


def run_dataset_experiments(
    dataset_name: str,
    project_root: Path,
    rounds: int = 5,
    n_clients: int = 30,
    client_size: int = 2000,
    aggregation_methods: list[str] = None,
    attacker_counts: list[int] = None,
) -> dict:
    """Run FL experiments on a single dataset."""
    
    if aggregation_methods is None:
        aggregation_methods = ["fedavg", "trimmed_mean", "cosine"]
    if attacker_counts is None:
        attacker_counts = [5, 10]
    
    print(f"\n{'='*80}")
    print(f"Starting experiments for {dataset_name.upper()} dataset")
    print(f"{'='*80}")
    
    # Load dataset
    print(f"\n[1/4] Loading {dataset_name.upper()} dataset...")
    if dataset_name.lower() == 'cicids':
        train_df, test_df = get_dataset(dataset_name, project_root, max_samples_per_file=None)
    else:
        train_df, test_df = get_dataset(dataset_name, project_root)
    
    print(f"  Train: {len(train_df):,} rows, {len(train_df.columns)} columns")
    print(f"  Test: {len(test_df):,} rows")
    print(f"  Label distribution: {train_df['label'].value_counts().to_dict()}")
    
    # Split into clients
    print(f"\n[2/4] Splitting into {n_clients} non-IID clients...")
    
    # Ensure we don't exceed available data
    if len(train_df) < n_clients * client_size:
        print(f"  WARNING: Limited data. Adjusting client_size from {client_size} to {len(train_df) // n_clients}")
        client_size = len(train_df) // n_clients
    
    clients, manifest = split_non_iid(
        train_df,
        n_clients=n_clients,
        client_size=client_size,
        seed=42,
        specs=None,
        keep_family_column=False,
    )
    
    client_dfs = [cdf for cdf in clients]
    print(f"  Created {len(client_dfs)} clients with {client_size} samples each")
    
    # Run experiments
    print(f"\n[3/4] Running experiments with {len(aggregation_methods)} aggregation methods...")
    
    results = {
        "dataset": dataset_name,
        "rounds": rounds,
        "n_clients": n_clients,
        "client_size": client_size,
        "experiments": []
    }
    
    for aggregation in aggregation_methods:
        print(f"\n  --- Testing {aggregation.upper()} aggregation ---")
        
        for k in attacker_counts:
            # Select malicious clients
            rng = np.random.default_rng(42 + 100 * k)
            malicious_ids = sorted(rng.choice(
                np.arange(1, n_clients + 1), 
                size=int(k), 
                replace=False
            ))
            malicious_clients = tuple(int(x) for x in malicious_ids)
            
            print(f"    Running with {k} attackers (ids: {malicious_clients[:3]}...): ", end="", flush=True)
            
            # Build config
            cfg = FLConfig(
                rounds=rounds,
                local_epochs=1,
                batch_size=256,
                lr=1e-3,
                device="cpu",
                seed=42,
                malicious_clients=malicious_clients,
                label_flip_rate=0.3,
                aggregation=aggregation,
                cosine_drop_k=1 if aggregation == "cosine" else 0,
                clip_norm=5 if aggregation in {"trimmed_mean", "median"} else None,
                trim_ratio=0.2 if aggregation in {"trimmed_mean", "median"} else 0.2,
                trust_alpha=1.0,
                trust_beta=0.5,
                trust_gamma=0.5,
            )
            
            # Run training
            result = train_fedavg_binary(client_dfs, test_df, config=cfg)
            
            metrics = dict(result.metrics)
            exp_result = {
                "aggregation": aggregation,
                "n_attackers": k,
                "malicious_clients": list(malicious_clients),
                "recall": float(metrics.get("recall", 0.0)),
                "false_positive_rate": float(metrics.get("false_positive_rate", 0.0)),
                "accuracy": float(metrics.get("accuracy", 0.0)),
                "f1": float(metrics.get("f1", 0.0)),
                "precision": float(metrics.get("precision", 0.0)),
            }
            results["experiments"].append(exp_result)
            
            print(f"Recall={metrics.get('recall', 0):.4f}, FPR={metrics.get('false_positive_rate', 0):.4f}, Acc={metrics.get('accuracy', 0):.4f}")
    
    # Save results
    print(f"\n[4/4] Saving results...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = project_root / "figures" / "multi_dataset_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = results_dir / f"{dataset_name}_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"  Saved to: {results_file}")
    
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Run multi-dataset FL-IDS experiments")
    parser.add_argument(
        "--datasets",
        type=str,
        default="nsl_kdd,unsw,cicids",
        help="Comma-separated dataset names (default: all three)",
    )
    parser.add_argument("--rounds", type=int, default=5, help="FL rounds per experiment")
    parser.add_argument("--n-clients", type=int, default=30, help="Number of federated clients")
    parser.add_argument("--client-size", type=int, default=2000, help="Samples per client")
    
    args = parser.parse_args()
    
    root = Path(__file__).resolve().parents[1]
    datasets = [d.strip() for d in args.datasets.split(",")]
    
    all_results = {}
    
    for dataset in datasets:
        try:
            results = run_dataset_experiments(
                dataset_name=dataset,
                project_root=root,
                rounds=args.rounds,
                n_clients=args.n_clients,
                client_size=args.client_size,
            )
            all_results[dataset] = results
        except Exception as e:
            print(f"\n❌ ERROR processing {dataset}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY - All Datasets Complete")
    print(f"{'='*80}\n")
    
    for dataset, results in all_results.items():
        print(f"\n{dataset.upper()}:")
        exp_df = pd.DataFrame(results["experiments"])
        print(exp_df[["aggregation", "n_attackers", "recall", "false_positive_rate", "accuracy", "f1"]].to_string(index=False))
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

