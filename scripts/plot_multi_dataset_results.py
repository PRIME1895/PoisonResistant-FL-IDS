#!/usr/bin/env python
"""Generate comparison plots from multi-dataset experiment results.

This script reads the multi-dataset result JSON files and generates:
1. Comparison plots per dataset (Accuracy, Recall, FPR vs aggregation method)
2. Attacker count sensitivity plots
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_multi_dataset_results(results_dir: Path, dataset_name: str) -> Dict:
    """Load the latest results file for a given dataset."""
    results_dir = Path(results_dir)
    result_files = sorted(results_dir.glob(f"{dataset_name}_results_*.json"), reverse=True)
    
    if not result_files:
        raise FileNotFoundError(f"No results found for {dataset_name} in {results_dir}")
    
    latest_file = result_files[0]
    print(f"Loading {dataset_name}: {latest_file.name}")
    
    with open(latest_file) as f:
        return json.load(f)


def plot_dataset_comparison(results: Dict, output_dir: Path) -> None:
    """Generate comparison plots for a single dataset."""
    dataset_name = results["dataset"].upper()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract data
    df_data = []
    for exp in results["experiments"]:
        df_data.append({
            "aggregation": exp["aggregation"],
            "n_attackers": exp["n_attackers"],
            "recall": exp["recall"],
            "fpr": exp["false_positive_rate"],
            "accuracy": exp["accuracy"],
            "f1": exp["f1"],
        })
    
    df = pd.DataFrame(df_data)
    
    # Get unique values
    aggregations = sorted(df["aggregation"].unique())
    attacker_counts = sorted(df["n_attackers"].unique())
    
    print(f"\n{dataset_name} Dataset Summary:")
    print(f"  Aggregation methods: {aggregations}")
    print(f"  Attacker counts: {attacker_counts}")
    print(f"  Total experiments: {len(df)}")
    
    # Color scheme
    colors = {"fedavg": "#1f77b4", "trimmed_mean": "#ff7f0e", "cosine": "#2ca02c", "median": "#d62728"}
    
    # Plot 1: Recall vs Aggregation (grouped by attacker count)
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(aggregations))
    width = 0.35
    
    for i, attacker_count in enumerate(attacker_counts):
        df_filtered = df[df["n_attackers"] == attacker_count]
        recalls = [df_filtered[df_filtered["aggregation"] == agg]["recall"].values[0] 
                  for agg in aggregations]
        ax.bar(x + i * width, recalls, width, label=f"{attacker_count} attackers", alpha=0.8)
    
    ax.set_xlabel("Aggregation Method", fontsize=12, fontweight="bold")
    ax.set_ylabel("Recall (TPR)", fontsize=12, fontweight="bold")
    ax.set_title(f"{dataset_name}: Recall vs Aggregation Method", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * (len(attacker_counts) - 1) / 2)
    ax.set_xticklabels(aggregations)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"multi_dataset_{dataset_name.lower()}_recall.png", dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved: multi_dataset_{dataset_name.lower()}_recall.png")
    plt.close()
    
    # Plot 2: FPR vs Aggregation
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, attacker_count in enumerate(attacker_counts):
        df_filtered = df[df["n_attackers"] == attacker_count]
        fprs = [df_filtered[df_filtered["aggregation"] == agg]["fpr"].values[0] 
               for agg in aggregations]
        ax.bar(x + i * width, fprs, width, label=f"{attacker_count} attackers", alpha=0.8)
    
    ax.set_xlabel("Aggregation Method", fontsize=12, fontweight="bold")
    ax.set_ylabel("False Positive Rate (FPR)", fontsize=12, fontweight="bold")
    ax.set_title(f"{dataset_name}: FPR vs Aggregation Method", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * (len(attacker_counts) - 1) / 2)
    ax.set_xticklabels(aggregations)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"multi_dataset_{dataset_name.lower()}_fpr.png", dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved: multi_dataset_{dataset_name.lower()}_fpr.png")
    plt.close()
    
    # Plot 3: Accuracy vs Aggregation
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, attacker_count in enumerate(attacker_counts):
        df_filtered = df[df["n_attackers"] == attacker_count]
        accs = [df_filtered[df_filtered["aggregation"] == agg]["accuracy"].values[0] 
               for agg in aggregations]
        ax.bar(x + i * width, accs, width, label=f"{attacker_count} attackers", alpha=0.8)
    
    ax.set_xlabel("Aggregation Method", fontsize=12, fontweight="bold")
    ax.set_ylabel("Accuracy", fontsize=12, fontweight="bold")
    ax.set_title(f"{dataset_name}: Accuracy vs Aggregation Method", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * (len(attacker_counts) - 1) / 2)
    ax.set_xticklabels(aggregations)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"multi_dataset_{dataset_name.lower()}_accuracy.png", dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved: multi_dataset_{dataset_name.lower()}_accuracy.png")
    plt.close()
    
    # Plot 4: F1 vs Aggregation
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, attacker_count in enumerate(attacker_counts):
        df_filtered = df[df["n_attackers"] == attacker_count]
        f1s = [df_filtered[df_filtered["aggregation"] == agg]["f1"].values[0] 
              for agg in aggregations]
        ax.bar(x + i * width, f1s, width, label=f"{attacker_count} attackers", alpha=0.8)
    
    ax.set_xlabel("Aggregation Method", fontsize=12, fontweight="bold")
    ax.set_ylabel("F1 Score", fontsize=12, fontweight="bold")
    ax.set_title(f"{dataset_name}: F1 Score vs Aggregation Method", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * (len(attacker_counts) - 1) / 2)
    ax.set_xticklabels(aggregations)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"multi_dataset_{dataset_name.lower()}_f1.png", dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved: multi_dataset_{dataset_name.lower()}_f1.png")
    plt.close()
    
    # Plot 5: Sensitivity to attacker count (Recall)
    fig, ax = plt.subplots(figsize=(10, 6))
    for agg in aggregations:
        df_filtered = df[df["aggregation"] == agg]
        recalls = df_filtered.sort_values("n_attackers")[["n_attackers", "recall"]].values
        ax.plot(recalls[:, 0], recalls[:, 1], marker="o", label=agg, linewidth=2, markersize=8)
    
    ax.set_xlabel("Number of Attackers", fontsize=12, fontweight="bold")
    ax.set_ylabel("Recall (TPR)", fontsize=12, fontweight="bold")
    ax.set_title(f"{dataset_name}: Recall Sensitivity to Attacker Count", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"multi_dataset_{dataset_name.lower()}_sensitivity_recall.png", dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved: multi_dataset_{dataset_name.lower()}_sensitivity_recall.png")
    plt.close()
    
    # Plot 6: Sensitivity to attacker count (FPR)
    fig, ax = plt.subplots(figsize=(10, 6))
    for agg in aggregations:
        df_filtered = df[df["aggregation"] == agg]
        fprs = df_filtered.sort_values("n_attackers")[["n_attackers", "fpr"]].values
        ax.plot(fprs[:, 0], fprs[:, 1], marker="s", label=agg, linewidth=2, markersize=8)
    
    ax.set_xlabel("Number of Attackers", fontsize=12, fontweight="bold")
    ax.set_ylabel("False Positive Rate (FPR)", fontsize=12, fontweight="bold")
    ax.set_title(f"{dataset_name}: FPR Sensitivity to Attacker Count", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"multi_dataset_{dataset_name.lower()}_sensitivity_fpr.png", dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved: multi_dataset_{dataset_name.lower()}_sensitivity_fpr.png")
    plt.close()
    
    # Summary table
    summary_df = df.pivot_table(
        index="aggregation",
        columns="n_attackers",
        values=["recall", "fpr", "accuracy", "f1"],
        aggfunc="first"
    )
    summary_df.to_csv(output_dir / f"multi_dataset_{dataset_name.lower()}_summary.csv")
    print(f"  ✓ Saved: multi_dataset_{dataset_name.lower()}_summary.csv")


def main():
    """Main entry point."""
    results_dir = PROJECT_ROOT / "figures" / "multi_dataset_results"
    output_dir = PROJECT_ROOT / "figures" / "multi_dataset_comparison"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots for CICIDS and UNSW
    for dataset_name in ["cicids", "unsw"]:
        try:
            results = load_multi_dataset_results(results_dir, dataset_name)
            plot_dataset_comparison(results, output_dir)
        except FileNotFoundError as e:
            print(f"⚠ Skipping {dataset_name}: {e}")
    
    print(f"\n✅ All plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
