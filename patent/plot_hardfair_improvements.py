#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_metrics(csv_path: Path):
    rows = {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[row["metric"]] = {
                "ft_mean": float(row["finetuned_mean"]),
                "ft_std": float(row["finetuned_std"]),
                "base_mean": float(row["base_mean"]),
                "base_std": float(row["base_std"]),
                "delta_mean": float(row["delta_mean"]),
                "delta_std": float(row["delta_std"]),
            }
    return rows


def compute_relative_improvements(rows, metrics):
    out = {}
    for m in metrics:
        base = rows[m]["base_mean"]
        delta = rows[m]["delta_mean"]
        if m == "mean_rank":
            # mean_rank is lower-is-better; convert to positive improvement percentage.
            rel = (-delta / base) * 100.0
        else:
            rel = (delta / base) * 100.0
        out[m] = rel
    return out


def plot_figure(rows, out_path: Path, title: str):
    abs_metrics = ["mrr", "map", "ndcg"]
    rel_metrics = ["mean_rank", "mrr", "map", "ndcg"]

    rel = compute_relative_improvements(rows, rel_metrics)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)

    # Panel A: absolute metrics (Base vs Fine-tuned)
    ax0 = axes[0]
    x = np.arange(len(abs_metrics))
    width = 0.34

    base_vals = [rows[m]["base_mean"] for m in abs_metrics]
    base_errs = [rows[m]["base_std"] for m in abs_metrics]
    ft_vals = [rows[m]["ft_mean"] for m in abs_metrics]
    ft_errs = [rows[m]["ft_std"] for m in abs_metrics]

    ax0.bar(
        x - width / 2,
        base_vals,
        width,
        yerr=base_errs,
        capsize=3,
        label="Base",
        color="#9aa0a6",
        edgecolor="black",
        linewidth=0.6,
    )
    ax0.bar(
        x + width / 2,
        ft_vals,
        width,
        yerr=ft_errs,
        capsize=3,
        label="Fine-tuned",
        color="#1f77b4",
        edgecolor="black",
        linewidth=0.6,
    )

    ax0.set_xticks(x)
    ax0.set_xticklabels([m.upper() for m in abs_metrics])
    ax0.set_ylim(0.0, max(ft_vals + base_vals) * 1.25)
    ax0.set_ylabel("Score (higher is better)")
    ax0.set_title("Absolute Retrieval Quality")
    ax0.legend(frameon=False, loc="upper left")

    # Panel B: relative improvements in percentage
    ax1 = axes[1]
    x2 = np.arange(len(rel_metrics))
    rel_vals = [rel[m] for m in rel_metrics]
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in rel_vals]

    bars = ax1.bar(
        x2,
        rel_vals,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
        width=0.62,
    )

    ax1.axhline(0.0, color="black", linewidth=0.8)
    ax1.set_xticks(x2)
    ax1.set_xticklabels(["MeanRank", "MRR", "MAP", "NDCG"])
    ax1.set_ylabel("Relative Improvement (%)")
    ax1.set_title("Fine-tuned vs Base: Relative Gains")

    for bar, val in zip(bars, rel_vals):
        y = bar.get_height()
        offset = 0.2 if y >= 0 else -0.2
        va = "bottom" if y >= 0 else "top"
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            y + offset,
            f"{val:+.2f}%",
            ha="center",
            va=va,
            fontsize=10,
        )

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.text(
        0.5,
        -0.02,
        "Note: MeanRank is transformed as (Base - FT) / Base so positive indicates improvement.",
        ha="center",
        fontsize=10,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot hard-fair retrieval improvements.")
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        required=True,
        help="Path to multiseed_metrics_table.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output PNG file path",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Hard-Fair Retrieval Benchmark (2 Seeds)",
        help="Figure title",
    )

    args = parser.parse_args()
    metric_rows = load_metrics(args.metrics_csv)
    plot_figure(metric_rows, args.out, args.title)
    print(f"Saved: {args.out}")
    print(f"Saved: {args.out.with_suffix('.pdf')}")
