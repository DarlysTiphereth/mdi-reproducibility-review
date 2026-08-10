"""Generate article tables and figures only from saved/open results."""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def create_figures(results_dir: Path, figures_dir: Path):
    figures_dir.mkdir(parents=True, exist_ok=True)
    post = pd.read_csv(results_dir / "zone_intervention_summary.csv").sort_values("MDI_raw")
    x = np.arange(len(post))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(x, post.MDI_raw, label="Baseline", color="#9c2f45", linewidth=2)
    ax.plot(x, post.MDI_raw_post, label="After mean heuristic allocation", color="#245a76", linewidth=2)
    ax.set_xlabel("Critical synthetic routes, ranked by baseline MDI")
    ax.set_ylabel("Raw MDI")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "figure2_mdi_before_after.png", dpi=300)
    plt.close(fig)

    summary = pd.read_csv(results_dir / "optimization_summary.csv").sort_values("objective_mean")
    labels = summary.method.str.replace(" allocation", "", regex=False).str.replace(" stochastic search", " search", regex=False)
    errors = summary.objective_std.fillna(0).to_numpy()
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.barh(labels, summary.objective_mean, xerr=errors, color="#3f6d86", alpha=0.9, capsize=4)
    ax.set_xlabel("Weighted post-allocation MDI objective (lower is better)")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "figure3_optimization_comparison.png", dpi=300)
    plt.close(fig)


def create_article_tables(results_dir: Path):
    anomaly = pd.read_csv(results_dir / "isolation_forest_metrics.csv")
    classifiers = pd.read_csv(results_dir / "classifier_summary.csv")
    optimization = pd.read_csv(results_dir / "optimization_summary.csv")
    anomaly.to_csv(results_dir / "table_anomaly_detection.csv", index=False)
    classifiers.to_csv(results_dir / "table_classifier_comparison.csv", index=False)
    optimization.to_csv(results_dir / "table_optimization_comparison.csv", index=False)

