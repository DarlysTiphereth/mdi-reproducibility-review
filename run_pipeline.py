"""Execute the complete revised MDI experiment and freeze its outputs."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys

import matplotlib
import numpy as np
import pandas as pd
import scipy
import sklearn

from config import CONFIG
from data_generation import generate_experiment
from evaluation import compare_classifiers, evaluate_isolation_forest
from optimization import compare_optimization
from reporting import create_article_tables, create_figures


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_safe(value):
    """Convert NumPy scalars and non-finite floats to strict JSON values."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main():
    for directory in (DATA, RESULTS, FIGURES):
        directory.mkdir(parents=True, exist_ok=True)

    nodes, edges, routes, paths, zones, stops, fleet, mobile, data_manifest = generate_experiment(CONFIG)
    for name, frame in [("network_nodes", nodes), ("network_edges", edges),
                        ("synthetic_routes", routes), ("route_paths", paths),
                        ("zones_mdi", zones), ("synthetic_stops", stops),
                        ("fleet_pings_with_truth", fleet), ("synthetic_mobile_points", mobile)]:
        frame.to_csv(DATA / f"{name}.csv", index=False)

    detected, anomaly_metrics = evaluate_isolation_forest(fleet, CONFIG)
    detected.to_csv(RESULTS / "isolation_forest_predictions.csv", index=False)
    anomaly_metrics.to_csv(RESULTS / "isolation_forest_metrics.csv", index=False)

    classifier_folds, classifier_summary, classifier_confusions = compare_classifiers(zones, CONFIG)
    classifier_folds.to_csv(RESULTS / "classifier_fold_metrics.csv", index=False)
    classifier_summary.to_csv(RESULTS / "classifier_summary.csv", index=False)
    classifier_confusions.to_csv(RESULTS / "classifier_confusion_matrices.csv", index=False)

    opt_runs, opt_allocations, opt_summary, post = compare_optimization(zones, CONFIG)
    opt_runs.to_csv(RESULTS / "optimization_runs.csv", index=False)
    opt_allocations.to_csv(RESULTS / "optimization_allocations.csv", index=False)
    opt_summary.to_csv(RESULTS / "optimization_summary.csv", index=False)
    post.to_csv(RESULTS / "zone_intervention_summary.csv", index=False)

    create_article_tables(RESULTS)
    create_figures(RESULTS, FIGURES)

    run_summary = json_safe({
        "anomaly": anomaly_metrics.to_dict(orient="records"),
        "classifiers": classifier_summary.to_dict(orient="records"),
        "optimization": opt_summary.to_dict(orient="records"),
        "critical_before": int(post.is_critical.sum()),
        "critical_after": int(post.is_critical_post.sum()),
    })
    (RESULTS / "run_stdout.json").write_text(
        json.dumps(run_summary, indent=2, allow_nan=False), encoding="utf-8"
    )

    script_files = sorted(list(ROOT.glob("*.py")) + list((ROOT / "tests").glob("*.py"))
                          + [ROOT / "README.md", ROOT / "requirements.txt"])
    manifest = {
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "versions": {"numpy": np.__version__, "pandas": pd.__version__,
                     "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
                     "matplotlib": matplotlib.__version__},
        "configuration": CONFIG.to_dict(),
        "data_generation": data_manifest,
        "source_hashes": {p.name: sha256(p) for p in script_files},
        "outputs": {},
        "limitations": [
            "All routes, socioeconomic attributes, stops, telemetry, and demand points are synthetic.",
            "The experiment does not estimate real waiting-time, social, or operational impacts.",
            "The binary target is mechanically derived from the MDI and therefore classifier results do not establish external predictive validity.",
            "The allocation method changes only O (audited supply); it does not optimize itineraries or topology.",
            "The elite-preserving search is not a BRKGA because it has no biased crossover or mutants mechanism.",
        ],
    }
    frozen_outputs = sorted(
        list(DATA.glob("*.csv"))
        + list(RESULTS.glob("*.csv"))
        + [RESULTS / "run_stdout.json"]
        + list(FIGURES.glob("*.png"))
    )
    for path in frozen_outputs:
        manifest["outputs"][str(path.relative_to(ROOT))] = sha256(path)
    (RESULTS / "execution_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    checksum_paths = frozen_outputs + [RESULTS / "execution_manifest.json"]
    checksum_lines = [
        f"{sha256(path)}  {path.relative_to(ROOT)}" for path in sorted(checksum_paths)
    ]
    (RESULTS / "SHA256SUMS.txt").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    print(json.dumps(run_summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
