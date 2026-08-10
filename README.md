# Reproducible synthetic MDI experiment

This package is the revised reproducibility experiment prepared in response to
the SBPO reviews. It does not overwrite the supplied notebook.

## Scientific scope

- All 50 route identifiers (`SYN-001` to `SYN-050`), topology, attributes,
  stops, GPS pings, and mobile points are synthetic.
- `P`, `I`, `O`, and `MDI` are calculated by the formulas stated in the
  manuscript. Every distribution and seed is recorded in the execution
  manifest.
- The target is binary and retains the original notebook rule: critical means
  `MDI_raw` above the empirical 0.85 quantile.
- Isolation Forest is evaluated against the preserved anomaly-injection truth.
- SVM, logistic regression, decision tree, and random forest use the same five
  stratified folds. The comparison is a reproducibility check for a synthetic,
  mechanically derived target, not evidence of external predictive validity.
- The original optimizer is correctly named **elite-preserving stochastic
  search**. It changes only `O`, uses 10 resource units, and is compared with
  uniform, vulnerability-proportional, greedy, and random allocations. Random
  search and the stochastic heuristic each receive 1,000 objective evaluations
  per run across 30 paired seeds.

## Run

Create an isolated Python 3.12 environment, install the pinned dependencies,
and execute:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_pipeline.py
python -m unittest discover -s tests -v
```

Outputs are written to `data/`, `results/`, and `figures/`. Tables and figures
are regenerated only from saved result files. `results/execution_manifest.json`
records versions, parameters, source hashes, output hashes, and limitations.

## Availability and licensing

This package is distributed through an anonymized repository for peer-review
reproducibility. No reuse license is granted in this review copy.
