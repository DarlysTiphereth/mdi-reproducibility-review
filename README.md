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
  search and the stochastic heuristic each receive 1,000 objective-function
  calls per run across 30 paired seeds. Because retained elites are reevaluated,
  the elite-preserving method explores 810 distinct candidates per run; random
  search explores 1,000.

## Planned Data.Rio validation boundary

[Data.Rio](https://www.data.rio/) is designated as a candidate source for a
separate future validation stage using observed public urban data. No Data.Rio
connector or observed record is part of this peer-review package, and no
Data.Rio record generated any frozen benchmark result reported here or in the
manuscript.

## Reviewer-request mapping

| Request | Implementation evidence |
|---|---|
| Complete MDI formulation | `mdi.py` implements `P`, `I`, `O`, raw MDI, 0-100 normalization, and denominator/zero-range checks. |
| Synthetic-data specification | `data_generation.py` implements the 5 x 10 network, 50 synthetic identifiers, declared sampling distributions, anomaly truth, arrivals, and seed-controlled generation. |
| Objective anomaly metrics | `evaluation.py` retains truth and predictions for all 15,000 points and reports the complete confusion matrix, precision, recall, and F1. |
| Common classifier comparison | `evaluation.py` evaluates SVM, logistic regression, decision tree, and random forest with the same five stratified folds and seed. |
| Allocation baselines | `optimization.py` evaluates uniform, vulnerability-proportional, greedy, random, and elite-preserving allocations under the same ten-unit resource constraint. |
| Exact integer optimum | `verify_greedy_optimality.py` enumerates all 19,448 feasible allocations, confirms the reported greedy optimum, checks decreasing marginal gains, and keeps the continuous relaxation explicitly separate. |

The reviewer requested these comparisons and disclosures; no reviewer prescribed
a particular stochastic optimizer. The implementation therefore complies with
the requested evaluation design while naming the implemented heuristic exactly.

## Run

Create an isolated Python 3.12 environment, install the pinned dependencies,
and execute:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_pipeline.py
python -m unittest discover -s tests -v
sha256sum -c results/SHA256SUMS.txt
```

Outputs are written to `data/`, `results/`, and `figures/`. Tables and figures
are regenerated only from saved result files. `results/execution_manifest.json`
records versions, parameters, source hashes, output hashes, and limitations.
The automated suite contains seven tests. The dedicated optimum-verification
script is invoked by the complete pipeline so that its JSON certificate is
included in the manifest and checksum list. It may also be run separately with
`python verify_greedy_optimality.py`; it exhaustively enumerates the reported
instance rather than using only the smaller instances exercised by the suite.

The repository mirror intentionally omits the pre-generated row-level
coordinate files in `data/` and `results/isolation_forest_predictions.csv`.
They are fully synthetic and are recreated deterministically by
`run_pipeline.py` with `seed=42`; their expected SHA-256 hashes remain recorded
in the manifest and `results/SHA256SUMS.txt`. Aggregate metrics, tables,
figures, the optimum certificate, and tests are included for direct inspection.

## Availability and licensing

This package is distributed through an anonymized repository for peer-review
reproducibility. No reuse license is granted in this review copy.
