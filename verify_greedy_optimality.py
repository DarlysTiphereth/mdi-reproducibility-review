"""Exhaustive verification that the greedy marginal rule attains the exact
optimum of the integer supply-allocation formulation reported in Section 4.2.

Why this script exists
----------------------
The manuscript states that exhaustive enumeration of all feasible integer
allocations confirms the greedy objective value. Without this script that claim
is not independently checkable from the reproducibility package, which would
leave the paper's central optimisation result unverifiable by a reader.

What is verified
----------------
1. The number of feasible allocations equals C(B + n - 1, n - 1), the count of
   weak compositions of B units over n routes.
2. The greedy objective equals the enumerated integer optimum, to machine
   precision.
3. The marginal gain Delta_i(k) = c_i / [(O_i + k)(O_i + k + 1)] is decreasing
   in k for every route, which is the structural property that makes successive
   selection of the largest marginal gain optimal for this class of separable
   integer allocation problems [Fox, 1966; Federgruen and Groenevelt, 1986].
4. The continuous relaxation optimum is strictly below the integer optimum, so
   the integer result is not confused with the continuous one.

Usage
-----
    python verify_greedy_optimality.py

Runtime is a few seconds for the reported instance (n = 8, B = 10, 19,448
allocations). The enumeration count grows combinatorially; the script refuses to
enumerate beyond ENUMERATION_LIMIT and reports the structural argument instead.
"""

from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import CONFIG
from data_generation import generate_experiment
from optimization import deterministic_allocations, objective

ENUMERATION_LIMIT = 5_000_000
OUTPUT = ROOT / "results" / "greedy_optimality_verification.json"


def weak_compositions(total: int, parts: int):
    """Yield every non-negative integer vector of length `parts` summing to `total`."""
    for cut in itertools.combinations(range(total + parts - 1), parts - 1):
        prev, alloc = -1, []
        for c in cut:
            alloc.append(c - prev - 1)
            prev = c
        alloc.append(total + parts - 1 - prev - 1)
        yield alloc


def continuous_optimum(c: np.ndarray, o: np.ndarray, budget: float) -> float:
    """Objective of the continuous relaxation, by bisection on the KKT multiplier.

    Reported only to make explicit that the integer optimum is a distinct and
    strictly larger quantity. The manuscript claims optimality for the INTEGER
    formulation, not coincidence with this value.
    """
    lo, hi = 1e-18, float((c / np.maximum(o, 1e-12) ** 2).max()) * 10.0
    for _ in range(400):
        mid = (lo + hi) / 2.0
        if np.maximum(0.0, np.sqrt(c / mid) - o).sum() > budget:
            lo = mid
        else:
            hi = mid
    x = np.maximum(0.0, np.sqrt(c / ((lo + hi) / 2.0)) - o)
    return float(np.sum(c / (o + x)))


def main() -> int:
    *_, zones, _stops, _fleet, _mobile, _manifest = generate_experiment(CONFIG)
    critical = zones[zones.is_critical == 1].reset_index(drop=True)

    p = critical.P.to_numpy(float)
    i = critical.I.to_numpy(float)
    o = critical.O.to_numpy(float)
    n = int(p.size)
    budget = int(CONFIG.total_resource_units)

    weights = p / p.sum()
    c = weights * p * i

    greedy = deterministic_allocations(p, i, o, CONFIG)["Greedy marginal allocation"]
    greedy_value = objective(greedy, p, i, o)

    expected_count = math.comb(budget + n - 1, n - 1)
    if expected_count > ENUMERATION_LIMIT:
        print(f"Instance too large to enumerate ({expected_count:,} allocations).")
        return 1

    best_value, best_alloc, counted = math.inf, None, 0
    for alloc in weak_compositions(budget, n):
        counted += 1
        value = objective(np.asarray(alloc, dtype=float), p, i, o)
        if value < best_value:
            best_value, best_alloc = value, list(alloc)

    # Structural property: marginal gains must be decreasing in k for every route.
    decreasing = True
    for idx in range(n):
        gains = [c[idx] / ((o[idx] + k) * (o[idx] + k + 1)) for k in range(budget + 1)]
        if any(gains[k + 1] > gains[k] + 1e-15 for k in range(len(gains) - 1)):
            decreasing = False

    gap = abs(best_value - greedy_value)
    continuous = continuous_optimum(c, o, float(budget))

    result = {
        "instance": {
            "n_critical_routes": n,
            "budget_units": budget,
            "route_ids": critical.route_id.tolist(),
        },
        "enumeration": {
            "allocations_expected": expected_count,
            "allocations_enumerated": counted,
            "count_matches_formula": counted == expected_count,
        },
        "integer_optimum": {
            "objective": best_value,
            "allocation": best_alloc,
        },
        "greedy": {
            "objective": greedy_value,
            "allocation": [int(round(v)) for v in greedy],
        },
        "greedy_equals_integer_optimum": bool(gap < 1e-12),
        "absolute_gap": gap,
        "continuous_relaxation_objective": continuous,
        "integer_minus_continuous": greedy_value - continuous,
        "marginal_gains_decreasing_for_every_route": decreasing,
        "references": [
            "Fox, B. L. (1966). Discrete optimization via marginal analysis. "
            "Management Science, 13(3), 210-216.",
            "Federgruen, A., & Groenevelt, H. (1986). The greedy procedure for "
            "resource allocation problems: necessary and sufficient conditions "
            "for optimality. Operations Research, 34(6), 909-918.",
        ],
        "note": (
            "Optimality is claimed for the integer formulation only. The "
            "continuous relaxation attains a strictly lower objective and is "
            "reported here solely to keep the two quantities distinct."
        ),
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2))

    print(f"critical routes            : {n}")
    print(f"budget                     : {budget} integer units")
    print(f"allocations enumerated     : {counted:,} (formula: {expected_count:,})")
    print(f"integer optimum            : {best_value:.12f}  {best_alloc}")
    print(f"greedy                     : {greedy_value:.12f}")
    print(f"gap                        : {gap:.3e}")
    print(f"marginal gains decreasing  : {decreasing}")
    print(f"continuous relaxation      : {continuous:.12f} (strictly lower, as expected)")
    print(f"\nwrote {OUTPUT.relative_to(ROOT)}")

    ok = (
        counted == expected_count
        and gap < 1e-12
        and decreasing
        and continuous < best_value
    )
    print("VERIFICATION PASSED" if ok else "VERIFICATION FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
