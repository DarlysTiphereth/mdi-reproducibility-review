"""Integer fleet-supply allocation comparisons for the reported benchmark."""

from __future__ import annotations

import time
import numpy as np
import pandas as pd

from config import Config


def objective(allocation, p, inefficiency, supply):
    allocation = np.asarray(allocation, dtype=float)
    if np.any(allocation < -1e-10):
        raise ValueError("Allocation must be nonnegative")
    weights = p / p.sum()
    return float(np.sum(weights * ((p * inefficiency) / (supply + allocation))))


def _integer_budget(total_resource_units: float) -> int:
    budget = int(round(total_resource_units))
    if not np.isclose(total_resource_units, budget):
        raise ValueError("The reported allocation model requires an integer budget")
    return budget


def _uniform_integer_composition(rng: np.random.Generator, total: int,
                                 parts: int) -> np.ndarray:
    """Sample uniformly from all nonnegative integer vectors summing to total.

    A weak composition is represented by choosing ``parts - 1`` bar positions
    from ``total + parts - 1`` slots (the stars-and-bars bijection). Because the
    bar sets are sampled uniformly, every feasible integer allocation has the
    same probability.
    """
    if total < 0 or parts <= 0:
        raise ValueError("total must be nonnegative and parts must be positive")
    if parts == 1:
        return np.array([total], dtype=int)
    bars = np.sort(rng.choice(total + parts - 1, size=parts - 1, replace=False))
    boundaries = np.concatenate(([-1], bars, [total + parts - 1]))
    return np.diff(boundaries) - 1


def _integer_feasible(allocation: np.ndarray, budget: int) -> bool:
    allocation = np.asarray(allocation, dtype=float)
    return bool(
        np.all(allocation >= 0)
        and np.allclose(allocation, np.rint(allocation))
        and np.isclose(allocation.sum(), budget)
    )


def elite_preserving_search(p, inefficiency, supply, cfg: Config, seed: int):
    rng = np.random.default_rng(seed)
    budget = _integer_budget(cfg.total_resource_units)
    pop = np.vstack([
        _uniform_integer_composition(rng, budget, len(p))
        for _ in range(cfg.population_size)
    ])
    best_value, best_allocation, evaluations = np.inf, None, 0
    distinct_candidates: set[tuple[int, ...]] = set()
    start = time.perf_counter()
    elite_n = max(1, int(cfg.population_size * cfg.elite_fraction))
    for _ in range(cfg.generations):
        values = []
        for alloc in pop:
            value = objective(alloc, p, inefficiency, supply)
            evaluations += 1
            distinct_candidates.add(tuple(int(v) for v in alloc))
            values.append(value)
            if value < best_value:
                best_value, best_allocation = value, alloc.copy()
        elite_idx = np.argsort(values, kind="stable")[:elite_n]
        elite = pop[elite_idx]
        fresh = np.vstack([
            _uniform_integer_composition(rng, budget, len(p))
            for _ in range(cfg.population_size - elite_n)
        ])
        pop = np.vstack([elite, fresh])
    return (best_allocation, best_value, evaluations, len(distinct_candidates),
            time.perf_counter() - start)


def random_search(p, inefficiency, supply, cfg: Config, seed: int):
    rng = np.random.default_rng(seed)
    budget = _integer_budget(cfg.total_resource_units)
    best_value, best_allocation = np.inf, None
    distinct_candidates: set[tuple[int, ...]] = set()
    start = time.perf_counter()
    for _ in range(cfg.evaluation_budget):
        alloc = _uniform_integer_composition(rng, budget, len(p))
        distinct_candidates.add(tuple(int(v) for v in alloc))
        value = objective(alloc, p, inefficiency, supply)
        if value < best_value:
            best_value, best_allocation = value, alloc.copy()
    return (best_allocation, best_value, cfg.evaluation_budget,
            len(distinct_candidates), time.perf_counter() - start)


def deterministic_allocations(p, inefficiency, supply, cfg: Config):
    n = len(p)
    budget = _integer_budget(cfg.total_resource_units)

    # Integer uniform baseline: distribute the quotient to every route, then
    # place the remainder in stable route order.
    uniform = np.full(n, budget // n, dtype=int)
    uniform[: budget % n] += 1

    # Integer vulnerability-proportional baseline: Hamilton (largest-remainder)
    # apportionment, with stable route-order tie breaking.
    quotas = budget * np.asarray(p, dtype=float) / np.sum(p)
    proportional = np.floor(quotas).astype(int)
    remainder = budget - int(proportional.sum())
    order = np.argsort(-(quotas - proportional), kind="stable")
    proportional[order[:remainder]] += 1

    greedy = np.zeros(n, dtype=int)
    for _ in range(budget):
        candidates = []
        for j in range(n):
            trial = greedy.copy()
            trial[j] += 1
            candidates.append(objective(trial, p, inefficiency, supply))
        greedy[int(np.argmin(candidates))] += 1
    return {"Uniform allocation": uniform,
            "Vulnerability-proportional allocation": proportional,
            "Greedy marginal allocation": greedy}


def compare_optimization(zones: pd.DataFrame, cfg: Config):
    critical = zones[zones.is_critical == 1].copy().reset_index(drop=True)
    p = critical.P.to_numpy(float)
    ineff = critical.I.to_numpy(float)
    supply = critical.O.to_numpy(float)
    baseline = objective(np.zeros(len(critical)), p, ineff, supply)
    budget = _integer_budget(cfg.total_resource_units)
    run_rows, allocation_rows = [], []

    for method, alloc in deterministic_allocations(p, ineff, supply, cfg).items():
        start = time.perf_counter()
        value = objective(alloc, p, ineff, supply)
        elapsed = time.perf_counter() - start
        run_rows.append({"method": method, "run": 0, "seed": cfg.seed,
                         "objective": value, "baseline_objective": baseline,
                         "reduction_percent": 100 * (baseline - value) / baseline,
                         "evaluations": 1 if method != "Greedy marginal allocation" else int(cfg.total_resource_units) * len(p),
                         "distinct_candidates": 1 if method != "Greedy marginal allocation" else int(cfg.total_resource_units) * len(p),
                         "runtime_seconds": elapsed, "resource_sum": alloc.sum(),
                         "feasible": _integer_feasible(alloc, budget),
                         "integer_feasible": _integer_feasible(alloc, budget)})
        for route_id, amount in zip(critical.route_id, alloc):
            allocation_rows.append({"method": method, "run": 0, "route_id": route_id,
                                    "allocation": amount})

    child_seeds = np.random.SeedSequence(cfg.seed).spawn(cfg.stochastic_runs)
    for run, seed_seq in enumerate(child_seeds, start=1):
        seed = int(seed_seq.generate_state(1)[0])
        for method, fn in [("Elite-preserving stochastic search", elite_preserving_search),
                           ("Random search", random_search)]:
            alloc, value, evaluations, distinct, elapsed = fn(p, ineff, supply, cfg, seed)
            run_rows.append({"method": method, "run": run, "seed": seed,
                             "objective": value, "baseline_objective": baseline,
                             "reduction_percent": 100 * (baseline - value) / baseline,
                             "evaluations": evaluations,
                             "distinct_candidates": distinct,
                             "runtime_seconds": elapsed,
                             "resource_sum": alloc.sum(),
                             "feasible": _integer_feasible(alloc, budget),
                             "integer_feasible": _integer_feasible(alloc, budget)})
            for route_id, amount in zip(critical.route_id, alloc):
                allocation_rows.append({"method": method, "run": run,
                                        "route_id": route_id, "allocation": amount})

    runs = pd.DataFrame(run_rows)
    allocations = pd.DataFrame(allocation_rows)
    summary = runs.groupby("method", as_index=False).agg(
        runs=("run", "count"), objective_mean=("objective", "mean"),
        objective_std=("objective", "std"),
        reduction_percent_mean=("reduction_percent", "mean"),
        reduction_percent_std=("reduction_percent", "std"),
        evaluations_mean=("evaluations", "mean"),
        distinct_candidates_mean=("distinct_candidates", "mean"),
        runtime_seconds_mean=("runtime_seconds", "mean"),
        all_feasible=("feasible", "all"),
        all_integer_feasible=("integer_feasible", "all"),
    )

    eps = allocations[allocations.method == "Elite-preserving stochastic search"]
    mean_alloc = eps.groupby("route_id", as_index=False).allocation.mean()
    post = critical.merge(mean_alloc, on="route_id", how="left")
    post["O_post"] = post.O + post.allocation
    post["MDI_raw_post"] = post.P * post.I / post.O_post
    threshold = float(zones.MDI_raw.quantile(cfg.critical_quantile))
    post["is_critical_post"] = (post.MDI_raw_post > threshold).astype(int)
    post["crossed_below_baseline_threshold"] = ((post.is_critical == 1) & (post.is_critical_post == 0)).astype(int)
    return runs, allocations, summary, post
