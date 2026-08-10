"""Fleet-supply allocation comparisons using the objective in the prototype."""

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


def _normalize_keys(keys, total):
    keys = np.asarray(keys, dtype=float)
    if keys.sum() <= 0:
        return np.full(keys.size, total / keys.size)
    return total * keys / keys.sum()


def elite_preserving_search(p, inefficiency, supply, cfg: Config, seed: int):
    rng = np.random.default_rng(seed)
    pop = rng.random((cfg.population_size, len(p)))
    best_value, best_allocation, evaluations = np.inf, None, 0
    start = time.perf_counter()
    elite_n = max(1, int(cfg.population_size * cfg.elite_fraction))
    for _ in range(cfg.generations):
        values = []
        allocations = []
        for keys in pop:
            alloc = _normalize_keys(keys, cfg.total_resource_units)
            value = objective(alloc, p, inefficiency, supply)
            evaluations += 1
            values.append(value)
            allocations.append(alloc)
            if value < best_value:
                best_value, best_allocation = value, alloc.copy()
        elite_idx = np.argsort(values)[:elite_n]
        elite = pop[elite_idx]
        fresh = rng.random((cfg.population_size - elite_n, len(p)))
        pop = np.vstack([elite, fresh])
    return best_allocation, best_value, evaluations, time.perf_counter() - start


def random_search(p, inefficiency, supply, cfg: Config, seed: int):
    rng = np.random.default_rng(seed)
    best_value, best_allocation = np.inf, None
    start = time.perf_counter()
    for _ in range(cfg.evaluation_budget):
        alloc = _normalize_keys(rng.random(len(p)), cfg.total_resource_units)
        value = objective(alloc, p, inefficiency, supply)
        if value < best_value:
            best_value, best_allocation = value, alloc.copy()
    return best_allocation, best_value, cfg.evaluation_budget, time.perf_counter() - start


def deterministic_allocations(p, inefficiency, supply, cfg: Config):
    n = len(p)
    uniform = np.full(n, cfg.total_resource_units / n)
    proportional = cfg.total_resource_units * p / p.sum()
    greedy = np.zeros(n)
    # Ten one-unit decisions; unit count comes directly from the prototype's
    # total resource value of 10.
    for _ in range(int(cfg.total_resource_units)):
        candidates = []
        for j in range(n):
            trial = greedy.copy()
            trial[j] += 1.0
            candidates.append(objective(trial, p, inefficiency, supply))
        greedy[int(np.argmin(candidates))] += 1.0
    return {"Uniform allocation": uniform,
            "Vulnerability-proportional allocation": proportional,
            "Greedy marginal allocation": greedy}


def compare_optimization(zones: pd.DataFrame, cfg: Config):
    critical = zones[zones.is_critical == 1].copy().reset_index(drop=True)
    p = critical.P.to_numpy(float)
    ineff = critical.I.to_numpy(float)
    supply = critical.O.to_numpy(float)
    baseline = objective(np.zeros(len(critical)), p, ineff, supply)
    run_rows, allocation_rows = [], []

    for method, alloc in deterministic_allocations(p, ineff, supply, cfg).items():
        start = time.perf_counter()
        value = objective(alloc, p, ineff, supply)
        elapsed = time.perf_counter() - start
        run_rows.append({"method": method, "run": 0, "seed": cfg.seed,
                         "objective": value, "baseline_objective": baseline,
                         "reduction_percent": 100 * (baseline - value) / baseline,
                         "evaluations": 1 if method != "Greedy marginal allocation" else int(cfg.total_resource_units) * len(p),
                         "runtime_seconds": elapsed, "resource_sum": alloc.sum(),
                         "feasible": bool(np.all(alloc >= 0) and np.isclose(alloc.sum(), cfg.total_resource_units))})
        for route_id, amount in zip(critical.route_id, alloc):
            allocation_rows.append({"method": method, "run": 0, "route_id": route_id,
                                    "allocation": amount})

    child_seeds = np.random.SeedSequence(cfg.seed).spawn(cfg.stochastic_runs)
    for run, seed_seq in enumerate(child_seeds, start=1):
        seed = int(seed_seq.generate_state(1)[0])
        for method, fn in [("Elite-preserving stochastic search", elite_preserving_search),
                           ("Random search", random_search)]:
            alloc, value, evaluations, elapsed = fn(p, ineff, supply, cfg, seed)
            run_rows.append({"method": method, "run": run, "seed": seed,
                             "objective": value, "baseline_objective": baseline,
                             "reduction_percent": 100 * (baseline - value) / baseline,
                             "evaluations": evaluations, "runtime_seconds": elapsed,
                             "resource_sum": alloc.sum(),
                             "feasible": bool(np.all(alloc >= 0) and np.isclose(alloc.sum(), cfg.total_resource_units))})
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
        runtime_seconds_mean=("runtime_seconds", "mean"),
        all_feasible=("feasible", "all"),
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

