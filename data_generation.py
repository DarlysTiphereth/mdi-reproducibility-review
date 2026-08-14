"""Transparent synthetic spatial, route, telemetry, and socioeconomic data."""

from __future__ import annotations

from dataclasses import asdict
import math

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from config import Config
from mdi import audited_supply, mobility_desert_index, routing_inefficiency, socioeconomic_pressure


EARTH_RADIUS_M = 6_371_008.8


def haversine_m(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def build_synthetic_network():
    """Build a 5x10 grid with a documented central barrier and two crossings."""
    rows, cols = 5, 10
    lat0, lon0, step = -9.65, -35.75, 0.01
    records = []
    for r in range(rows):
        for c in range(cols):
            records.append({"node": r * cols + c, "row": r, "col": c,
                            "latitude": lat0 + r * step,
                            "longitude": lon0 + c * step})
    nodes = pd.DataFrame(records)
    adjacency = np.full((rows * cols, rows * cols), np.inf, dtype=float)
    np.fill_diagonal(adjacency, 0.0)
    edges = []
    for r in range(rows):
        for c in range(cols):
            u = r * cols + c
            for dr, dc in [(0, 1), (1, 0)]:
                rr, cc = r + dr, c + dc
                if rr >= rows or cc >= cols:
                    continue
                # Synthetic barrier between columns 4 and 5; crossings at rows 1 and 4.
                if dr == 0 and c == 4 and r not in (1, 4):
                    continue
                v = rr * cols + cc
                nu, nv = nodes.iloc[u], nodes.iloc[v]
                w = float(haversine_m(nu.latitude, nu.longitude, nv.latitude, nv.longitude))
                adjacency[u, v] = adjacency[v, u] = w
                edges.append({"u": u, "v": v, "length_m": w})
    return nodes, pd.DataFrame(edges), adjacency


def _path_from_predecessors(pred: np.ndarray, origin: int, destination: int) -> list[int]:
    path = [destination]
    current = destination
    while current != origin:
        current = int(pred[origin, current])
        if current < 0:
            raise RuntimeError(f"No path from {origin} to {destination}")
        path.append(current)
    return list(reversed(path))


def make_routes(nodes: pd.DataFrame, adjacency: np.ndarray, cfg: Config):
    distances, predecessors = dijkstra(csr_matrix(adjacency), directed=False, return_predecessors=True)
    route_rows, path_rows = [], []
    for origin in range(cfg.n_routes):
        destination = (origin * 7 + 13) % cfg.n_routes
        if destination == origin:
            destination = (destination + 11) % cfg.n_routes
        path = _path_from_predecessors(predecessors, origin, destination)
        network_m = float(distances[origin, destination])
        a, b = nodes.iloc[origin], nodes.iloc[destination]
        direct_m = float(haversine_m(a.latitude, a.longitude, b.latitude, b.longitude))
        route_id = f"SYN-{origin + 1:03d}"
        route_rows.append({"route_id": route_id, "origin_node": origin,
                           "destination_node": destination,
                           "network_distance_m": network_m,
                           "geodesic_distance_m": direct_m,
                           "path_nodes": "-".join(map(str, path))})
        for seq, node in enumerate(path):
            path_rows.append({"route_id": route_id, "sequence": seq, "node": node})
    routes = pd.DataFrame(route_rows)
    routes["I"] = routing_inefficiency(routes.network_distance_m, routes.geodesic_distance_m)
    return routes, pd.DataFrame(path_rows)


def _sample_point_on_route(route_path: list[int], nodes: pd.DataFrame, rng: np.random.Generator):
    if len(route_path) < 2:
        n = nodes.iloc[route_path[0]]
        return float(n.latitude), float(n.longitude)
    j = int(rng.integers(0, len(route_path) - 1))
    a, b = nodes.iloc[route_path[j]], nodes.iloc[route_path[j + 1]]
    t = float(rng.random())
    return (float(a.latitude + t * (b.latitude - a.latitude)),
            float(a.longitude + t * (b.longitude - a.longitude)))


def _route_path_map(routes: pd.DataFrame):
    return {r.route_id: [int(x) for x in r.path_nodes.split("-")] for r in routes.itertuples()}


def generate_route_points(routes: pd.DataFrame, nodes: pd.DataFrame, count: int,
                          rng: np.random.Generator, prefix: str) -> pd.DataFrame:
    route_ids = routes.route_id.to_numpy()
    paths = _route_path_map(routes)
    chosen = rng.choice(route_ids, size=count, replace=True)
    rows = []
    for i, route_id in enumerate(chosen):
        lat, lon = _sample_point_on_route(paths[route_id], nodes, rng)
        rows.append({"point_id": f"{prefix}-{i + 1:05d}", "route_id": route_id,
                     "latitude": lat, "longitude": lon})
    return pd.DataFrame(rows)


def generate_experiment(cfg: Config):
    rng = np.random.default_rng(cfg.seed)
    nodes, edges, adjacency = build_synthetic_network()
    routes, route_paths = make_routes(nodes, adjacency, cfg)

    # Synthetic socioeconomic attributes. The exact ranges are inherited from
    # the original notebook; all cells have an explicit area of 1 km2.
    zones = routes[["route_id", "origin_node", "destination_node",
                    "network_distance_m", "geodesic_distance_m", "I"]].copy()
    zones["area_km2"] = 1.0
    zones["population"] = rng.integers(300, 1500, size=cfg.n_routes)
    zones["mean_income"] = rng.uniform(900.0, 15000.0, size=cfg.n_routes)
    p_raw, p_scaled = socioeconomic_pressure(zones.population, zones.area_km2, zones.mean_income)
    zones["population_density_km2"] = zones.population / zones.area_km2
    zones["P_raw"] = p_raw
    zones["P"] = p_scaled

    # Vehicle arrivals are a synthetic Poisson process. Frequency rates retain
    # the original notebook's U(0.5, 5.0) range, now explicitly in vehicles/hour.
    target_frequency = rng.uniform(0.5, 5.0, size=cfg.n_routes)
    supply, entropy, observed_frequency, headway_serialized = [], [], [], []
    for rate in target_frequency:
        t, arrivals = 0.0, []
        while True:
            t += float(rng.exponential(60.0 / rate))
            if t > cfg.observation_window_minutes:
                break
            arrivals.append(t)
        if len(arrivals) < 2:
            # Deterministic edge-case completion to make the announced headway
            # calculation defined; it is recorded in the dataset manifest.
            arrivals = [60.0 / rate, 120.0 / rate]
        headways = np.diff(np.r_[0.0, arrivals])
        freq = len(arrivals) / (cfg.observation_window_minutes / 60.0)
        o, h_irr = audited_supply(freq, headways)
        supply.append(o)
        entropy.append(h_irr)
        observed_frequency.append(freq)
        headway_serialized.append(";".join(f"{x:.6f}" for x in headways))
    zones["frequency_target_per_hour"] = target_frequency
    zones["frequency_observed_per_hour"] = observed_frequency
    zones["H_irregularity"] = entropy
    zones["O"] = supply
    zones["headways_minutes"] = headway_serialized

    mdi_raw, mdi_score = mobility_desert_index(zones.P, zones.I, zones.O)
    zones["MDI_raw"] = mdi_raw
    zones["MDI_score"] = mdi_score
    threshold = float(zones.MDI_raw.quantile(cfg.critical_quantile))
    zones["is_critical"] = (zones.MDI_raw > threshold).astype(int)

    stops = generate_route_points(routes, nodes, cfg.n_stops, rng, "STOP")
    fleet = generate_route_points(routes, nodes, cfg.n_fleet_pings, rng, "GPS")
    fleet["is_injected_anomaly"] = 0
    anomaly_n = int(round(cfg.n_fleet_pings * cfg.anomaly_fraction))
    anomaly_idx = rng.choice(fleet.index.to_numpy(), size=anomaly_n, replace=False)
    fleet.loc[anomaly_idx, "is_injected_anomaly"] = 1
    fleet.loc[anomaly_idx, "latitude"] += rng.normal(0.0, cfg.anomaly_sigma_degrees, size=anomaly_n)
    fleet.loc[anomaly_idx, "longitude"] += rng.normal(0.0, cfg.anomaly_sigma_degrees, size=anomaly_n)

    mobile = generate_route_points(routes, nodes, cfg.n_mobile_points, rng, "MOB")
    sigma_lat = cfg.mobile_noise_metres / 111_320.0
    sigma_lon = cfg.mobile_noise_metres / (111_320.0 * math.cos(math.radians(-9.65)))
    mobile["latitude"] += rng.normal(0.0, sigma_lat, size=cfg.n_mobile_points)
    mobile["longitude"] += rng.normal(0.0, sigma_lon, size=cfg.n_mobile_points)

    manifest = {
        "configuration": cfg.to_dict(),
        "data_status": "fully synthetic; no municipal route or operational data",
        "network": "5x10 grid centered on coordinates present in the original prototype; one synthetic barrier with crossings at rows 1 and 4",
        "direct_distance_method": "Haversine on stored latitude/longitude coordinates with Earth radius 6371008.8 metres; no projected CRS is used",
        "network_distance_method": "Dijkstra shortest path on edges weighted by the same Haversine distance",
        "connectivity": "the generated graph is connected and every reported origin-destination pair has a path; path construction raises an error if a pair is unreachable",
        "route_identifiers": "SYN-001 to SYN-050; synthetic network paths, not official lines",
        "income_distribution": "Uniform(900, 15000), bounds inherited from original notebook",
        "population_distribution": "DiscreteUniform{300,...,1499}, inherited from original notebook",
        "stop_generation": "fixed-count conditional uniform sampling along synthetic route edges",
        "fleet_generation": "fixed-count conditional uniform sampling along synthetic route edges",
        "arrival_generation": "Poisson process through exponential headways over a 180-minute window",
        "anomaly_injection": "5% of fleet pings; independent Normal(0, 0.005 degrees) per coordinate",
        "mobile_noise": "independent Gaussian coordinate noise with local standard deviation of 15 metres",
        "critical_label": "1 when MDI_raw is above the empirical 0.85 quantile; otherwise 0",
        "low_arrival_edge_case": "if fewer than two arrivals occur, two deterministic expected-time arrivals are inserted and remain disclosed",
    }
    return nodes, edges, routes, route_paths, zones, stops, fleet, mobile, manifest
