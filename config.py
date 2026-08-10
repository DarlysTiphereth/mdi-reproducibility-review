"""Central configuration for the reproducible synthetic MDI experiment."""

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class Config:
    seed: int = 42
    n_routes: int = 50
    n_stops: int = 3000
    n_fleet_pings: int = 15000
    n_mobile_points: int = 20000
    anomaly_fraction: float = 0.05
    anomaly_sigma_degrees: float = 0.005
    mobile_noise_metres: float = 15.0
    observation_window_minutes: int = 180
    critical_quantile: float = 0.85
    cv_folds: int = 5
    total_resource_units: float = 10.0
    population_size: int = 50
    generations: int = 20
    elite_fraction: float = 0.20
    stochastic_runs: int = 30

    @property
    def evaluation_budget(self) -> int:
        return self.population_size * self.generations

    def to_dict(self):
        data = asdict(self)
        data["evaluation_budget"] = self.evaluation_budget
        return data


CONFIG = Config()

