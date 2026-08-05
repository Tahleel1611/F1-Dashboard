"""Deterministic one-stop race strategy search using the tyre degradation model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from models.inference.tyre_model import F1TyrePredictor


@dataclass(frozen=True)
class StrategyCandidate:
    pit_laps: tuple[int, ...]
    compounds: tuple[int, ...]
    stops: int
    total_race_time: float



class F1StrategySimulator:
    def __init__(self, total_race_laps: int = 57, pit_loss_seconds: float = 23.0, fuel_burn_kg_per_lap: float = 1.6, track_evolution_per_lap: float = 0.0):
        self.total_race_laps = total_race_laps
        self.pit_loss_seconds = pit_loss_seconds
        self.fuel_burn_kg_per_lap = fuel_burn_kg_per_lap
        self.track_evolution_per_lap = track_evolution_per_lap
        self.predictor = F1TyrePredictor()
        self.predictor.ensure_trained()

    def simulate_stint_pace(self, stint_length: int, compound: int, initial_fuel: float, track_temp: float = 35.0) -> float:
        if stint_length <= 0:
            return 0.0
        projected_laps = self.predictor.predict_stint_trajectory(
            start_lap=1,
            end_lap=stint_length,
            track_temp=track_temp,
            fuel_load=initial_fuel,
            compound=compound,
        )
        evolution_bonus = np.linspace(0.0, self.track_evolution_per_lap * max(stint_length - 1, 0), stint_length)
        return float(np.sum(projected_laps - evolution_bonus))

    def _evaluate_strategy(self, pit_lap: int, start_compound: int, end_compound: int, initial_fuel: float, track_temp: float) -> float:
        first_stint_time = self.simulate_stint_pace(pit_lap, start_compound, initial_fuel, track_temp)
        remaining_fuel = max(initial_fuel - (pit_lap * self.fuel_burn_kg_per_lap), 0.0)
        second_stint_length = self.total_race_laps - pit_lap
        second_stint_time = self.simulate_stint_pace(second_stint_length, end_compound, remaining_fuel, track_temp)
        return first_stint_time + self.pit_loss_seconds + second_stint_time

    def _evaluate_two_stop_strategy(self, first_pit_lap: int, second_pit_lap: int, compounds: tuple[int, int, int], initial_fuel: float, track_temp: float) -> float:
        if first_pit_lap <= 1 or second_pit_lap >= self.total_race_laps or second_pit_lap <= first_pit_lap + 1:
            return float("inf")

        first_stint_time = self.simulate_stint_pace(first_pit_lap, compounds[0], initial_fuel, track_temp)
        fuel_after_first = max(initial_fuel - (first_pit_lap * self.fuel_burn_kg_per_lap), 0.0)
        second_stint_length = second_pit_lap - first_pit_lap
        second_stint_time = self.simulate_stint_pace(second_stint_length, compounds[1], fuel_after_first, track_temp)
        fuel_after_second = max(fuel_after_first - (second_pit_lap * self.fuel_burn_kg_per_lap), 0.0)
        third_stint_length = self.total_race_laps - second_pit_lap
        third_stint_time = self.simulate_stint_pace(third_stint_length, compounds[2], fuel_after_second, track_temp)
        return first_stint_time + (2 * self.pit_loss_seconds) + second_stint_time + third_stint_time

    def evaluate_grid_strategies(
        self,
        initial_fuel: float,
        track_temp: float = 35.0,
        candidate_compound_sets: tuple[tuple[int, ...], ...] | None = None,
    ) -> StrategyCandidate:
        best_candidate = StrategyCandidate(pit_laps=(), compounds=(), stops=1, total_race_time=float("inf"))

        one_stop_candidates = tuple(candidate for candidate in ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)) if len(candidate) == 2)
        two_stop_candidates = tuple(candidate for candidate in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)) if len(candidate) == 3)
        if candidate_compound_sets:
            one_stop_candidates = tuple(candidate for candidate in candidate_compound_sets if len(candidate) == 2) or one_stop_candidates
            two_stop_candidates = tuple(candidate for candidate in candidate_compound_sets if len(candidate) == 3) or two_stop_candidates

        for pit_lap in range(8, self.total_race_laps - 7):
            for start_compound, end_compound in one_stop_candidates:
                total_race_time = self._evaluate_strategy(pit_lap, start_compound, end_compound, initial_fuel, track_temp)
                if total_race_time < best_candidate.total_race_time:
                    best_candidate = StrategyCandidate(pit_laps=(pit_lap,), compounds=(start_compound, end_compound), stops=1, total_race_time=total_race_time)

        for first_pit_lap in range(8, self.total_race_laps - 14):
            for second_pit_lap in range(first_pit_lap + 6, self.total_race_laps - 6):
                for compounds in two_stop_candidates:
                    total_race_time = self._evaluate_two_stop_strategy(first_pit_lap, second_pit_lap, compounds, initial_fuel, track_temp)
                    if total_race_time < best_candidate.total_race_time:
                        best_candidate = StrategyCandidate(pit_laps=(first_pit_lap, second_pit_lap), compounds=compounds, stops=2, total_race_time=total_race_time)

        return best_candidate

    def evaluate_one_stop_strategies(self, initial_fuel: float, track_temp: float = 35.0) -> Tuple[int, str, float]:
        best_candidate = self.evaluate_grid_strategies(initial_fuel, track_temp)
        if not best_candidate.pit_laps:
            return -1, "", float("inf")
        return best_candidate.pit_laps[0], " -> ".join(str(compound) for compound in best_candidate.compounds), best_candidate.total_race_time


if __name__ == "__main__":
    simulator = F1StrategySimulator()
    pit_lap, compound_plan, total_time_seconds = simulator.evaluate_one_stop_strategies(100.0)
    print(pit_lap, compound_plan, total_time_seconds)