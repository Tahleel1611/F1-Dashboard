"""Deterministic one-stop race strategy search using the tyre degradation model."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from models.inference.tyre_model import F1TyrePredictor


class F1StrategySimulator:
    def __init__(self, total_race_laps: int = 57, pit_loss_seconds: float = 23.0):
        self.total_race_laps = total_race_laps
        self.pit_loss_seconds = pit_loss_seconds
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
        return float(np.sum(projected_laps))

    def _evaluate_strategy(self, pit_lap: int, start_compound: int, end_compound: int, initial_fuel: float, track_temp: float) -> float:
        first_stint_time = self.simulate_stint_pace(pit_lap, start_compound, initial_fuel, track_temp)
        remaining_fuel = max(initial_fuel - (pit_lap * 1.6), 0.0)
        second_stint_length = self.total_race_laps - pit_lap
        second_stint_time = self.simulate_stint_pace(second_stint_length, end_compound, remaining_fuel, track_temp)
        return first_stint_time + self.pit_loss_seconds + second_stint_time

    def evaluate_one_stop_strategies(self, initial_fuel: float, track_temp: float = 35.0) -> Tuple[int, str, float]:
        best_total_time = float("inf")
        best_pit_lap = -1
        best_compound_sequence = ""

        candidate_compounds = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
        for pit_lap in range(8, self.total_race_laps - 7):
            for start_compound, end_compound in candidate_compounds:
                total_race_time = self._evaluate_strategy(pit_lap, start_compound, end_compound, initial_fuel, track_temp)
                if total_race_time < best_total_time:
                    best_total_time = total_race_time
                    best_pit_lap = pit_lap
                    best_compound_sequence = f"{start_compound} -> {end_compound}"

        return best_pit_lap, best_compound_sequence, best_total_time


if __name__ == "__main__":
    simulator = F1StrategySimulator()
    pit_lap, compound_plan, total_time_seconds = simulator.evaluate_one_stop_strategies(100.0)
    print(pit_lap, compound_plan, total_time_seconds)