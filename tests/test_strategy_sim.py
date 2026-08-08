from telemetry.engine.strategy_sim import F1StrategySimulator


def test_strategy_solver_respects_pit_boundaries() -> None:
    simulator = F1StrategySimulator(total_race_laps=57, pit_loss_seconds=23.0)
    best_candidate = simulator.evaluate_grid_strategies(initial_fuel=100.0, track_temp=35.0)

    assert best_candidate.stops in {1, 2}
    assert all(pit_lap not in {1, 57} for pit_lap in best_candidate.pit_laps)
    assert best_candidate.total_race_time > 0.0


def test_one_stop_strategy_returns_valid_tuple() -> None:
    simulator = F1StrategySimulator(total_race_laps=57, pit_loss_seconds=23.0)
    pit_lap, compounds, total_time = simulator.evaluate_one_stop_strategies(initial_fuel=100.0, track_temp=35.0)

    assert pit_lap >= 8 or pit_lap == -1
    assert isinstance(compounds, str)
    assert total_time > 0.0 or total_time == float("inf")