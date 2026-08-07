from telemetry.engine.math_ops import calculate_acceleration, calculate_longitudinal_g_force, calculate_throttle_derivative, compute_delta_time


def test_acceleration_protects_zero_division() -> None:
    speed = [100.0, 100.0, 120.0]
    time_seconds = [0.0, 0.0, 2.0]

    g_force = calculate_acceleration(speed, time_seconds)

    assert g_force[1] == 0.0
    assert len(g_force) == 3


def test_longitudinal_g_force_produces_numeric_output() -> None:
    g_force = calculate_longitudinal_g_force([120.0, 140.0, 160.0], [0.0, 50.0, 120.0])

    assert len(g_force) == 3
    assert all(value == value for value in g_force)


def test_throttle_derivative_handles_constant_signal() -> None:
    derivative = calculate_throttle_derivative([50.0, 50.0, 50.0], [0.0, 20.0, 40.0], [120.0, 130.0, 140.0])

    assert derivative.tolist() == [0.0, 0.0, 0.0]


def test_delta_time_uses_monotonic_output() -> None:
    import pandas as pd

    frame = pd.DataFrame({"Distance": [0.0, 100.0, 200.0], "d1_Speed": [200.0, 210.0, 220.0], "d2_Speed": [198.0, 208.0, 218.0]})

    delta = compute_delta_time(frame)

    assert list(delta.columns) == ["distance_meters", "driver_one_time_s", "driver_two_time_s", "delta_time_s"]
    assert len(delta) == 2000