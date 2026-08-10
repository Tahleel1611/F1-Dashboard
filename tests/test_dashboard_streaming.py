import pandas as pd

from frontend.dashboard import _build_live_stream_figure, _build_live_stream_packets


def test_build_live_stream_packets_returns_fallback_rows() -> None:
    frame = pd.DataFrame(
        {
            "Distance": [0.0, 100.0],
            "d1_Speed": [200.0, 210.0],
            "d2_Speed": [195.0, 205.0],
            "d1_Throttle": [70.0, 65.0],
            "d2_Throttle": [60.0, 58.0],
            "d1_Brake": [0, 1],
            "d2_Brake": [1, 0],
            "d1_Gear": [3, 4],
            "d2_Gear": [4, 5],
            "d1_RPM": [10000, 10500],
            "d2_RPM": [9500, 10200],
            "d1_X": [1.0, 1.2],
            "d1_Y": [2.0, 2.2],
            "d2_X": [1.1, 1.3],
            "d2_Y": [2.1, 2.3],
        }
    )

    packets = _build_live_stream_packets(frame, max_packets=2, driver_one="VER", driver_two="LEC")

    assert len(packets) == 2
    assert packets[0]["driver_one"]["driver_code"] == "VER"
    assert packets[0]["driver_two"]["driver_code"] == "LEC"
    assert packets[0]["driver_one"]["speed_kph"] == 200.0


def test_build_live_stream_figure_creates_driver_traces() -> None:
    packets = [
        {
            "driver_one": {"driver_code": "VER", "speed_kph": 200.0, "distance_m": 0.0},
            "driver_two": {"driver_code": "LEC", "speed_kph": 195.0, "distance_m": 0.0},
        },
        {
            "driver_one": {"driver_code": "VER", "speed_kph": 205.0, "distance_m": 100.0},
            "driver_two": {"driver_code": "LEC", "speed_kph": 200.0, "distance_m": 100.0},
        },
    ]

    figure = _build_live_stream_figure(packets, driver_one="VER", driver_two="LEC")

    assert len(figure.data) == 2
    assert figure.data[0].name == "VER Speed"
    assert figure.data[1].name == "LEC Speed"
