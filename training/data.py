"""Deterministic synthetic hourly electricity-load dataset.

The data is generated, not real. It is shaped like a regional grid load curve
(daily double peak, weekend and holiday dips, heating/cooling response to
temperature) so that training, serving and drift detection can be exercised
end to end without any download or licence question.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES: list[str] = [
    "temperature_c",
    "hour_of_day",
    "day_of_week",
    "is_holiday",
    "wind_speed_ms",
    "cloud_cover",
]
TARGET = "load_mw"

SCENARIOS = ("none", "heatwave")


def make_energy_dataset(n: int = 5000, seed: int = 42, scenario: str = "none") -> pd.DataFrame:
    """Generate ``n`` hourly observations.

    ``scenario="heatwave"`` shifts the input distribution (hotter, calmer, clearer)
    and is what the drift tests and the traffic generator use as "drifted" data.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}, expected one of {SCENARIOS}")
    rng = np.random.default_rng(seed)

    hour = rng.integers(0, 24, n)
    dow = rng.integers(0, 7, n)
    day_of_year = rng.integers(0, 365, n)
    is_holiday = (rng.random(n) < 0.03).astype(int)

    seasonal = 10.0 + 11.0 * np.sin(2 * np.pi * (day_of_year - 110) / 365)
    diurnal = 4.0 * np.sin(2 * np.pi * (hour - 9) / 24)
    temperature = seasonal + diurnal + rng.normal(0, 2.5, n)
    wind = rng.weibull(2.0, n) * 6.0
    cloud = np.clip(rng.beta(2.0, 2.0, n), 0, 1)

    if scenario == "heatwave":
        temperature = temperature + 9.0
        wind = wind * 0.5
        cloud = cloud * 0.4

    daily_profile = 6000 * np.exp(-((hour - 8) ** 2) / 8) + 8000 * np.exp(-((hour - 19) ** 2) / 10)
    weekend = np.where(dow >= 5, -5000, 0)
    holiday = np.where(is_holiday == 1, -7000, 0)
    thermal = 380 * np.abs(temperature - 17.0)
    noise = rng.normal(0, 900, n)
    load = 42000 + daily_profile + weekend + holiday + thermal - 40 * wind * cloud + noise

    return pd.DataFrame(
        {
            "temperature_c": temperature.round(2),
            "hour_of_day": hour,
            "day_of_week": dow,
            "is_holiday": is_holiday,
            "wind_speed_ms": wind.round(2),
            "cloud_cover": cloud.round(3),
            TARGET: load.round(1),
        }
    )
