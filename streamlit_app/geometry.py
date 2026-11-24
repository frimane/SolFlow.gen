from __future__ import annotations

import datetime as dt

from solflow_gen.geometry import day_conditions

from config import CLEARSKY_MODEL


def clearsky_function(lat: float, lon: float, altitude_m: float):
    # None means the Haurwitz curve that solflow_gen.geometry uses by default.
    if CLEARSKY_MODEL == "haurwitz":
        return None
    if CLEARSKY_MODEL != "ineichen":
        raise ValueError("CLEARSKY_MODEL in config.py must be 'ineichen' or 'haurwitz'.")
    import pvlib
    location = pvlib.location.Location(float(lat), float(lon), tz="UTC", altitude=float(altitude_m))

    def ineichen(times, zenith):
        return location.get_clearsky(times, model="ineichen")["ghi"].to_numpy(float)

    return ineichen


def build_conditions(model, lat: float, lon: float, altitude_m: float, start: dt.date, end: dt.date):
    # One conditioning window per start date in [start, end], laid out like the
    # training windows of this checkpoint. No measurements are involved.
    cfg = model.cfg
    return day_conditions(lat, lon, altitude_m, start.isoformat(), end.isoformat(),
                          model.K, int(model.extra["noon_col"]), cfg["data"],
                          forecast_days=model.n_days,
                          altitude_scale_m=float(cfg["site"].get("altitude_scale_m", 2000.0)),
                          clearsky=clearsky_function(lat, lon, altitude_m))
