# Sun geometry and the day grid, so conditioning inputs can be built for any site
# and date without measurements. This is what lets the generator run at places
# with no data.
#
# Note that the model was trained on the clear-sky column of your own data. The
# Haurwitz curve used by default here can differ from it, which shifts the
# clear-sky channel. Pass the clear-sky model you trained with through clearsky=.
from __future__ import annotations

import datetime as _dt
from typing import Any, Callable

import numpy as np
import pandas as pd

from .data.geo import site_vector


def solar_zenith_deg(times: pd.DatetimeIndex, lat: float, lon: float) -> np.ndarray:
    # Spencer (1971) declination and equation of time, then the usual zenith formula.
    doy = times.dayofyear.to_numpy()
    g = 2.0 * np.pi * (doy - 1) / 365.0
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
            + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    eot = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                    - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    utc_h = times.hour.to_numpy() + times.minute.to_numpy() / 60.0
    solar_h = utc_h + lon / 15.0 + eot / 60.0
    ha = np.deg2rad(15.0 * (solar_h - 12.0))
    la = np.deg2rad(lat)
    cosz = np.sin(la) * np.sin(decl) + np.cos(la) * np.cos(decl) * np.cos(ha)
    return np.rad2deg(np.arccos(np.clip(cosz, -1.0, 1.0)))


def haurwitz_ghi(zenith_deg: np.ndarray) -> np.ndarray:
    cosz = np.cos(np.deg2rad(zenith_deg))
    out = np.zeros_like(cosz)
    up = cosz > 0.01
    out[up] = 1098.0 * cosz[up] * np.exp(-0.057 / cosz[up])
    return out


def utc_offset_hours(longitude: float) -> int:
    # Whole-hour offset that puts solar noon near 12:00 local; it matches the rule
    # used when the training windows were built from the raw files.
    return int(round(float(longitude) / 15.0))


def day_conditions(latitude: float, longitude: float, altitude: float, start: str, end: str,
                   K: int, noon_col: int, data_cfg: dict[str, Any], forecast_days: int = 1,
                   altitude_scale_m: float = 2000.0,
                   clearsky: Callable[[pd.DatetimeIndex, np.ndarray], np.ndarray] | None = None
                   ) -> dict[str, np.ndarray]:
    # One window per start date in [start, end]. Returns fut_zen, fut_mask, fut_ghi_cs
    # and fut_time (UTC, datetime64), each [N, forecast_days * K], plus date_ord [N]
    # and site_coords [N, 4]. The layout is the same as in the training archive
    # (columns centred on each day's minimum-zenith step).
    res = int(data_cfg["resolution_min"])
    per_day = 24 * 60 // res
    offset = utc_offset_hours(longitude)
    days = pd.date_range(start, end, freq="D")
    if len(days) == 0:
        raise ValueError("empty date range")
    zen_d, gcs_d, mask_d, time_d = {}, {}, {}, {}
    for day in days.append(pd.date_range(days[-1] + pd.Timedelta(days=1), periods=max(forecast_days - 1, 0), freq="D")):
        local = day + pd.to_timedelta(np.arange(per_day) * res, unit="min")
        utc = (local - pd.Timedelta(hours=offset)).tz_localize("UTC")
        zen = solar_zenith_deg(utc, latitude, longitude)
        gcs = (clearsky(utc, zen) if clearsky is not None else haurwitz_ghi(zen)).astype(float)
        up = (zen < float(data_cfg["zenith_daylight_max"])) & (gcs > float(data_cfg["clearsky_floor"]))
        if up.sum() < int(data_cfg.get("min_daylight_steps", 1)):
            zen_d[day] = gcs_d[day] = mask_d[day] = time_d[day] = None
            continue
        steps = np.flatnonzero(up)
        noon = steps[np.argmin(zen[steps])]
        cols = noon_col + (steps - noon)
        keep = (cols >= 0) & (cols < K)
        z, g, m = np.full(K, np.nan), np.full(K, np.nan), np.zeros(K, bool)
        z[cols[keep]], g[cols[keep]], m[cols[keep]] = zen[steps[keep]], gcs[steps[keep]], True
        zen_d[day], gcs_d[day], mask_d[day] = z, g, m
        # UTC time of every column, extended from solar noon (also for masked columns)
        offsets = ((np.arange(K) - noon_col) * res).astype("timedelta64[m]")
        time_d[day] = utc[noon].tz_localize(None).to_datetime64() + offsets
    rows = []
    for day in days:
        span = [day + pd.Timedelta(days=i) for i in range(forecast_days)]
        if any(mask_d.get(d) is None for d in span):
            continue
        rows.append((day, span))
    if not rows:
        raise ValueError("no usable dates (no daylight on the requested days)")
    cat = lambda store, span: np.concatenate([store[d] for d in span])
    ords = np.array([_dt.date(d.year, d.month, d.day).toordinal() for d, _ in rows], dtype=np.int64)
    coords = site_vector(latitude, longitude, altitude, altitude_scale_m)
    return {"fut_zen": np.stack([cat(zen_d, s) for _, s in rows]).astype(np.float32),
            "fut_time": np.stack([cat(time_d, s) for _, s in rows]),
            "fut_mask": np.stack([cat(mask_d, s) for _, s in rows]),
            "fut_ghi_cs": np.stack([cat(gcs_d, s) for _, s in rows]).astype(np.float32),
            "date_ord": ords, "site_coords": np.tile(coords, (len(rows), 1))}
