from __future__ import annotations

import io
import json
from datetime import date, timedelta

import numpy as np
import pandas as pd

from geometry import build_conditions


def _day_conditions(window, left: int, right: int, day: date) -> dict:
    # Slice one calendar day out of a (possibly multi-day) conditioning window.
    return {"times": window["fut_time"][0, left:right],
            "fut_zen": window["fut_zen"][:, left:right],
            "fut_mask": window["fut_mask"][:, left:right],
            "fut_ghi_cs": window["fut_ghi_cs"][:, left:right],
            "fut_date_ord": np.asarray([day.toordinal()], dtype=np.int64)}


def generate_date_range(model, lat, lon, altitude_m, start_date, end_date, ensemble_size, seed):
    # Inclusive date range. A checkpoint with n_days=1 is run once per day; one with
    # n_days>1 is run on non-overlapping horizons whose days are then split up.
    # Days without daylight (or inside a window that has one) are skipped.
    if end_date < start_date:
        raise ValueError("end date must be on or after start date")
    cond = build_conditions(model, lat, lon, altitude_m, start_date, end_date)
    row_of = {int(o): i for i, o in enumerate(cond["date_ord"])}
    csi_max = float(model.cfg["data"].get("csi_max", 1.8))
    n_days, K = int(model.n_days), int(model.K)
    days, skipped = [], []
    start = start_date
    while start <= end_date:
        horizon = [start + timedelta(days=h) for h in range(n_days)]
        row = row_of.get(start.toordinal())
        if row is None:
            skipped += [d for d in horizon if d <= end_date]
            start += timedelta(days=n_days)
            continue
        # The seed depends on the requested seed and the date, so a day keeps its
        # samples when the range around it is made longer or shorter.
        run_seed = (int(seed) + start.toordinal()) % (2 ** 63 - 1)
        window = {k: v[row:row + 1] for k, v in cond.items()}
        samples = model.generate(window, int(ensemble_size), np.random.default_rng(run_seed))[0]
        if samples.shape != (int(ensemble_size), n_days * K):
            raise ValueError(f"checkpoint returned samples with shape {samples.shape}")
        if not np.isfinite(samples).all():
            raise ValueError("generator returned non-finite samples")
        csi = np.clip(samples, 0.0, csi_max).astype(np.float32)
        for h, day in enumerate(horizon):
            if day > end_date:
                break
            left, right = h * K, (h + 1) * K
            conditions = _day_conditions(window, left, right, day)
            ghi = csi[:, left:right] * np.nan_to_num(conditions["fut_ghi_cs"][0])[None, :]
            days.append({"samples_csi": csi[:, left:right], "samples_ghi": ghi, "conditions": conditions,
                         "lat": float(lat), "lon": float(lon), "altitude_m": float(altitude_m),
                         "forecast_date": day, "forecast_start_date": start, "horizon_day": h,
                         "seed": run_seed, "ensemble_size": int(ensemble_size)})
        start += timedelta(days=n_days)
    if not days:
        raise ValueError("no daylight in the requested range")
    days.sort(key=lambda item: item["forecast_date"])
    return {"days": days, "skipped_days": skipped, "start_date": start_date, "end_date": end_date,
            "ensemble_size": int(ensemble_size), "seed": int(seed), "model": model}


def result_table(result):
    csi = result["samples_csi"]
    ghi = result["samples_ghi"]
    cond = result["conditions"]
    c_q = np.percentile(csi, [10, 50, 90], axis=0)
    g_q = np.percentile(ghi, [10, 50, 90], axis=0)
    data = {"time_utc": pd.to_datetime(cond["times"], utc=True),
            "valid": cond["fut_mask"][0], "zenith_deg": cond["fut_zen"][0],
            "ghi_clearsky_wm2": cond["fut_ghi_cs"][0],
            "csi_p10": c_q[0], "csi_p50": c_q[1], "csi_p90": c_q[2],
            "ghi_p10_wm2": g_q[0], "ghi_p50_wm2": g_q[1], "ghi_p90_wm2": g_q[2]}
    for i in range(csi.shape[0]):
        data[f"csi_sample_{i+1:04d}"] = csi[i]
        data[f"ghi_sample_{i+1:04d}_wm2"] = ghi[i]
    return pd.DataFrame(data)


def range_table(result):
    frames = []
    for day_index, day in enumerate(result["days"], start=1):
        frame = result_table(day)
        frame.insert(0, "generator_date", day["forecast_date"].isoformat())
        frame.insert(1, "generator_start_date", day["forecast_start_date"].isoformat())
        frame.insert(2, "generator_day_index", day_index)
        frame.insert(3, "generator_horizon_day", int(day["horizon_day"]))
        frame.insert(4, "latitude", float(day["lat"]))
        frame.insert(5, "longitude", float(day["lon"]))
        frame.insert(6, "altitude_m", float(day["altitude_m"]))
        frame.insert(7, "ensemble_size", int(day["ensemble_size"]))
        frame.insert(8, "seed", int(day["seed"]))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def csv_bytes(result):
    frame = range_table(result)
    # Text that starts with these characters could be run as a formula by a
    # spreadsheet, so it gets a leading tab.
    formula_prefixes = ("=", "+", "-", "@", "\t", "\r", "\n", "＝", "＋", "－", "＠")
    for column in frame.select_dtypes(include=["object"]).columns:
        frame[column] = frame[column].map(
            lambda value: "\t" + value if isinstance(value, str) and value.startswith(formula_prefixes) else value
        )
    return frame.to_csv(index=False).encode("utf-8")


def npz_bytes(result):
    days = result["days"]
    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        samples_csi=np.stack([d["samples_csi"] for d in days]),
        samples_ghi=np.stack([d["samples_ghi"] for d in days]),
        csi_quantiles=np.stack([np.percentile(d["samples_csi"], [10, 50, 90], axis=0) for d in days]),
        ghi_quantiles=np.stack([np.percentile(d["samples_ghi"], [10, 50, 90], axis=0) for d in days]),
        forecast_dates=np.asarray([d["forecast_date"].isoformat() for d in days]),
        times_utc=np.stack([d["conditions"]["times"].astype("datetime64[ns]") for d in days]),
        fut_zen=np.stack([d["conditions"]["fut_zen"][0] for d in days]),
        fut_ghi_cs=np.stack([d["conditions"]["fut_ghi_cs"][0] for d in days]),
        fut_mask=np.stack([d["conditions"]["fut_mask"][0] for d in days]),
    )
    return buf.getvalue()


def metadata(result):
    model = result["model"]
    return json.dumps({"family": model.family, "representation": model.rep.kind,
                       "prior_kernel": model.cfg["prior"]["kernel"], "regime_conditioned": model.regime_enabled,
                       "K": model.K, "n_days": model.n_days, "horizon_steps": model.H,
                       "best_epoch": model.best_epoch,
                       "start_date": str(result["start_date"]), "end_date": str(result["end_date"]),
                       "n_generated_days": len(result["days"]),
                       "skipped_days": [d.isoformat() for d in result["skipped_days"]],
                       "ensemble_size": result["ensemble_size"], "seed": result["seed"],
                       "sites": [{"latitude": d["lat"], "longitude": d["lon"], "altitude_m": d["altitude_m"]}
                                 for d in result["days"][:1]],
                       "exports": {"csv_columns": list(range_table(result).columns), "quantiles": [10, 50, 90]},
                       "generator_days": [d["forecast_date"].isoformat() for d in result["days"]]},
                      indent=2).encode()
