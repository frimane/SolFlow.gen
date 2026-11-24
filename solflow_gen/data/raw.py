# Load raw CSV observations and derive the clear-sky index (CSI).
from __future__ import annotations

import glob
import warnings
from typing import Any

import numpy as np
import pandas as pd

_UNIX_EPOCH_ORDINAL = 719163  # datetime.date(1970, 1, 1).toordinal()


def infer_utc_offset_hours(df: pd.DataFrame) -> float:
    # Whole-hour offset that puts the day boundary at local midnight.
    #
    # The lowest-zenith rows (about solar noon) are located in UTC time of day
    # with a circular mean; the offset moves that to 12:00 local time.
    z = df["zenith"].to_numpy(float)
    valid = np.isfinite(z)
    if valid.sum() < 100:
        warnings.warn("too few valid zenith rows to infer the UTC offset; using 0",
                      RuntimeWarning)
        return 0.0
    near_noon = valid & (z <= np.nanpercentile(z[valid], 1))
    tod = (df["datetime"].dt.hour + df["datetime"].dt.minute / 60
           + df["datetime"].dt.second / 3600).to_numpy()[near_noon]
    ang = tod / 24.0 * 2.0 * np.pi
    mean_ang = np.arctan2(np.mean(np.sin(ang)), np.mean(np.cos(ang))) % (2.0 * np.pi)
    offset = round(12.0 - mean_ang / (2.0 * np.pi) * 24.0)
    return float(((offset + 12) % 24) - 12)


def load_raw(cfg: dict[str, Any], raw_glob: str) -> pd.DataFrame:
    # Read the CSVs matching raw_glob onto the model grid.
    #
    # Returns a frame with columns datetime (UTC), ghi, ghi_cs,
    # zenith, date (local calendar day) and step (grid slot of the day).
    d = cfg["data"]
    files = sorted(glob.glob(str(raw_glob)))
    if not files:
        raise FileNotFoundError(f"no CSV files match {raw_glob}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    missing = [d[k] for k in ("col_datetime", "col_ghi", "col_ghi_cs", "col_zenith")
               if d[k] not in df.columns]
    if missing:
        raise ValueError(f"CSV files are missing columns {missing}; "
                         "set data.col_* in the config or pass the --*-column options")
    df[d["col_datetime"]] = pd.to_datetime(df[d["col_datetime"]], utc=True)
    df = df.rename(columns={d["col_datetime"]: "datetime", d["col_ghi"]: "ghi",
                            d["col_ghi_cs"]: "ghi_cs", d["col_zenith"]: "zenith"})
    df = df[["datetime", "ghi", "ghi_cs", "zenith"]].drop_duplicates("datetime")
    df = df.sort_values("datetime").reset_index(drop=True)

    # a non-physical reading invalidates both irradiances so CSI never mixes them
    nonphys = ((~np.isfinite(df["ghi"])) | (df["ghi"] < 0)
               | (~np.isfinite(df["ghi_cs"])) | (df["ghi_cs"] < 0))
    df.loc[nonphys, ["ghi", "ghi_cs"]] = np.nan

    res_min = int(d["resolution_min"])
    if d.get("resample", False):
        rule = f"{res_min}min"
        idx = df.set_index("datetime")
        agg = idx[["ghi", "ghi_cs", "zenith"]].resample(rule, label="left", closed="left").mean()
        cnt = idx["ghi"].resample(rule, label="left", closed="left").count()
        max_samples = max(1, res_min // int(d.get("source_resolution_min", 1)))
        min_samples = max(1, int(np.ceil(float(d.get("resample_min_frac", 0.6)) * max_samples)))
        df = agg[cnt >= min_samples].reset_index()

    local = (df["datetime"] + pd.Timedelta(hours=infer_utc_offset_hours(df))).dt.tz_localize(None)
    df["date"] = local.dt.normalize()
    df["step"] = ((local - df["date"]).dt.total_seconds() // (res_min * 60)).astype(int)
    return df


def clean(cfg: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    # Derive CSI and the daylight flag.
    #
    # csi_raw is the unclipped ratio, csi is clipped to
    # [csi_min, csi_max] and is_day marks valid daylight rows.
    d = cfg["data"]
    df = df.sort_values(["date", "step"]).reset_index(drop=True)
    ghi, ghi_cs = df["ghi"].to_numpy(float), df["ghi_cs"].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        csi = np.where(ghi_cs > float(d["clearsky_floor"]), ghi / ghi_cs, np.nan)
    df["csi_raw"] = csi
    is_day = (df["zenith"] < float(d["zenith_daylight_max"])).to_numpy() & np.isfinite(csi)
    if d.get("drop_suspect_zero", True):
        is_day &= ~(csi <= 0.0)
    df["is_day"] = is_day
    df["csi"] = np.clip(csi, float(d["csi_min"]), float(d["csi_max"]))
    return df


def date_ordinals(dates: pd.Series) -> np.ndarray:
    # Proleptic Gregorian ordinals (datetime.date.toordinal) of a date column.
    days = dates.to_numpy().astype("datetime64[D]").astype(np.int64)
    return days + _UNIX_EPOCH_ORDINAL
