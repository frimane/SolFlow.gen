# CSV observations -> window archive (.npz).
from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from .archive import validate_archive
from .days import assemble_days, repad_days
from .geo import site_vector
from .raw import clean, date_ordinals, load_raw
from .windows import build_day_windows


def _site_days(cfg: dict[str, Any], raw_glob: str):
    # CSV glob -> assembled day grids of one site (None when it has no daylight).
    df = clean(cfg, load_raw(cfg, raw_glob))
    day = df[df["is_day"]]
    if len(day) == 0:
        return None
    return assemble_days(
        date_ordinals(day["date"]), day["step"].to_numpy(), day["csi"].to_numpy(),
        day["zenith"].to_numpy(), day["ghi_cs"].to_numpy(),
        min_daylight_steps=int(cfg["data"]["min_daylight_steps"]))


def _coords(site: dict[str, Any], cfg: dict[str, Any]) -> np.ndarray:
    missing = [k for k in ("latitude", "longitude", "altitude") if site.get(k) is None]
    if missing:
        raise ValueError(
            f"site {site.get('name', '?')!r} lacks {missing}. Site coordinates are required "
            "when model.condition_site is true; pass --latitude/--longitude/--altitude "
            "or set `site` (or `sites[]`) in the config.")
    return site_vector(site["latitude"], site["longitude"], site["altitude"],
                       float(cfg["site"].get("altitude_scale_m", 2000.0)))


def preprocess(cfg: dict[str, Any], multi_site: bool = False) -> dict[str, np.ndarray]:
    # Run the full preprocessing and write cfg['paths']['windows'].
    #
    # multi_site=False reads paths.raw_glob as one station with the
    # coordinates in site. multi_site=True reads every entry of sites
    # and pools them on one shared noon-centred grid.
    forecast_days = int(cfg["task"]["forecast_days"])
    need_site = bool(cfg["model"].get("condition_site", True))

    if multi_site:
        sites = list(cfg.get("sites") or [])
        if not sites:
            raise ValueError("--multi-site needs a non-empty `sites` list in the config")
        entries = [(s, s.get("raw_glob")) for s in sites]
    else:
        if not cfg["paths"].get("raw_glob"):
            raise ValueError("no input CSV glob: set paths.raw_glob or pass --input")
        entries = [(cfg["site"], cfg["paths"]["raw_glob"])]

    per_site = []   # (site dict, days)
    for i, (site, glob_) in enumerate(entries):
        if not glob_:
            raise ValueError(f"sites[{i}] has no raw_glob")
        days = _site_days(cfg, glob_)
        name = str(site.get("name", f"site{i}"))
        if days is None:
            warnings.warn(f"site {name} has no daylight rows; skipped", RuntimeWarning)
            continue
        per_site.append((site, name, days))
        print(f"  site {name}: {len(days['day_ord'])} days, K={days['K']}, noon_col={days['noon_col']}")
    if not per_site:
        raise ValueError("no site produced any daylight day; check the CSV paths and columns")

    # one shared grid: the widest left and right extents across stations
    left = max(int(d["noon_col"]) for _, _, d in per_site)
    right = max(int(d["K"]) - int(d["noon_col"]) for _, _, d in per_site)
    K_new, noon_new = left + right, left

    parts, names = [], []
    for site_id, (site, name, days) in enumerate(per_site):
        Wi = build_day_windows(repad_days(days, K_new, noon_new), forecast_days)
        n_i = len(Wi["fut_csi"])
        Wi["site_id"] = np.full(n_i, site_id, dtype=np.int64)
        if need_site:
            Wi["site_coords"] = np.tile(_coords(site, cfg), (n_i, 1)).astype(np.float32)
        parts.append(Wi)
        names.append(name)
        print(f"  site {name}: {n_i} windows on the shared grid (K={K_new})")

    W: dict[str, np.ndarray] = {}
    for key in parts[0]:
        if key in ("K", "noon_col"):
            continue
        W[key] = np.concatenate([p[key] for p in parts], axis=0)
    W["K"], W["noon_col"] = np.int64(K_new), np.int64(noon_new)
    W["site_names"] = np.asarray(names)
    validate_archive(W)

    out = Path(cfg["paths"]["windows"])
    os.makedirs(out.parent, exist_ok=True)
    np.savez_compressed(out, **W)
    print(f"windows: {W['fut_csi'].shape} | sites: {len(names)} | K={K_new} noon_col={noon_new} "
          f"| years: {sorted(np.unique(W['forecast_year']).tolist())}")
    return W
