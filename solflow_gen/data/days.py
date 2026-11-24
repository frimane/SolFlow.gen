# Assemble daylight rows into noon-centred day grids.
#
# Every day is placed on a fixed-length canonical grid of K columns, centred
# so that the minimum-zenith step of every day sits on the same column
# (noon_col). Padding is NaN with mask = False. Rows are placed by their
# step offset from solar noon, so a mid-day data gap stays a gap instead of
# shifting the afternoon left.
from __future__ import annotations

from typing import Any

import numpy as np


def assemble_days(dates, steps, csi, zen, ghi_cs, min_daylight_steps: int = 0) -> dict[str, Any]:
    # Group daylight rows into days on a shared grid.
    #
    # All inputs are 1-D arrays of daylight rows only. dates are integer day
    # ordinals. The grid width K is the widest observed day (before + after
    # noon), so every day fits without truncation.
    dates = np.asarray(dates)
    steps = np.asarray(steps).astype(int)
    csi = np.asarray(csi, dtype=np.float64)
    zen = np.asarray(zen, dtype=np.float64)
    ghi_cs = np.asarray(ghi_cs, dtype=np.float64)

    order = np.argsort(dates, kind="stable")
    dates, steps, csi, zen, ghi_cs = dates[order], steps[order], csi[order], zen[order], ghi_cs[order]
    uniq, start, counts = np.unique(dates, return_index=True, return_counts=True)
    keep = counts >= max(int(min_daylight_steps), 1)
    if not keep.any():
        raise ValueError("assemble_days: no day meets data.min_daylight_steps")

    spans = []
    noon_step, ext_left, ext_right = [], [], []
    for u, s0, c in zip(uniq[keep], start[keep], counts[keep]):
        sl = slice(s0, s0 + c)
        s = steps[sl]
        noon = int(s[np.argmin(zen[sl])])
        spans.append(sl)
        noon_step.append(noon)
        ext_left.append(noon - int(s.min()))
        ext_right.append(int(s.max()) - noon)

    left, right = max(ext_left), max(ext_right)
    K, noon_col = left + right + 1, left
    D = len(spans)
    csi_day = np.full((D, K), np.nan)
    zen_day = np.full((D, K), np.nan)
    gcs_day = np.full((D, K), np.nan)
    mask_day = np.zeros((D, K), dtype=bool)
    for i, (sl, noon) in enumerate(zip(spans, noon_step)):
        cols = noon_col + (steps[sl] - noon)
        csi_day[i, cols] = csi[sl]
        zen_day[i, cols] = zen[sl]
        gcs_day[i, cols] = ghi_cs[sl]
        mask_day[i, cols] = True
    return {"day_ord": uniq[keep].astype(np.int64), "csi_day": csi_day, "zen_day": zen_day,
            "ghi_cs_day": gcs_day, "mask_day": mask_day, "K": int(K), "noon_col": int(noon_col)}


def repad_days(days: dict[str, Any], K_new: int, noon_new: int) -> dict[str, Any]:
    # Shift a site's day grids onto a wider shared grid (for multi-site pooling).
    K_old, noon_old = int(days["K"]), int(days["noon_col"])
    if K_old == K_new and noon_old == noon_new:
        return days
    offset = noon_new - noon_old
    if offset < 0 or offset + K_old > K_new:
        raise ValueError(f"cannot repad: offset={offset}, K_old={K_old}, K_new={K_new}")
    out = dict(days)
    out["K"], out["noon_col"] = int(K_new), int(noon_new)
    D = len(days["day_ord"])
    for key in ("csi_day", "zen_day", "ghi_cs_day", "mask_day"):
        fill_bool = key == "mask_day"
        new = np.zeros((D, K_new), dtype=bool) if fill_bool else np.full((D, K_new), np.nan)
        new[:, offset:offset + K_old] = days[key]
        out[key] = new
    return out
