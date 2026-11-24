# Build training windows: runs of consecutive calendar days.
#
# A window is forecast_days consecutive days; a missing day breaks it. Each window
# keeps the ordinal of its first and last day, so two windows overlap exactly when
# those day intervals intersect. That is what the split purging relies on.
#
# Fields written to the archive (N windows, H = forecast_days * K steps):
#   fut_csi, fut_zen, fut_mask, fut_ghi_cs   [N, H]
#   date_ord                                 [N] first day, comparable across sites
#   first_day_ord, last_day_ord              [N]
#   forecast_year                            [N]
#   K, noon_col                              scalars
from __future__ import annotations

from typing import Any

import numpy as np

_UNIX_EPOCH_ORDINAL = 719163


def years_from_ordinals(ordinals) -> np.ndarray:
    # Calendar year of proleptic Gregorian ordinals.
    days = np.asarray(ordinals, dtype=np.int64) - _UNIX_EPOCH_ORDINAL
    return days.astype("datetime64[D]").astype("datetime64[Y]").astype(np.int64) + 1970


def build_day_windows(days: dict[str, Any], forecast_days: int) -> dict[str, np.ndarray]:
    span = int(forecast_days)
    if span < 1:
        raise ValueError("forecast_days must be >= 1")
    order = np.argsort(days["day_ord"])
    day_ord = np.asarray(days["day_ord"])[order]
    csi, zen = days["csi_day"][order], days["zen_day"][order]
    gcs, msk = days["ghi_cs_day"][order], days["mask_day"][order]

    starts = [i for i in range(len(day_ord) - span + 1)
              if day_ord[i + span - 1] - day_ord[i] == span - 1]   # a gap day breaks the window
    if not starts:
        raise ValueError("no windows: days are not consecutive enough for "
                         f"task.forecast_days={span}")
    take = lambda arr: np.stack([arr[i:i + span].reshape(-1) for i in starts])
    first = day_ord[starts]
    return {
        "fut_csi": take(csi).astype(np.float32),
        "fut_zen": take(zen).astype(np.float32),
        "fut_mask": take(msk).astype(bool),
        "fut_ghi_cs": take(gcs).astype(np.float32),
        "date_ord": first.astype(np.int64),
        "first_day_ord": first.astype(np.int64),
        "last_day_ord": (first + span - 1).astype(np.int64),
        "forecast_year": years_from_ordinals(first),
        "K": np.int64(days["K"]),
        "noon_col": np.int64(days["noon_col"]),
    }
