# The window archive (.npz): loading, validation and row selection.
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np

# Arrays with one row per window. Everything else is archive-level metadata.
ROW_KEYS = ("fut_csi", "fut_zen", "fut_mask", "fut_ghi_cs", "date_ord", "first_day_ord",
            "last_day_ord", "forecast_year", "site_id", "site_coords")
# Arrays a generator needs to condition on (no observed CSI).
CONDITION_KEYS = ("fut_zen", "fut_mask", "fut_ghi_cs", "date_ord", "site_coords")
REQUIRED_KEYS = ("fut_csi", "fut_zen", "fut_mask", "date_ord", "first_day_ord",
                 "last_day_ord", "K")
_METADATA = {"K", "noon_col", "site_names"}


class ArchiveError(ValueError):
    # Raised when a window archive violates the expected contract.
    pass


def load_windows(path: str | Path) -> dict[str, np.ndarray]:
    # Load and validate a window archive written by preprocessing.
    z = np.load(path, allow_pickle=False)
    W = {k: z[k] for k in z.files}
    validate_archive(W)
    return W


def n_windows(W: dict[str, Any]) -> int:
    return int(len(np.asarray(W["fut_csi"])))


def validate_archive(W: dict[str, Any], require_target: bool = True) -> None:
    # Check keys, shapes, alignment and the mask contract.
    required = REQUIRED_KEYS if require_target else tuple(k for k in REQUIRED_KEYS if k != "fut_csi")
    missing = [k for k in required if k not in W]
    if missing:
        raise ArchiveError("archive is missing required arrays: " + ", ".join(missing))
    zen = np.asarray(W["fut_zen"])
    mask = np.asarray(W["fut_mask"])
    n = len(zen)
    if zen.ndim != 2 or mask.shape != zen.shape:
        raise ArchiveError(f"fut_zen/fut_mask shapes are inconsistent: {zen.shape} vs {mask.shape}")
    K = int(np.asarray(W["K"]).reshape(-1)[0])
    if K < 1 or zen.shape[1] % K:
        raise ArchiveError(f"horizon {zen.shape[1]} is not a multiple of K={K}")
    for key, value in W.items():
        arr = np.asarray(value)
        if key in _METADATA or arr.ndim == 0:
            continue
        if len(arr) != n:
            raise ArchiveError(f"array {key!r} has {len(arr)} rows but the archive has {n} windows")
    if require_target:
        fut, m = np.asarray(W["fut_csi"]), mask.astype(bool)
        if fut.shape != zen.shape:
            raise ArchiveError(f"fut_csi shape {fut.shape} != fut_zen shape {zen.shape}")
        if np.any(m & ~np.isfinite(fut)):
            raise ArchiveError("fut_mask marks non-finite CSI values as valid")
        if not m.any():
            raise ArchiveError("fut_mask contains no valid daylight step")
    if "site_coords" in W:
        coords = np.asarray(W["site_coords"], dtype=np.float32)
        if coords.shape != (n, 4):
            raise ArchiveError(f"site_coords must have shape {(n, 4)}, got {coords.shape}")
        if not np.isfinite(coords).all():
            raise ArchiveError("site_coords contains non-finite values")
        if np.max(np.abs(np.linalg.norm(coords[:, :3], axis=1) - 1.0)) > 2e-3:
            raise ArchiveError("site_coords[:, :3] must be unit-length (Earth-centred) vectors")
    if "date_ord" in W and not np.isfinite(np.asarray(W["date_ord"], dtype=float)).all():
        raise ArchiveError("date_ord must be finite for every window")


def horizon_days(W: dict[str, Any]) -> tuple[int, int]:
    # Return (K, n_days) of an archive.
    K = int(np.asarray(W["K"]).reshape(-1)[0])
    return K, int(np.asarray(W["fut_zen"]).shape[1] // K)


def select(W: dict[str, Any], idx, keys: Iterable[str] | None = None) -> dict[str, np.ndarray]:
    # Row-select the per-window arrays of W (a copy-free view where NumPy allows).
    idx = np.asarray(idx, dtype=int)
    wanted = ROW_KEYS if keys is None else tuple(keys)
    return {k: np.asarray(W[k])[idx] for k in wanted if k in W}


def conditions(view: dict[str, Any]) -> dict[str, np.ndarray]:
    # Keep only the arrays a generator conditions on (drops observed CSI).
    return {k: view[k] for k in CONDITION_KEYS if k in view}
