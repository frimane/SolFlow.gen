# Train/test splits over window indices, purged against leakage.
#
# Every scheme purges by day interval: a window can only go into a training set if
# its [first_day, last_day] interval does not touch the evaluation interval.
# date_ord is comparable across stations, so this also holds for pooled archives.
#
# Schemes (split.scheme):
#   holdout                         one chronological cut
#   forward                         forward chaining over n_folds blocks
#   blocked_cv                      n_folds contiguous blocks, each held out once
#   loso                            leave one station out
#   strict_spatiotemporal_holdout   train on other stations in the training years,
#                                   test on a dispersed group of stations in the test
#                                   years, repeated spatial_repeats times
from __future__ import annotations

import datetime as _dt
from typing import Any

import numpy as np

Fold = tuple[np.ndarray, np.ndarray]


def _purge(cand_idx, first, last, eval_min_day, eval_max_day, before_only=False):
    # Keep candidates whose day interval does not touch the evaluation interval.
    cand_idx = np.asarray(cand_idx)
    f, l = np.asarray(first)[cand_idx], np.asarray(last)[cand_idx]
    keep = (l < eval_min_day) if before_only else ((l < eval_min_day) | (f > eval_max_day))
    return cand_idx[keep]


def _chronological(W) -> np.ndarray:
    return np.argsort(np.asarray(W["date_ord"]), kind="stable")


def blocked_cv_folds(cfg, W) -> list[Fold]:
    n = int(cfg["split"]["n_folds"])
    first, last = W["first_day_ord"], W["last_day_ord"]
    order = _chronological(W)
    folds = []
    for block in np.array_split(order, n):
        test = np.sort(block)
        rest = np.setdiff1d(order, block)
        train = np.sort(_purge(rest, first, last, first[test].min(), last[test].max()))
        folds.append((train, test))
    return folds


def forward_chaining_folds(cfg, W) -> list[Fold]:
    n = int(cfg["split"]["n_folds"])
    first, last = W["first_day_ord"], W["last_day_ord"]
    order = _chronological(W)
    blocks = np.array_split(order, n + 1)
    folds = []
    for k in range(1, len(blocks)):
        test = np.sort(blocks[k])
        tmin = int(first[test].min())
        train = np.sort(_purge(np.concatenate(blocks[:k]), first, last, tmin, tmin,
                               before_only=True))
        folds.append((train, test))
    return folds


def holdout_split(cfg, W) -> Fold:
    first, last = W["first_day_ord"], W["last_day_ord"]
    order = _chronological(W)
    hold = cfg["split"].get("holdout") or {}
    if hold.get("train_end"):
        cut_day = _dt.date.fromisoformat(str(hold["train_end"])).toordinal()
        test = order[first[order] > cut_day]
        cand = order[last[order] <= cut_day]
        if len(test) == 0 or len(cand) == 0:
            raise ValueError(f"holdout train_end={hold['train_end']} leaves an empty side")
    else:
        cut = int(len(order) * float(hold.get("train_frac", 0.8)))
        test, cand = np.sort(order[cut:]), order[:cut]
        if len(test) == 0 or len(cand) == 0:
            raise ValueError("holdout.train_frac leaves an empty side")
    train = np.sort(_purge(cand, first, last, first[test].min(), last[test].max()))
    return train, np.sort(test)


def loso_folds(cfg, W) -> list[Fold]:
    if "site_id" not in W:
        raise ValueError("split scheme 'loso' needs a multi-site archive (no site_id found)")
    sid = np.asarray(W["site_id"])
    order = _chronological(W)
    folds = []
    for s in np.unique(sid):
        test, train = np.sort(order[sid[order] == s]), np.sort(order[sid[order] != s])
        if len(test) and len(train):
            folds.append((train, test))
    if not folds:
        raise ValueError("loso produced no usable folds")
    return folds


def _station_points(W):
    if "site_id" not in W or "site_coords" not in W:
        raise ValueError("strict_spatiotemporal_holdout needs site_id and site_coords")
    sid = np.asarray(W["site_id"], dtype=np.int64)
    coords = np.asarray(W["site_coords"], dtype=np.float64)
    sites = np.unique(sid)
    points = np.asarray([np.nanmean(coords[sid == s, :3], axis=0) for s in sites])
    if not np.isfinite(points).all():
        raise ValueError("station coordinates are incomplete or non-finite")
    points /= np.maximum(np.linalg.norm(points, axis=1, keepdims=True), 1e-12)
    return sites, points


def representative_station_holdouts(W, n_test: int, repeats: int, seed: int):
    # Reproducible, disjoint, geographically dispersed station groups.
    #
    # Farthest-point sampling spreads each held-out group over the domain; the
    # candidate pool shrinks after every repeat so no station is tested twice.
    sites, points = _station_points(W)
    if not 1 <= n_test < len(sites):
        raise ValueError("split.spatial_test_stations must be between 1 and n_stations-1")
    if repeats < 1:
        raise ValueError("split.spatial_repeats must be positive")
    if repeats * n_test > len(sites):
        raise ValueError("disjoint repeats need spatial_repeats * spatial_test_stations "
                         f"<= n_stations ({repeats} * {n_test} > {len(sites)})")
    rng = np.random.default_rng(seed)
    groups, remaining = [], np.arange(len(sites), dtype=int)
    for _ in range(repeats):
        first = int(rng.choice(remaining))
        chosen = [first]
        min_dist = np.linalg.norm(points[remaining] - points[first], axis=1)
        for _ in range(1, n_test):
            best = np.max(min_dist)
            candidates = remaining[np.isclose(min_dist, best, rtol=1e-10, atol=1e-12)]
            pick = int(rng.choice(candidates))
            chosen.append(pick)
            min_dist = np.minimum(min_dist, np.linalg.norm(points[remaining] - points[pick], axis=1))
        groups.append(np.asarray(sites[chosen], dtype=np.int64))
        remaining = remaining[~np.isin(remaining, chosen)]
    return groups


def strict_spatiotemporal_folds(cfg, W) -> list[Fold]:
    if "site_id" not in W or "forecast_year" not in W:
        raise ValueError("strict_spatiotemporal_holdout needs a multi-site archive with forecast_year")
    split = cfg["split"]
    years = np.unique(np.asarray(W["forecast_year"], dtype=np.int64))
    train_years = split.get("train_years")
    test_years = split.get("test_years")
    if train_years is None:
        if len(years) < 2:
            raise ValueError("the strict split needs at least two calendar years")
        train_years = years[:-1]
    if test_years is None:
        test_years = years[-1:]
    train_years, test_years = np.asarray(train_years, np.int64), np.asarray(test_years, np.int64)
    if not len(train_years) or not len(test_years) or np.intersect1d(train_years, test_years).size:
        raise ValueError("split.train_years and split.test_years must be non-empty and disjoint")
    groups = representative_station_holdouts(
        W, int(split["spatial_test_stations"]), int(split["spatial_repeats"]),
        int(split.get("spatial_seed", cfg.get("seed", 0))))
    sid = np.asarray(W["site_id"], dtype=np.int64)
    year = np.asarray(W["forecast_year"], dtype=np.int64)
    folds = []
    for test_sites in groups:
        test = np.flatnonzero(np.isin(sid, test_sites) & np.isin(year, test_years))
        train = np.flatnonzero(~np.isin(sid, test_sites) & np.isin(year, train_years))
        if not len(train) or not len(test):
            raise ValueError("a strict spatiotemporal fold has an empty side; "
                             "check split.train_years / split.test_years")
        folds.append((np.sort(train), np.sort(test)))
    return folds


def make_folds(cfg: dict[str, Any], W: dict[str, Any]) -> list[Fold]:
    # Return the (train_idx, test_idx) pairs of split.scheme.
    scheme = cfg["split"]["scheme"]
    if scheme == "strict_spatiotemporal_holdout":
        return strict_spatiotemporal_folds(cfg, W)
    if scheme == "loso":
        return loso_folds(cfg, W)
    if scheme == "holdout":
        return [holdout_split(cfg, W)]
    if scheme == "forward":
        return forward_chaining_folds(cfg, W)
    if scheme == "blocked_cv":
        return blocked_cv_folds(cfg, W)
    raise ValueError(f"unknown split.scheme {scheme!r}")


def carve_val(cfg: dict[str, Any], train_idx, W: dict[str, Any]) -> Fold:
    # Split the chronological tail of train_idx off as a purged validation set.
    first, last = W["first_day_ord"], W["last_day_ord"]
    train_idx = np.asarray(train_idx)
    order = train_idx[np.argsort(np.asarray(W["date_ord"])[train_idx], kind="stable")]
    n_val = max(1, int(len(order) * float(cfg["split"]["val_frac"])))
    val = np.sort(order[-n_val:])
    tr = np.sort(_purge(order[:-n_val], first, last, first[val].min(), last[val].max()))
    return tr, val
