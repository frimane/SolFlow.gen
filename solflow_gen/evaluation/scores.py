# Proper scoring rules for ensembles of whole-day trajectories.
#
# Everything here works on numpy arrays shaped [N, M, H]: N windows, M ensemble
# members, H horizon steps, plus a boolean validity mask [N, H] that is False at
# night. Scores never look at masked-out steps.
from __future__ import annotations

import warnings

import numpy as np


def crps_masked(pred_ens, truth, mask, fair=False, per_step=False):
    # Sample CRPS per step, averaged over valid steps. "fair" divides the
    # member-spread term by M(M-1) so the score does not depend on ensemble size.
    pred = np.asarray(pred_ens, np.float64)
    truth = np.asarray(truth, np.float64)
    mask = np.asarray(mask, bool)
    _, m, _ = pred.shape
    term1 = np.abs(pred - truth[:, None, :]).mean(axis=1)
    s = np.sort(pred, axis=1)
    i = np.arange(m, dtype=np.float64)[None, :, None]
    pair = 2.0 * (s * (2 * i - m + 1.0)).sum(axis=1)
    denom = float(m * (m - 1)) if (fair and m > 1) else float(m * m)
    per = np.where(mask, term1 - 0.5 * pair / denom, np.nan)
    return per if per_step else float(np.nanmean(per))


def crps_per_window(pred_ens, truth, mask, fair=False):
    per = crps_masked(pred_ens, truth, mask, fair=fair, per_step=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(per, axis=1)


def energy_score(ens, y, valid, fair=False):
    # ens [M, H], y [H], valid [H]. Euclidean energy score over valid steps.
    if valid.sum() < 1:
        return np.nan
    x = ens[:, valid]
    yv = y[valid]
    d1 = np.linalg.norm(x - yv[None, :], axis=1).mean()
    pair = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
    d2 = pair[~np.eye(len(x), dtype=bool)].mean() if (fair and len(x) > 1) else pair.mean()
    return float(d1 - 0.5 * d2)


def variogram_score(ens, y, valid, p=0.5):
    # ens [M, H], y [H], valid [H]. Variogram score of order p; it is sensitive
    # to the dependence between steps, which the energy score only weakly is.
    idx = np.where(valid)[0]
    if len(idx) < 2:
        return np.nan
    x = ens[:, idx]
    yv = y[idx]
    obs = np.abs(yv[:, None] - yv[None, :]) ** p
    exp = (np.abs(x[:, :, None] - x[:, None, :]) ** p).mean(0)
    d = len(idx)
    return float(((obs - exp) ** 2).sum() / (d * d))


def joint_validation_score(energy, variogram):
    # Early-stopping score for the Flow: the mean of the two normalised scores.
    values = np.asarray([energy, variogram], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("the joint validation score needs finite energy and variogram scores")
    return float(values.mean())


def score_steps(horizon: int, cap: int = 64) -> np.ndarray:
    # Sub-sampled step indices for the variogram score (it is quadratic in steps).
    return np.unique(np.linspace(0, horizon - 1, min(cap, horizon), dtype=int))


def window_joint_scores(pred, target, mask):
    # Per-window fair energy score and variogram score for pred [N, M, H].
    steps = score_steps(target.shape[1])
    es = np.array([energy_score(pred[i], target[i], mask[i], fair=True) for i in range(len(pred))])
    vs = np.array([variogram_score(pred[i][:, steps], target[i, steps], mask[i, steps])
                   for i in range(len(pred))])
    return es, vs


def pinball_loss(curves, levels, truth, mask):
    # Mean pinball loss of quantile curves [N, M, H] at levels [M].
    c = np.asarray(curves, np.float64)
    q = np.asarray(levels, np.float64)[None, :, None]
    e = np.asarray(truth, np.float64)[:, None, :] - c
    pin = np.maximum(q * e, (q - 1.0) * e)
    w = np.asarray(mask, bool)[:, None, :] & np.isfinite(e)
    return float(np.where(w, pin, 0.0).sum() / max(w.sum(), 1))


def quantile_metrics(truth, mask, forecasts, levels):
    # Pinball loss and empirical coverage per quantile level for forecasts
    # [N, M, H] at levels [M], on the masked held-out values.
    truth = np.asarray(truth, float)
    mask = np.asarray(mask, bool)
    fc = np.asarray(forecasts, float)
    levels = np.asarray(levels, float)
    if fc.ndim != 3 or fc.shape[0] != len(truth) or fc.shape[2] != truth.shape[1] or fc.shape[1] != len(levels):
        raise ValueError("quantile forecasts must have shape [rows, levels, horizon]")
    if np.any((levels <= 0.0) | (levels >= 1.0)) or np.any(np.diff(levels) <= 0.0):
        raise ValueError("quantile levels must be strictly increasing inside (0, 1)")
    valid = np.broadcast_to(mask[:, None, :], fc.shape) & np.isfinite(fc) & np.isfinite(truth[:, None, :])
    if not valid.any():
        raise ValueError("no finite masked values for quantile evaluation")
    err = truth[:, None, :] - fc
    tau = levels[None, :, None]
    loss = np.maximum(tau * err, (tau - 1.0) * err)
    counts = valid.sum(axis=(0, 2))
    nan = np.full(len(levels), np.nan)
    pinball = np.divide(np.where(valid, loss, 0.0).sum(axis=(0, 2)), counts, out=nan.copy(), where=counts > 0)
    coverage = np.divide(((truth[:, None, :] <= fc) & valid).sum(axis=(0, 2)), counts,
                         out=nan.copy(), where=counts > 0)
    return {"mean_pinball": float(np.nanmean(pinball)),
            "mean_absolute_coverage_error": float(np.nanmean(np.abs(coverage - levels))),
            "pinball_by_quantile": pinball.tolist(), "coverage_by_quantile": coverage.tolist(),
            "n_valid_per_quantile": counts.astype(int).tolist()}


def aggregate_quantile_metrics(per_fold, models=("flow", "deepquantile")):
    # Pool per-fold quantile metrics, weighting every level by its number of valid values.
    out = {}
    levels = None
    for model in models:
        items = [f.get("marginal_quantile_metrics", {}).get(model, {}) for f in per_fold]
        items = [i for i in items if i.get("mean_pinball") is not None]
        if not items:
            continue
        levels = np.asarray(per_fold[0]["marginal_quantile_metrics"]["quantile_levels"], float)
        pin = np.asarray([i["pinball_by_quantile"] for i in items], float)
        cov = np.asarray([i["coverage_by_quantile"] for i in items], float)
        cnt = np.asarray([i["n_valid_per_quantile"] for i in items], float)
        tot = cnt.sum(axis=0)
        nan = np.full(len(levels), np.nan)
        pin_p = np.divide((pin * cnt).sum(axis=0), tot, out=nan.copy(), where=tot > 0)
        cov_p = np.divide((cov * cnt).sum(axis=0), tot, out=nan.copy(), where=tot > 0)
        weight = np.where(np.isfinite(pin_p), tot, 0.0)
        out[model] = {"mean_pinball": float(np.nansum(pin_p * weight) / weight.sum()),
                      "mean_absolute_coverage_error": float(np.nanmean(np.abs(cov_p - levels))),
                      "pinball_by_quantile": pin_p.tolist(), "coverage_by_quantile": cov_p.tolist(),
                      "n_folds": len(items), "n_valid_per_quantile": tot.astype(int).tolist()}
    if out:
        out["quantile_levels"] = levels.tolist()
    return out
