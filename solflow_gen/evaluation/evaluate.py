# Held-out evaluation of the Flow and of DeepQuantile (independent and copula-coupled).
#
# For every fold the checkpoints are loaded, the fold's held-out days are generated
# (never scored against anything the models saw in training) and compared with the
# real days: value distribution, ramps, daily totals, autocorrelation, cross-step
# covariance, joint scores and sky-regime mix. Both models are also scored on
# marginal quantiles (pinball, coverage) at the same levels, so the marginal
# comparison is like for like.
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..data.archive import conditions, load_windows, select
from ..data.splits import make_folds
from ..models.checkpoint import load_model
from ..models.deepquantile import quantile_curves_to_scenarios
from ..utils import to_plain, write_json
from .copula_fit import fit_deepquantile_copula
from .distribution import aggregate_folds, compare_distributions
from .scores import aggregate_quantile_metrics, quantile_metrics


def checkpoint_path(models_dir, family: str, fold: int) -> Path:
    return Path(models_dir) / f"{family}_fold{fold}.pt"


def fold_test_indices(model, W, fold: int) -> np.ndarray:
    # Prefer the test windows recorded in the checkpoint; otherwise rebuild the split.
    meta = model.fold_metadata or {}
    saved = meta.get("test_window_indices")
    if saved is not None and len(saved):
        idx = np.asarray(saved, dtype=int)
        n = len(W["fut_csi"])
        if np.any((idx < 0) | (idx >= n)):
            raise ValueError("checkpoint test indices do not fit this window archive "
                             "(was it rebuilt after training?)")
        if meta.get("fold_id") is not None and int(meta["fold_id"]) != int(fold):
            raise ValueError(f"checkpoint fold_id={meta['fold_id']} does not match requested fold={fold}")
        return idx
    folds = make_folds(model.cfg, W)
    if not 0 <= int(fold) < len(folds):
        raise ValueError(f"fold must be between 0 and {len(folds) - 1}")
    return np.asarray(folds[int(fold)][1], dtype=int)


def chronological_rows(W, idx):
    # Order rows by (site, date) so regime sequences are generated along real time.
    groups = np.asarray(W["site_id"])[idx] if "site_id" in W else np.zeros(len(idx), dtype=int)
    dates = np.asarray(W["date_ord"])[idx].astype(np.int64)
    order = np.lexsort((dates, groups))
    return np.asarray(idx)[order], groups[order], dates[order]


def _flow_scenarios(flow, cond, groups, count, rng):
    if not flow.regime_enabled:
        return flow.generate(cond, count, rng), False
    # One connected regime sequence per site and member, so day-to-day regime
    # persistence is part of what gets evaluated.
    out = np.empty((len(groups), count, flow.H), np.float32)
    for member in range(count):
        for g in np.unique(groups):
            rows = np.flatnonzero(groups == g)
            sub = {k: np.asarray(v)[rows] for k, v in cond.items()}
            pred, _ = flow.generate_sequence(sub, 1, rng)
            out[rows, member, :] = pred[:, 0, :]
    return out, True


def _selection_info(model) -> dict[str, Any]:
    return {"best_epoch": model.best_epoch, "epochs_trained": model.epochs_trained,
            "metric": model.selection_metric,
            "parameter_count": int(sum(p.numel() for p in model.net.parameters()))}


def evaluate_fold(W, fold: int, flow=None, dq=None, n_series: int | None = None,
                  copula_mode: str = "gaussian") -> dict[str, Any]:
    if flow is None and dq is None:
        raise ValueError("nothing to evaluate: pass a Flow, a DeepQuantile, or both")
    ref = flow if flow is not None else dq
    ecfg = ref.cfg.get("eval", {})
    count = int(n_series or ecfg.get("n_series", 100))
    if count < 1:
        raise ValueError("n_series must be positive")
    seed = int(ecfg.get("seed", 1990)) + int(fold)
    rng = np.random.default_rng(seed)
    test_idx = fold_test_indices(ref, W, fold)
    if flow is not None and dq is not None and not np.array_equal(
            np.sort(fold_test_indices(flow, W, fold)), np.sort(fold_test_indices(dq, W, fold))):
        raise ValueError("Flow and DeepQuantile checkpoints were trained for different test windows")
    rows, groups, dates = chronological_rows(W, test_idx)
    view = select(W, rows)
    cond, truth, mask = conditions(view), view["fut_csi"], view["fut_mask"]
    common = dict(bins=int(ecfg.get("histogram_bins", 50)), ks_subsample=int(ecfg.get("ks_subsample", 20000)))
    regime_cfg = ref.cfg["model"].get("regime", {})
    n_reg = int(ref.cfg["model"].get("n_regimes", 4)) or 4
    full = dict(regime_cfg=regime_cfg, n_regimes=n_reg, K=ref.K, n_days=ref.n_days, **common)

    out: dict[str, Any] = {"fold": int(fold), "n_test": int(len(rows)), "n_series_requested": count,
                           "seed": seed}
    quantile_part: dict[str, Any] = {}
    flow_scen = None
    if flow is not None:
        flow_scen, sequenced = _flow_scenarios(flow, cond, groups, count, rng)
        out["flow"] = compare_distributions(
            truth, mask, flow_scen, rng=rng,
            sequence_groups=groups if sequenced else None,
            sequence_ord=dates if sequenced else None, **full)
        out["flow_conditioning"] = {
            "site": flow.site_vec is not None, "regime": flow.regime_enabled,
            "clear_sky_ghi": flow.gcs_scale is not None}
        out["flow_checkpoint_selection"] = _selection_info(flow)
    if dq is not None:
        levels = np.asarray(dq.q_levels, float)
        curves = dq.predict_quantiles(cond)
        quantile_part["quantile_levels"] = levels.tolist()
        quantile_part["deepquantile"] = quantile_metrics(truth, mask, curves, levels)
        if flow_scen is not None:
            quantile_part["flow"] = quantile_metrics(
                truth, mask, np.quantile(flow_scen, levels, axis=1).transpose(1, 0, 2), levels)
        quantile_part["note"] = "both models scored at the same levels on the same masked held-out values"
        out["marginal_quantile_metrics"] = quantile_part
        out["deepquantile_quantile_curve_marginals"] = compare_distributions(
            truth, mask, curves, rng=rng, marginal_only=True, **common)
        indep = quantile_curves_to_scenarios(curves, levels, count, rng=np.random.default_rng(seed + 2))
        out["deepquantile_independent_scenarios"] = compare_distributions(
            truth, mask, indep, rng=np.random.default_rng(seed + 3), **full)
        if copula_mode == "gaussian":
            copula = fit_deepquantile_copula(dq, W, rows, fold)
            coupled = quantile_curves_to_scenarios(curves, levels, count,
                                                   rng=np.random.default_rng(seed + 4), copula=copula)
            out["deepquantile_gaussian_copula_scenarios"] = compare_distributions(
                truth, mask, coupled, rng=np.random.default_rng(seed + 5), **full)
            out["deepquantile_gaussian_copula_fit"] = copula.info
        out["deepquantile_checkpoint_selection"] = _selection_info(dq)
        out["deepquantile_n_quantiles"] = int(len(levels))
    return out


def aggregate_run(per_fold: list[dict[str, Any]]) -> dict[str, Any]:
    agg: dict[str, Any] = {"n_folds": len(per_fold)}
    for key in ("flow", "deepquantile_quantile_curve_marginals", "deepquantile_independent_scenarios",
                "deepquantile_gaussian_copula_scenarios"):
        rows = [f[key] for f in per_fold if key in f]
        if rows:
            agg[key] = aggregate_folds(rows)
    mq = aggregate_quantile_metrics(per_fold)
    if mq:
        agg["marginal_quantile_metrics"] = mq
    return agg


def evaluate_run(windows, models_dir, output, folds: list[int] | None = None, models=("flow", "deepquantile"),
                 n_series: int | None = None, copula_mode: str = "gaussian", device: str = "auto"):
    # Evaluate every fold found in models_dir and write one JSON file with the
    # per-fold results and their aggregate.
    W = load_windows(windows)
    use_flow, use_dq = "flow" in models, "deepquantile" in models
    first = checkpoint_path(models_dir, "flow" if use_flow else "deepquantile", 0)
    if folds is None:
        n_folds = len(make_folds(load_model(first, device).cfg, W))
        folds = list(range(n_folds))
    per_fold = []
    for fold in folds:
        print(f"[evaluate] fold {fold + 1}/{len(folds)}", flush=True)
        flow = load_model(checkpoint_path(models_dir, "flow", fold), device) if use_flow else None
        dq = load_model(checkpoint_path(models_dir, "deepquantile", fold), device) if use_dq else None
        per_fold.append(evaluate_fold(W, fold, flow, dq, n_series, copula_mode))
    payload = {"windows": str(Path(windows).resolve()), "models": list(models), "copula": copula_mode,
               "per_fold": per_fold, "aggregate": aggregate_run(per_fold)}
    write_json(output, to_plain(payload))
    print(f"[evaluate] wrote {output}", flush=True)
    return payload
