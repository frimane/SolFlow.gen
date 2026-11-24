# Distribution and trajectory checks of generated days against held-out real days.
#
# Inputs: real CSI [N, H] with mask [N, H] and generated scenarios [N, M, H]. The
# generated days are scored on the mask of the real day they stand in for. The
# result is a nested dict of plain floats / lists (JSON-ready).
from __future__ import annotations

import numpy as np

from ..models.regimes import classify_regimes, stationary_distribution, transition_matrix
from .scores import energy_score


def _flat(values, mask=None):
    x = np.asarray(values, np.float64)
    ok = np.isfinite(x) if mask is None else (np.asarray(mask, bool) & np.isfinite(x))
    return x[ok]


def ks_statistic(a, b, max_samples=20000, rng=None):
    # Two-sample Kolmogorov-Smirnov distance, sub-sampled for speed.
    a, b, rng = _flat(a), _flat(b), rng or np.random.default_rng(0)
    if max_samples > 0:
        if len(a) > max_samples:
            a = rng.choice(a, max_samples, replace=False)
        if len(b) > max_samples:
            b = rng.choice(b, max_samples, replace=False)
    if not len(a) or not len(b):
        return float("nan")
    a, b = np.sort(a), np.sort(b)
    z = np.sort(np.concatenate([a, b]))
    return float(np.max(np.abs(np.searchsorted(a, z, side="right") / len(a)
                               - np.searchsorted(b, z, side="right") / len(b))))


def overlap_coefficient(real, synthetic, bins=50, value_range=None):
    # Shared mass of two histograms: 1 means identical, 0 means disjoint.
    a, b = _flat(real), _flat(synthetic)
    if not len(a) or not len(b):
        return float("nan")
    if value_range is None:
        lo, hi = float(min(a.min(), b.min())), float(max(a.max(), b.max()))
    else:
        lo, hi = map(float, value_range)
        if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
            raise ValueError("value_range must contain finite increasing bounds")
    if hi <= lo:
        return 1.0
    edges = np.linspace(lo, hi, max(int(bins), 2) + 1)
    pa = np.histogram(np.clip(a, lo, hi), bins=edges)[0].astype(float)
    pb = np.histogram(np.clip(b, lo, hi), bins=edges)[0].astype(float)
    return float(np.minimum(pa / max(pa.sum(), 1.0), pb / max(pb.sum(), 1.0)).sum())


def _mean(values):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    return float(v.mean()) if len(v) else float("nan")


def _pair(real, synth, bins, ks_n, rng, value_range=None):
    return {"ks_statistic": ks_statistic(real, synth, ks_n, rng),
            "overlap_coefficient": overlap_coefficient(real, synth, bins, value_range),
            "real_mean": float(np.mean(real)) if len(real) else np.nan,
            "synthetic_mean": float(np.mean(synth)) if len(synth) else np.nan,
            "real_std": float(np.std(real)) if len(real) else np.nan,
            "synthetic_std": float(np.std(synth)) if len(synth) else np.nan}


def _acf(x, mask, max_lag):
    # Per-row autocorrelation at lags 1..max_lag over steps where both ends are valid.
    x = np.asarray(x, float)
    m = np.asarray(mask, bool) & np.isfinite(x)
    out = []
    for row, ok in zip(x, m):
        a = []
        for lag in range(1, max_lag + 1):
            v = ok[:-lag] & ok[lag:]
            if v.sum() < 3:
                a.append(np.nan)
                continue
            u, w = row[:-lag][v], row[lag:][v]
            su, sw = np.std(u), np.std(w)
            a.append(float(np.mean((u - u.mean()) * (w - w.mean())) / (su * sw))
                     if su > 1e-12 and sw > 1e-12 else np.nan)
        out.append(a)
    return np.asarray(out, float)


def _masked_cov(x, mask, max_rows=20000):
    # Step-by-step covariance matrix from pairwise-valid rows.
    x = np.asarray(x, float)
    m = np.asarray(mask, bool) & np.isfinite(x)
    if len(x) > max_rows:
        rows = np.sort(np.random.default_rng(0).choice(len(x), size=int(max_rows), replace=False))
        x, m = x[rows], m[rows]
    v, valid = np.where(m, x, 0.0), m.astype(float)
    count = valid.T @ valid
    with np.errstate(divide="ignore", invalid="ignore"):
        cov = (v.T @ v) / count - ((v.T @ valid) / count) * ((valid.T @ v) / count)
    cov[count < 3] = np.nan
    return cov


def _ramps(x, mask):
    # Largest absolute step-to-step change of each trajectory.
    x = np.asarray(x, float)
    m = np.asarray(mask, bool) & np.isfinite(x)
    out = []
    for row, ok in zip(x, m):
        v = ok[:-1] & ok[1:]
        if v.any():
            out.append(float(np.max(np.abs(np.diff(row)[v]))))
    return np.asarray(out, float)


def _roughness(x, mask):
    x = np.asarray(x, float)
    m = np.asarray(mask, bool) & np.isfinite(x)
    out = []
    for row, ok in zip(x, m):
        v = ok[:-1] & ok[1:]
        if v.any():
            out.append(float(np.mean(np.abs(np.diff(row)[v]))))
    return np.asarray(out, float)


def _daily_totals(x, mask, K, n_days):
    x, m = np.asarray(x, float), np.asarray(mask, bool)
    out = []
    for d in range(int(n_days)):
        sl = slice(d * int(K), (d + 1) * int(K))
        ok = m[:, sl] & np.isfinite(x[:, sl])
        out.extend(np.where(ok, x[:, sl], 0.0).sum(axis=1)[ok.any(axis=1)].tolist())
    return np.asarray(out, float)


def _joint_scores(real, mask, syn):
    # Mean fair energy score and mean variogram score (<= 64 evenly spaced valid steps).
    es, vs = [], []
    for i in range(len(real)):
        ok = mask[i] & np.isfinite(syn[i]).all(axis=0)
        if ok.sum() < 1:
            continue
        es.append(energy_score(syn[i].astype(float), real[i].astype(float), ok, fair=True))
        idx = np.flatnonzero(mask[i])
        if len(idx) > 64:
            idx = idx[np.unique(np.linspace(0, len(idx) - 1, 64, dtype=int))]
        good = np.isfinite(syn[i][:, idx]).all(axis=1)
        if len(idx) >= 2 and good.any():
            x, y = syn[i][good][:, idx].astype(float), real[i][idx].astype(float)
            yd = np.abs(y[:, None] - y[None, :]) ** 0.5
            xd = np.mean(np.abs(x[:, :, None] - x[:, None, :]) ** 0.5, axis=0)
            vs.append(float(((yd - xd) ** 2).mean()))
    return _mean(es), _mean(vs)


def _sequence_transition(labels, groups, ordinals, n_regimes):
    # Transition matrix from labels, only counting steps between consecutive dates
    # of the same site (a break is inserted wherever dates are not consecutive).
    labels, groups, ordinals = np.asarray(labels, np.int64), np.asarray(groups), np.asarray(ordinals, np.int64)
    seq = []
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        idx = idx[np.argsort(ordinals[idx], kind="stable")]
        prev = None
        for r in idx:
            if prev is not None and ordinals[r] != ordinals[prev] + 1:
                seq.append(-1)
            seq.append(int(labels[r]))
            prev = r
        seq.append(-1)
    tm = transition_matrix(np.asarray(seq, np.int64), n_regimes, smoothing=0.0)
    return tm, stationary_distribution(tm)


def compare_distributions(real, real_mask, synthetic, regime_cfg=None, n_regimes=4, bins=50,
                          ks_subsample=20000, rng=None, K=None, n_days=None, sequence_groups=None,
                          sequence_ord=None, marginal_only=False):
    # Compare real days with generated scenarios. marginal_only=True is for quantile
    # curves, which are not scenarios, so only the value distribution is meaningful.
    rng = rng or np.random.default_rng(0)
    real = np.asarray(real, np.float32)
    rm = np.asarray(real_mask, bool)
    syn = np.asarray(synthetic, np.float32)
    if syn.ndim != 3 or real.ndim != 2 or syn.shape[0] != real.shape[0] or syn.shape[2] != real.shape[1]:
        raise ValueError("real must be [N, H] and synthetic [N, M, H] with matching shapes")
    sm = np.broadcast_to(rm[:, None, :], syn.shape)
    rv, sv = _flat(real, rm), _flat(syn, sm)
    res = {"n_real_values": int(len(rv)), "n_synthetic_values": int(len(sv)),
           "n_generated_series": int(syn.shape[1]),
           "value_distribution": _pair(rv, sv, bins, ks_subsample, rng, value_range=(0.0, 1.6))}
    if marginal_only:
        res["diagnostic_scope"] = "marginal values only; the members are quantile curves, not scenarios"
        return res

    # Diversity: are the members actually different from each other?
    spread, dist_sum, dup, n_pairs = [], 0.0, 0, 0
    for i in range(len(syn)):
        ok = rm[i] & np.isfinite(syn[i]).all(axis=0)
        if ok.any() and syn.shape[1] > 1:
            prof = syn[i][:, ok]
            d = np.sqrt(np.mean((prof[:, None, :] - prof[None, :, :]) ** 2, axis=2))
            pv = d[np.triu_indices(syn.shape[1], k=1)]
            dist_sum, dup, n_pairs = dist_sum + float(pv.sum()), dup + int((pv < 1e-3).sum()), n_pairs + len(pv)
            spread.extend(np.std(prof, axis=0).tolist())
    res["ensemble_diversity"] = {
        "mean_within_window_stepwise_std": float(np.mean(spread)) if spread else np.nan,
        "mean_pairwise_profile_rmse": dist_sum / n_pairs if n_pairs else np.nan,
        "near_duplicate_pair_fraction": dup / n_pairs if n_pairs else np.nan}

    flat, flat_m = syn.reshape(-1, syn.shape[2]), sm.reshape(-1, sm.shape[2])
    rr, sr = _ramps(real, rm), _ramps(flat, flat_m)
    res["ramp_distribution"] = {"ks_statistic": ks_statistic(rr, sr, ks_subsample, rng),
                                "overlap_coefficient": overlap_coefficient(rr, sr, bins),
                                "real_mean": float(np.mean(rr)) if len(rr) else np.nan,
                                "synthetic_mean": float(np.mean(sr)) if len(sr) else np.nan}
    if K is not None and n_days is not None:
        rt, st = _daily_totals(real, rm, K, n_days), _daily_totals(flat, flat_m, K, n_days)
        res["daily_total_distribution"] = {"ks_statistic": ks_statistic(rt, st, ks_subsample, rng),
                                           "overlap_coefficient": overlap_coefficient(rt, st, bins),
                                           "real_mean": float(np.mean(rt)) if len(rt) else np.nan,
                                           "synthetic_mean": float(np.mean(st)) if len(st) else np.nan}
    es, vs = _joint_scores(real, rm, syn)
    res["joint_metrics"] = {"energy_score": es, "variogram_score": vs}

    max_lag = min(10, max(1, real.shape[1] - 1))
    ra, sa = _acf(real, rm, max_lag), _acf(flat, flat_m, max_lag)
    r_acf = np.asarray([_mean(ra[:, i]) for i in range(ra.shape[1])])
    s_acf = np.asarray([_mean(sa[:, i]) for i in range(sa.shape[1])])
    res["temporal_autocorrelation"] = {"max_lag": int(max_lag), "real_mean_by_lag": r_acf.tolist(),
                                       "synthetic_mean_by_lag": s_acf.tolist(),
                                       "mae": _mean(np.abs(r_acf - s_acf))}
    rc, sc = _masked_cov(real, rm), _masked_cov(flat, flat_m)
    ok = np.isfinite(rc) & np.isfinite(sc)
    scale = np.nanmean(np.abs(rc[ok])) if ok.any() else np.nan
    mae = float(np.nanmean(np.abs(rc[ok] - sc[ok]))) if ok.any() else np.nan
    res["cross_step_covariance"] = {
        "mae": mae, "normalized_mae": float(mae / max(scale, 1e-12)) if ok.any() and np.isfinite(scale) else np.nan}
    rrough, srough = _roughness(real, rm), _roughness(flat, flat_m)
    res["per_trajectory_temporal_coherence"] = {
        "real_roughness_mean": _mean(rrough), "synthetic_roughness_mean": _mean(srough),
        "roughness_error": abs(_mean(rrough) - _mean(srough)) if len(rrough) and len(srough) else np.nan,
        "lag1_autocorrelation_error": abs(_mean(ra[:, 0]) - _mean(sa[:, 0])) if len(ra) and len(sa) else np.nan}
    # A single diagnostic number: normalised energy + normalised variogram. Lower is better.
    sc_abs = max(float(np.nanmean(np.abs(rv))) if len(rv) else 1.0, 1e-12)
    nes = es / sc_abs if np.isfinite(es) else np.nan
    nvs = vs / (sc_abs ** 2) if np.isfinite(vs) else np.nan
    res["transport_quality"] = {
        "definition": "normalised energy score + normalised variogram score; lower is better; a diagnostic, not a likelihood",
        "normalized_energy_score": nes, "normalized_variogram_score": nvs,
        "joint_transport_proxy": nes + nvs if np.isfinite(nes) and np.isfinite(nvs) else np.nan}

    # Sky-regime mix of the days, real vs generated.
    rcfg = regime_cfg or {}
    real_lab = classify_regimes(real, rm, rcfg)[0]
    syn_lab = np.stack([classify_regimes(syn[:, m, :], rm, rcfg)[0] for m in range(syn.shape[1])])
    r_share = np.bincount(real_lab, minlength=n_regimes).astype(float)
    r_share /= max(r_share.sum(), 1.0)
    s_share = np.mean([np.bincount(row, minlength=n_regimes) / max(len(row), 1) for row in syn_lab], axis=0)
    res["regimes"] = {"n_regimes": int(n_regimes), "real_proportions": r_share.tolist(),
                      "synthetic_proportions_mean": s_share.tolist(),
                      "proportion_mae": float(np.mean(np.abs(r_share - s_share)))}
    if sequence_groups is not None and sequence_ord is not None:
        g, o = np.asarray(sequence_groups), np.asarray(sequence_ord)
        if g.shape != (len(real),) or o.shape != (len(real),):
            raise ValueError("sequence_groups and sequence_ord need one value per row")
        r_tm, r_st = _sequence_transition(real_lab, g, o, n_regimes)
        tms, sts = zip(*[_sequence_transition(lab, g, o, n_regimes) for lab in syn_lab])
        m_tm, m_st = np.mean(tms, axis=0), np.mean(sts, axis=0)
        res["regimes"].update({
            "transition_scope": "day labels, grouped by site and consecutive dates",
            "real_transition_matrix": r_tm.tolist(), "synthetic_transition_matrix_mean": m_tm.tolist(),
            "transition_matrix_mae": float(np.mean(np.abs(m_tm - r_tm))),
            "real_stationary_distribution": r_st.tolist(),
            "synthetic_stationary_distribution_mean": m_st.tolist(),
            "stationary_distribution_mae": float(np.mean(np.abs(m_st - r_st)))})
    else:
        res["regimes"]["transition_scope"] = "not computed; no chronological regime sequence was generated"
    return res


_SECTIONS = ("value_distribution", "regimes", "ensemble_diversity", "joint_metrics", "ramp_distribution",
             "daily_total_distribution", "temporal_autocorrelation", "cross_step_covariance",
             "per_trajectory_temporal_coherence", "transport_quality")


def aggregate_folds(fold_results):
    # Mean of every scalar metric across folds (non-finite values are skipped).
    if not fold_results:
        return {}
    out = {}
    for s in _SECTIONS:
        if not any(s in r for r in fold_results):
            continue
        vals: dict[str, list[float]] = {}
        for r in fold_results:
            for k, v in r.get(s, {}).items():
                if isinstance(v, (int, float)) and np.isfinite(v):
                    vals.setdefault(k, []).append(float(v))
        out[s] = {k: float(np.mean(v)) for k, v in vals.items()}
    out["n_folds"] = len(fold_results)
    return out
