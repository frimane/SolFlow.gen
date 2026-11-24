# Gaussian temporal copula for the DeepQuantile baseline.
#
# DeepQuantile gives the marginal distribution at each step but nothing about how
# steps depend on each other. To get whole-day scenarios the marginals have to be
# coupled. Independent uniforms are the trivial option; this module fits a Gaussian
# copula instead. Fitting works on validation windows only:
#
#   1. PIT: u = F(obs), with F the piecewise-linear CDF through the predicted
#      quantile curve (the inverse of the scenario interpolation).
#   2. Normal scores: z = Phi^-1(u).
#   3. Correlation R between steps, from pairs of steps that are both observed.
#      Pairs with too few common windows get the mean correlation of reliable pairs
#      at the same lag. R is then repaired to a valid correlation matrix.
#   4. Sampling: eps ~ N(0, I), z = L eps with L L^T = R, U = Phi(z), then each
#      step's quantile curve is read at U.
#
# The fit uses observations, so the windows passed in must share no station with
# the test windows and come strictly before them in time. See
# evaluation/copula_fit.py, which checks this.
from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from ..transforms.stats import norm_cdf, norm_ppf


class GaussianTemporalCopula:
    def __init__(self, correlation, q_levels, info: dict[str, Any] | None = None):
        corr = np.asarray(correlation, dtype=np.float64)
        if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
            raise ValueError("correlation must be a square matrix")
        self.correlation = corr
        self.q_levels = np.asarray(q_levels, dtype=np.float64)
        self.info = dict(info or {})
        vals, vecs = np.linalg.eigh(corr)
        self._factor = vecs * np.sqrt(np.maximum(vals, 0.0))[None, :]

    @property
    def horizon(self) -> int:
        return int(self.correlation.shape[0])

    @staticmethod
    def pit_from_quantiles(curves, levels, obs) -> np.ndarray:
        # PIT of obs [N, T] under quantile curves [N, M, T] at levels [M].
        #
        # Ties (e.g. at the CSI clipping bounds) take the mean of the tied levels.
        # Observations below the lowest / above the highest quantile get the
        # mid-point of the remaining tail probability.
        curves = np.sort(np.asarray(curves, dtype=np.float64), axis=1)
        levels = np.asarray(levels, dtype=np.float64)
        obs = np.asarray(obs, dtype=np.float64)
        n, m, t = curves.shape
        if obs.shape != (n, t) or levels.shape != (m,):
            raise ValueError("curves must be [N, M, T], obs [N, T] and levels [M]")
        o = obs[:, None, :]
        lt = (curves < o).sum(axis=1)                  # quantiles strictly below
        le = (curves <= o).sum(axis=1)                 # quantiles at or below
        csum = np.concatenate([[0.0], np.cumsum(levels)])
        u = np.empty((n, t), dtype=np.float64)

        tie = le > lt
        if tie.any():
            u[tie] = (csum[le[tie]] - csum[lt[tie]]) / (le[tie] - lt[tie])
        below = (~tie) & (lt == 0)
        above = (~tie) & (lt == m)
        mid = (~tie) & (~below) & (~above)
        u[below] = 0.5 * levels[0]
        u[above] = 0.5 * (1.0 + levels[-1])
        if mid.any():
            hi_idx = np.clip(lt, 1, m - 1)
            hi = np.take_along_axis(curves, hi_idx[:, None, :], axis=1)[:, 0, :]
            lo = np.take_along_axis(curves, hi_idx[:, None, :] - 1, axis=1)[:, 0, :]
            q_hi, q_lo = levels[hi_idx], levels[hi_idx - 1]
            with np.errstate(divide="ignore", invalid="ignore"):
                frac = (obs - lo) / (hi - lo)
            u[mid] = (q_lo + frac * (q_hi - q_lo))[mid]
        return u

    @classmethod
    def fit(cls, chunks: Iterable[tuple], q_levels, min_pairs: int = 30) -> "GaussianTemporalCopula":
        # Estimate the copula from an iterable of (curves, truth, mask) chunks.
        #
        # curves [N, M, T] are predicted quantile curves in CSI space, truth
        # [N, T] the observed CSI and mask [N, T] the valid steps. Sufficient
        # statistics are accumulated chunk by chunk, so memory stays bounded.
        levels = np.asarray(q_levels, dtype=np.float64)
        cnt = sab = sa = saa = None
        n_windows = n_valid = n_extreme = 0
        z_sum = z_sq = 0.0
        for curves, truth, mask in chunks:
            truth = np.asarray(truth, dtype=np.float64)
            valid = (np.asarray(mask, dtype=bool) & np.isfinite(truth)
                     & np.isfinite(np.asarray(curves)).all(axis=1))
            u = np.clip(cls.pit_from_quantiles(curves, levels, np.where(valid, truth, 0.0)),
                        1e-6, 1.0 - 1e-6)
            z = np.where(valid, norm_ppf(u), 0.0)
            v = valid.astype(np.float64)
            if cnt is None:
                t = z.shape[1]
                cnt, sab, sa, saa = (np.zeros((t, t)) for _ in range(4))
            cnt += v.T @ v
            sab += z.T @ z
            sa += z.T @ v
            saa += (z * z).T @ v
            n_windows += int(len(z))
            n_valid += int(valid.sum())
            z_sum += float(z[valid].sum())
            z_sq += float((z[valid] ** 2).sum())
            n_extreme += int(((u[valid] <= levels[0]) | (u[valid] >= levels[-1])).sum())
        if cnt is None or n_valid == 0:
            raise ValueError("copula fit received no valid validation steps")

        with np.errstate(divide="ignore", invalid="ignore"):
            mean_a, mean_b = sa / cnt, sa.T / cnt
            var_a, var_b = saa / cnt - mean_a ** 2, saa.T / cnt - mean_b ** 2
            corr = (sab / cnt - mean_a * mean_b) / np.sqrt(var_a * var_b)
        thin = cnt < float(min_pairs)
        reliable = np.isfinite(corr) & ~thin
        corr = np.where(reliable, corr, np.nan)
        corr = np.clip(0.5 * (corr + corr.T), -0.999, 0.999)

        # Thin pairs (typically day-edge steps that exist only in some seasons)
        # take the mean correlation of reliable pairs at the same step lag; lags
        # without any reliable pair get 0.
        n_steps = len(corr)
        filled = np.zeros(corr.shape, dtype=bool)
        for lag in range(1, n_steps):
            diag = np.diag(corr, lag).copy()
            ok = np.isfinite(diag)
            fill = float(diag[ok].mean()) if ok.any() else 0.0
            for r in np.arange(n_steps - lag)[~ok]:
                corr[r, r + lag] = corr[r + lag, r] = fill
                filled[r, r + lag] = filled[r + lag, r] = True
        np.fill_diagonal(corr, 1.0)
        corr, min_eig = cls.repair_correlation(corr)

        off = ~np.eye(n_steps, dtype=bool)
        info = {
            "fit_split": "validation",
            "n_fit_windows": int(n_windows),
            "n_fit_valid_steps": int(n_valid),
            "min_common_windows_per_pair": float(cnt[off].min()) if off.any() else float(cnt.min()),
            "share_pairs_below_min_pairs": float(thin[off].mean()) if off.any() else 0.0,
            "share_pairs_filled_by_lag_average": float(filled[off].mean()) if off.any() else 0.0,
            "min_pairs": int(min_pairs),
            "min_eigenvalue_before_repair": float(min_eig),
            "normal_score_mean": float(z_sum / n_valid),
            "normal_score_std": float(np.sqrt(max(z_sq / n_valid - (z_sum / n_valid) ** 2, 0.0))),
            "share_pit_outside_quantile_range": float(n_extreme / n_valid),
            "mean_lag1_correlation": float(np.mean(np.diag(corr, 1))) if n_steps > 1 else float("nan"),
        }
        return cls(corr, levels, info)

    @staticmethod
    def repair_correlation(corr, floor: float = 1e-6):
        # Nearest positive-definite correlation matrix by eigenvalue clipping.
        corr = 0.5 * (np.asarray(corr, dtype=np.float64) + np.asarray(corr).T)
        vals, vecs = np.linalg.eigh(corr)
        min_eig = float(vals.min())
        fixed = (vecs * np.maximum(vals, floor)[None, :]) @ vecs.T
        d = np.sqrt(np.clip(np.diag(fixed), 1e-12, None))
        fixed = fixed / np.outer(d, d)
        fixed = 0.5 * (fixed + fixed.T)
        np.fill_diagonal(fixed, 1.0)
        return fixed, min_eig

    def uniforms(self, n_rows: int, n_scenarios: int, rng: np.random.Generator) -> np.ndarray:
        # Correlated uniforms [n_rows, n_scenarios, T]; margins are U(0, 1).
        eps = rng.standard_normal((int(n_rows), int(n_scenarios), self.horizon))
        return norm_cdf(eps @ self._factor.T)

    def state(self) -> dict[str, Any]:
        return {"correlation": self.correlation.tolist(),
                "q_levels": self.q_levels.tolist(), "info": self.info}

    @classmethod
    def from_state(cls, s) -> "GaussianTemporalCopula | None":
        return None if s is None else cls(s["correlation"], s["q_levels"], s.get("info"))
