# Seasonal clear-sky prior, the source distribution of the flow.
#
# A source trajectory is x0 = anchor + noise.
#
# anchor: mean profile of the training targets in model space, one K-column
#   profile per 30-day month of the year. With clear-sky centring it sits near 0
#   (expect clear sky). It follows the calendar date, so training and generation
#   use the same seasonal reference.
# noise: correlated Gaussian noise, independent between days. Per day the
#   covariance is D C D + jitter * I, where C is a Matern or RBF correlation
#   (length scale solved from the lag-1 autocorrelation of the training residuals)
#   and D holds the per-column residual standard deviations.
# burst (optional): with probability `weight` a one-sided term pushes the source
#   above clear sky, for cloud enhancement.
#
# Night (padded) steps get no noise and sit on the anchor.
from __future__ import annotations

from typing import Any

import numpy as np

KERNELS = ("matern12", "matern32", "rbf")


def _corr(kind: str, lag, length_scale: float):
    l = max(float(length_scale), 1e-3)
    if kind == "rbf":
        return np.exp(-0.5 * (lag / l) ** 2)
    if kind == "matern12":
        return np.exp(-lag / l)
    if kind == "matern32":
        r = np.sqrt(3.0) * lag / l
        return (1.0 + r) * np.exp(-r)
    raise ValueError(f"unknown kernel {kind!r}")


def _corr_matrix(kind: str, K: int, length_scale: float) -> np.ndarray:
    idx = np.arange(K)
    return _corr(kind, np.abs(idx[:, None] - idx[None, :]).astype(float), length_scale)


def _solve_length_scale(kind: str, rho1: float, K: int) -> float:
    # Bisection for the length scale whose lag-1 correlation equals rho1.
    lo, hi = 0.05, float(max(K, 2))
    f = lambda l: float(_corr(kind, np.array([1.0]), l)[0])
    if rho1 <= f(lo):
        return lo
    if rho1 >= f(hi):
        return hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) < rho1 else (lo, mid)
    return 0.5 * (lo + hi)


def _masked_lag1(resid, fut_mask, K: int) -> float:
    resid = np.asarray(resid, np.float64)
    fm = np.asarray(fut_mask, bool)
    num = den = 0.0
    for b in range(resid.shape[1] // K):
        blk, bm = resid[:, b * K:(b + 1) * K], fm[:, b * K:(b + 1) * K]
        if not bm.any():
            continue
        mu = float(np.nanmean(np.where(bm, blk, np.nan)))
        blk = blk - (mu if np.isfinite(mu) else 0.0)
        a, c = blk[:, :-1], blk[:, 1:]
        pair = bm[:, :-1] & bm[:, 1:] & np.isfinite(a) & np.isfinite(c)
        num += float((a[pair] * c[pair]).sum())
        den += float((blk[bm & np.isfinite(blk)] ** 2).sum())
    return 0.9 if den <= 0 else float(np.clip(num / den, 1e-3, 0.999))


def month_index(day_ord) -> np.ndarray:
    # 30-day month bucket (0..11) of day ordinals; the same rule is used by regimes.
    d = np.asarray(day_ord, dtype=np.int64)
    return np.minimum(((d - 1) % 365) // 30, 11)


class SeasonalClearSkyPrior:
    def __init__(self, cfg: dict[str, Any], K: int, n_days: int):
        self.cfg = cfg
        self.K, self.n_days = int(K), int(n_days)
        self.H = self.K * self.n_days
        self._clearsky_code: float | None = None
        self._clim_seasonal: np.ndarray | None = None   # [12, K] month-conditioned mean profile
        self._L: np.ndarray | None = None               # Cholesky of Sigma_K
        self._Le: np.ndarray | None = None              # Cholesky of the burst kernel
        self._enh_weight = 0.0
        self.fit_info: dict[str, Any] = {}

    @property
    def clearsky_code(self) -> float:
        if self._clearsky_code is None:
            raise RuntimeError("prior is not fitted; call fit() first")
        return self._clearsky_code

    def anchor(self, date_ord) -> np.ndarray:
        # Seasonal mean profile for each row's forecast days: [N, H].
        if self._clim_seasonal is None:
            raise RuntimeError("prior is not fitted; call fit() first")
        dates = np.asarray(date_ord).reshape(-1).astype(np.int64)
        days = dates[:, None] + np.arange(self.n_days)[None, :]          # [N, n_days]
        return self._clim_seasonal[month_index(days)].reshape(len(dates), self.H)

    def fit(self, x1_train, fut_mask, clearsky_code: float, date_ord) -> "SeasonalClearSkyPrior":
        # Fit on encoded training targets x1_train [N, H] (training rows only).
        if date_ord is None:
            raise ValueError("the seasonal prior needs date_ord for every training window")
        self._clearsky_code = float(clearsky_code)
        x1 = np.asarray(x1_train, np.float64)
        fm = np.asarray(fut_mask, bool)
        ords = np.asarray(date_ord).reshape(-1)
        if len(ords) != len(x1):
            raise ValueError("date_ord must align with the training windows")

        # per-column mean profile; fallback = clear-sky code for never-seen columns
        clim = np.full(self.K, self._clearsky_code)
        for k in range(self.K):
            vals = x1[:, k::self.K][fm[:, k::self.K]]
            vals = vals[np.isfinite(vals)]
            if vals.size:
                clim[k] = float(vals.mean())
        seasonal = np.tile(clim, (12, 1))
        months = month_index(ords)
        for m in range(12):
            sel = months == m
            if not sel.any():
                continue
            for k in range(self.K):
                vals = x1[sel, k::self.K][fm[sel, k::self.K]]
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    seasonal[m, k] = float(vals.mean())
        self._clim_seasonal = seasonal

        resid = x1 - self.anchor(ords)
        valid = fm & np.isfinite(resid)
        pooled_std = max(float(np.std(resid[valid])) if valid.any() else 1.0, 1e-4)
        pc = self.cfg["prior"]
        floor = float(pc.get("col_std_floor_frac", 0.10)) * pooled_std
        col_std = np.full(self.K, pooled_std)
        for k in range(self.K):
            vals = resid[:, k::self.K][valid[:, k::self.K]]
            if vals.size >= 10:
                col_std[k] = float(np.std(vals))
        col_std = np.maximum(col_std, floor)

        # correlation length from the lag-1 autocorrelation of the standardised
        # residual, so heteroscedasticity does not bleed into the estimate
        rho1 = _masked_lag1(resid / np.tile(col_std, self.n_days)[None, :], fm, self.K)
        length_scale = _solve_length_scale(pc["kernel"], rho1, self.K)
        C = _corr_matrix(pc["kernel"], self.K, length_scale)
        Sk = col_std[:, None] * C * col_std[None, :] + float(pc["jitter"]) * np.eye(self.K)
        self._L = np.linalg.cholesky(Sk)

        ec = pc.get("enhancement") or {}
        self._enh_weight = float(ec.get("weight", 0.0) or 0.0)
        self._Le = None
        if self._enh_weight > 0:
            amp = float(ec.get("amplitude_frac", 0.5)) * pooled_std
            Se = amp ** 2 * _corr_matrix(pc["kernel"], self.K, float(ec.get("length_scale", 1.0))) \
                + float(pc["jitter"]) * np.eye(self.K)
            self._Le = np.linalg.cholesky(Se)

        self.fit_info = {"rho1": rho1, "length_scale": length_scale, "pooled_std": pooled_std,
                         "col_std_min": float(col_std.min()), "col_std_max": float(col_std.max()),
                         "enhancement_weight": self._enh_weight}
        return self

    def sample(self, n: int, rng: np.random.Generator, fut_mask, fut_date_ord) -> np.ndarray:
        # Draw n source trajectories [n, H] (float32).
        if self._L is None:
            raise RuntimeError("prior is not fitted; call fit() first")
        fm = np.asarray(fut_mask, bool)
        if fm.shape != (n, self.H):
            raise ValueError(f"fut_mask shape {fm.shape} != {(n, self.H)}")
        if fut_date_ord is None or len(np.asarray(fut_date_ord).reshape(-1)) != n:
            raise ValueError("fut_date_ord must give one date per sampled row")
        eps = rng.standard_normal((n, self.n_days, self.K))
        noise = np.einsum("nbk,jk->nbj", eps, self._L).reshape(n, self.H)
        if self._enh_weight > 0 and self._Le is not None:
            fire = rng.random(n) < self._enh_weight
            if fire.any():
                ee = rng.standard_normal((int(fire.sum()), self.n_days, self.K))
                noise[fire] += np.abs(np.einsum("nbk,jk->nbj", ee, self._Le)).reshape(-1, self.H)
        anchor = self.anchor(fut_date_ord)
        out = np.where(fm, anchor + noise, anchor)       # padded steps: no noise
        return out.astype(np.float32)

    def state(self) -> dict[str, Any]:
        return {"K": self.K, "n_days": self.n_days, "clearsky_code": self._clearsky_code,
                "clim_seasonal": self._clim_seasonal, "L": self._L, "Le": self._Le,
                "enh_weight": self._enh_weight, "fit_info": self.fit_info}

    @classmethod
    def from_state(cls, s: dict[str, Any], cfg: dict[str, Any]) -> "SeasonalClearSkyPrior":
        p = cls(cfg, s["K"], s["n_days"])
        p._clearsky_code = None if s["clearsky_code"] is None else float(s["clearsky_code"])
        p._clim_seasonal = None if s["clim_seasonal"] is None else np.asarray(s["clim_seasonal"], np.float64)
        p._L = None if s["L"] is None else np.asarray(s["L"], np.float64)
        p._Le = None if s["Le"] is None else np.asarray(s["Le"], np.float64)
        p._enh_weight = float(s.get("enh_weight", 0.0))
        p.fit_info = dict(s.get("fit_info", {}))
        return p
