# Sky-regime utilities for the generator's optional regime variable.
#
# Four daily states are derived from the observed daylight CSI trajectory:
# clear, broken_cloud, overcast and high_variability.
#
# Training labels come from training targets only. At generation time no
# observation exists, so regimes are sampled from persisted, train-only
# statistics: site-specific (and seasonal) proportions with a day-to-day Markov
# transition matrix, shrunk toward the pooled model for sparse sites.
from __future__ import annotations

from typing import Any

import numpy as np

REGIME_NAMES = ("clear", "broken_cloud", "overcast", "high_variability")


def daily_regime_features(csi, mask, cfg: dict[str, Any] | None = None) -> np.ndarray:
    # Per-row [mean, std, clear_fraction, valid_fraction] over valid steps.
    x = np.asarray(csi, dtype=np.float32)
    ok = np.asarray(mask, dtype=bool) & np.isfinite(x)
    count = ok.sum(axis=1).astype(np.float32)
    mean = np.where(ok, x, 0.0).sum(axis=1) / np.maximum(count, 1.0)
    centred = np.where(ok, x - mean[:, None], 0.0)
    std = np.sqrt((centred * centred).sum(axis=1) / np.maximum(count, 1.0))
    clear_thr = float((cfg or {}).get("clear_csi_min", 0.90))
    clear_fraction = (ok & (x >= clear_thr)).sum(axis=1) / np.maximum(count, 1.0)
    out = np.stack([mean, std, clear_fraction, ok.mean(axis=1)], axis=1)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def classify_regimes(csi, mask, cfg: dict[str, Any] | None = None):
    # Label each row with a regime in {0: clear, 1: broken, 2: overcast, 3: variable}.
    c = cfg or {}
    f = daily_regime_features(csi, mask, c)
    mean, std = f[:, 0], f[:, 1]
    overcast_max = float(c.get("overcast_mean_max", 0.45))
    labels = np.ones(len(mean), dtype=np.int64)
    labels[(mean >= float(c.get("clear_mean_min", 0.90))) & (std <= float(c.get("clear_std_max", 0.08)))] = 0
    labels[mean <= overcast_max] = 2
    labels[(std >= float(c.get("high_variability_std_min", 0.22))) & (mean > overcast_max)] = 3
    return labels, f


def regime_proportions(labels, n_regimes: int = 4) -> np.ndarray:
    y = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(y[(y >= 0) & (y < n_regimes)], minlength=n_regimes)
    total = counts.sum()
    return (counts / total).astype(np.float32) if total else np.full(n_regimes, 1.0 / n_regimes, np.float32)


def transition_counts(labels, n_regimes: int = 4) -> np.ndarray:
    y = np.asarray(labels, dtype=np.int64)
    counts = np.zeros((n_regimes, n_regimes), dtype=np.float64)
    for a, b in zip(y[:-1], y[1:]):
        if 0 <= a < n_regimes and 0 <= b < n_regimes:
            counts[a, b] += 1
    return counts


def _normalize_rows(matrix) -> np.ndarray:
    m = np.maximum(np.asarray(matrix, dtype=np.float64), 0.0)
    sums = m.sum(axis=1, keepdims=True)
    return np.divide(m, sums, out=np.full_like(m, 1.0 / m.shape[1]), where=sums > 0).astype(np.float32)


def transition_matrix(labels, n_regimes: int = 4, smoothing: float = 0.0) -> np.ndarray:
    counts = transition_counts(labels, n_regimes)
    if smoothing > 0:
        counts += float(smoothing)
    return _normalize_rows(counts)


def stationary_distribution(transition) -> np.ndarray:
    p = np.asarray(transition, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] != p.shape[1]:
        raise ValueError("transition must be a square matrix")
    vals, vecs = np.linalg.eig(p.T)
    v = np.abs(np.real(vecs[:, int(np.argmin(np.abs(vals - 1.0)))]))
    return (v / v.sum()).astype(np.float32) if v.sum() else np.full(p.shape[0], 1.0 / p.shape[0], np.float32)


def allocate_regimes(n_rows: int, proportions, rng: np.random.Generator) -> np.ndarray:
    # Shuffled labels whose counts follow proportions as exactly as integers allow.
    p = np.asarray(proportions, dtype=np.float64)
    p = p / p.sum() if p.sum() > 0 else np.full(len(p), 1.0 / len(p))
    raw = p * int(n_rows)
    counts = np.floor(raw).astype(int)
    remainder = int(n_rows) - int(counts.sum())
    if remainder:
        counts[np.argsort(-(raw - counts))[:remainder]] += 1
    labels = (np.concatenate([np.full(c, i, np.int64) for i, c in enumerate(counts)])
              if n_rows else np.empty(0, np.int64))
    rng.shuffle(labels)
    return labels


def _month(day_ord) -> int:
    return min(11, int(((int(day_ord) - 1) % 365) // 30))


class SiteRegimeModel:
    # Site-aware regime proportions (seasonal) and transitions.
    #
    # Known sites use exact coordinate keys; unseen coordinates use
    # distance-weighted interpolation; sparse sites shrink toward the pooled model.

    def __init__(self, n_regimes=4, alpha_proportions=12.0, alpha_transition=12.0,
                 coordinate_scale=None, sites=None, global_proportions=None,
                 global_transition=None, seasonal_proportions=None):
        self.n_regimes = int(n_regimes)
        self.alpha_proportions = float(alpha_proportions)
        self.alpha_transition = float(alpha_transition)
        self.coordinate_scale = np.asarray(
            coordinate_scale if coordinate_scale is not None else [1.0] * 4, np.float32)
        self.sites = sites or []
        self.global_proportions = np.asarray(
            global_proportions if global_proportions is not None
            else np.full(self.n_regimes, 1.0 / self.n_regimes), np.float32)
        self.global_transition = _normalize_rows(
            global_transition if global_transition is not None
            else np.ones((self.n_regimes, self.n_regimes)))
        self.global_seasonal = np.asarray(
            seasonal_proportions if seasonal_proportions is not None
            else np.tile(self.global_proportions, (12, 1)), np.float32)

    @staticmethod
    def _key(coord) -> tuple:
        return tuple(np.round(np.asarray(coord, np.float32).reshape(-1), 6).tolist())

    @classmethod
    def fit(cls, labels, site_coords, n_regimes=4, alpha_proportions=12.0, alpha_transition=12.0,
            coordinate_scale=None, date_ord=None) -> "SiteRegimeModel":
        y = np.asarray(labels, np.int64).reshape(-1)
        coords = np.asarray(site_coords, np.float32)
        if coords.ndim != 2 or len(coords) != len(y):
            raise ValueError("site_coords must be [n_rows, n_features] aligned with labels")
        valid = (y >= 0) & (y < n_regimes) & np.isfinite(coords).all(axis=1)
        if not valid.any():
            return cls(n_regimes, alpha_proportions, alpha_transition, coordinate_scale,
                       global_proportions=regime_proportions(y, n_regimes),
                       global_transition=transition_matrix(y, n_regimes, alpha_transition))
        y, coords = y[valid], coords[valid]
        ords = None if date_ord is None else np.asarray(date_ord).reshape(-1)[valid]
        gp = regime_proportions(y, n_regimes).astype(np.float64)
        gt = transition_matrix(y, n_regimes, alpha_transition)
        scale = np.asarray(coordinate_scale if coordinate_scale is not None
                           else np.maximum(np.std(coords, axis=0), 0.25), np.float32)
        if scale.ndim != 1 or len(scale) != coords.shape[1] or np.any(scale <= 0):
            raise ValueError("coordinate_scale must match site_coords and be positive")
        months = None if ords is None else np.minimum(((ords.astype(np.int64) - 1) % 365) // 30, 11)
        global_seasonal = np.tile(gp, (12, 1))
        if months is not None:
            for m in range(12):
                sel = months == m
                if sel.any():
                    global_seasonal[m] = regime_proportions(y[sel], n_regimes)
        sites = []
        for key in sorted({cls._key(c) for c in coords}):
            sel = np.array([cls._key(c) == key for c in coords])
            sy = y[sel]
            pc = np.bincount(sy, minlength=n_regimes).astype(np.float64)
            p = (pc + alpha_proportions * gp) / (len(sy) + alpha_proportions)
            tc = transition_counts(sy, n_regimes)
            t = (tc + alpha_transition * gt) / (tc.sum(axis=1, keepdims=True) + alpha_transition)
            seasonal = np.tile(gp, (12, 1)).astype(np.float64)
            if months is not None:
                for m in range(12):
                    sm = sel & (months == m)
                    if sm.any():
                        counts = np.bincount(y[sm], minlength=n_regimes).astype(np.float64)
                        seasonal[m] = ((counts + alpha_proportions * global_seasonal[m])
                                       / (counts.sum() + alpha_proportions))
            sites.append({"key": list(key), "coords": np.asarray(coords[sel][0], np.float32).tolist(),
                          "n_days": int(len(sy)), "proportions": p.tolist(),
                          "seasonal_proportions": seasonal.tolist(),
                          "transition": _normalize_rows(t).tolist()})
        return cls(n_regimes, alpha_proportions, alpha_transition, scale, sites,
                   gp.astype(np.float32), gt, global_seasonal.astype(np.float32))

    def _params(self, coord, day_ord=None):
        month = None if day_ord is None else _month(day_ord)
        if not self.sites:
            p = self.global_proportions if month is None else self.global_seasonal[month]
            return p, self.global_transition
        q = np.asarray(coord, np.float32).reshape(-1)
        if q.size != len(self.coordinate_scale):
            raise ValueError(f"coordinate has {q.size} features; expected {len(self.coordinate_scale)}")
        d = np.array([np.linalg.norm((q - np.asarray(s["coords"], np.float32)) / self.coordinate_scale)
                      for s in self.sites])
        nearest = int(np.argmin(d))

        def site_p(s):
            seasonal = s.get("seasonal_proportions")
            if month is not None and seasonal is not None:
                return np.asarray(seasonal, np.float64)[month]
            return np.asarray(s["proportions"], np.float64)

        if d[nearest] <= 1e-6:
            s = self.sites[nearest]
            return site_p(s).astype(np.float32), np.asarray(s["transition"], np.float32)
        bandwidth = max(float(np.median(d)), 0.35)
        w = np.exp(-0.5 * (d / bandwidth) ** 2)
        if not np.isfinite(w).all() or w.sum() <= 1e-12:
            return self.global_proportions, self.global_transition
        w /= w.sum()
        p = sum(wi * site_p(s) for wi, s in zip(w, self.sites))
        t = sum(wi * np.asarray(s["transition"], np.float64) for wi, s in zip(w, self.sites))
        return np.asarray(p, np.float32), _normalize_rows(t)

    def parameters_for(self, coords, day_ord=None):
        c = np.asarray(coords, np.float32)
        if c.ndim == 1:
            c = c[None, :]
        d = None if day_ord is None else np.asarray(day_ord).reshape(-1)
        values = [self._params(row, None if d is None else d[i]) for i, row in enumerate(c)]
        return (np.asarray([v[0] for v in values], np.float32),
                np.asarray([v[1] for v in values], np.float32))

    def sample_independent(self, coords, rng, day_ord=None) -> np.ndarray:
        # One regime per row, drawn from that row's site/season proportions.
        p, _ = self.parameters_for(coords, day_ord=day_ord)
        return np.asarray([rng.choice(self.n_regimes, p=row / row.sum()) for row in p], np.int64)

    def sample_sequence(self, coords, rng, initial_state=None, day_ord=None) -> np.ndarray:
        # A connected regime sequence along consecutive rows (chronological order).
        c = np.asarray(coords, np.float32)
        if c.ndim != 2 or len(c) == 0:
            return np.empty(0, np.int64)
        p, t = self.parameters_for(c, day_ord=day_ord)
        out = np.empty(len(c), np.int64)
        out[0] = (int(rng.choice(self.n_regimes, p=p[0] / p[0].sum()))
                  if initial_state is None else int(initial_state))
        if not 0 <= out[0] < self.n_regimes:
            raise ValueError("initial_state is outside the regime range")
        for i in range(1, len(c)):
            row = t[i - 1, out[i - 1]]
            out[i] = int(rng.choice(self.n_regimes, p=row / row.sum()))
        return out

    def state(self) -> dict[str, Any]:
        return {"n_regimes": self.n_regimes, "alpha_proportions": self.alpha_proportions,
                "alpha_transition": self.alpha_transition,
                "coordinate_scale": self.coordinate_scale.tolist(), "sites": self.sites,
                "global_proportions": self.global_proportions.tolist(),
                "global_transition": self.global_transition.tolist(),
                "global_seasonal": self.global_seasonal.tolist()}

    @classmethod
    def from_state(cls, state) -> "SiteRegimeModel | None":
        if state is None:
            return None
        return cls(state["n_regimes"], state["alpha_proportions"], state["alpha_transition"],
                   state["coordinate_scale"], state.get("sites", []), state["global_proportions"],
                   state["global_transition"], state.get("global_seasonal"))


class MarkovRegimeSampler:
    # Pooled day-to-day Markov chain, used when no site model / coordinates exist.

    def __init__(self, transition, initial=None):
        self.transition = _normalize_rows(transition)
        if initial is None:
            initial = stationary_distribution(self.transition)
        initial = np.asarray(initial, np.float64)
        self.initial = (initial / initial.sum()).astype(np.float32)

    @classmethod
    def from_labels(cls, labels, n_regimes=4, smoothing=1.0) -> "MarkovRegimeSampler":
        return cls(transition_matrix(labels, n_regimes, smoothing), regime_proportions(labels, n_regimes))

    def sample(self, n_steps: int, rng, initial_state=None) -> np.ndarray:
        n_steps = int(n_steps)
        if n_steps <= 0:
            return np.empty(0, np.int64)
        state = (int(rng.choice(len(self.initial), p=self.initial))
                 if initial_state is None else int(initial_state))
        if not 0 <= state < len(self.initial):
            raise ValueError("initial_state is outside the regime range")
        out = np.empty(n_steps, np.int64)
        out[0] = state
        for i in range(1, n_steps):
            out[i] = int(rng.choice(len(self.initial), p=self.transition[out[i - 1]]))
        return out

    def state(self) -> dict[str, Any]:
        return {"transition": self.transition.tolist(), "initial": self.initial.tolist()}

    @classmethod
    def from_state(cls, state) -> "MarkovRegimeSampler | None":
        return None if state is None else cls(state["transition"], state.get("initial"))
