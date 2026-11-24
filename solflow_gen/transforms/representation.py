# Map clear-sky index (CSI) to model space and back.
#
#   z = (rep(csi) - center) / scale
#
# rep is either log, which is log(csi), or gauss, an empirical Gaussianisation
# (CSI -> empirical training CDF -> normal score). Gauss makes the marginals
# standard normal, which takes care of the two modes (clear near 1, cloudy below).
#
# center and scale are fitted on training values only. With center="clearsky",
# CSI = 1 maps to exactly 0, so clouds act as perturbations around zero.
# NaN padding passes through unchanged.
from __future__ import annotations

from typing import Any

import numpy as np

from .stats import norm_cdf, norm_ppf

KINDS = ("log", "gauss")


class Representation:
    def __init__(self, kind: str, center: str, csi_min: float, csi_max: float,
                 rep_cfg: dict[str, Any] | None = None):
        if kind not in KINDS:
            raise ValueError(f"unknown representation {kind!r}; choose from {KINDS}")
        if center not in ("clearsky", "mean"):
            raise ValueError("center must be 'clearsky' or 'mean'")
        if not csi_min > 0.0:
            raise ValueError("csi_min must be > 0 (log transform)")
        rc = rep_cfg or {}
        self.kind, self.center_mode = kind, center
        self.csi_min, self.csi_max = float(csi_min), float(csi_max)
        self.gauss_knots = int(rc.get("gauss_knots", 1001))
        self.gauss_p_clip = float(rc.get("gauss_p_clip", 1e-6))
        self.center, self.scale = 0.0, 1.0
        self._knot_v: np.ndarray | None = None
        self._knot_p: np.ndarray | None = None

    def _to_rep(self, csi) -> np.ndarray:
        csi = np.asarray(csi, dtype=np.float64)
        out = np.full(csi.shape, np.nan)
        ok = np.isfinite(csi)
        if not ok.any():
            return out
        c = np.clip(csi[ok], self.csi_min, self.csi_max)
        if self.kind == "log":
            out[ok] = np.log(c)
        else:
            if self._knot_v is None:
                raise RuntimeError("gauss representation used before fit()")
            p = np.clip(np.interp(c, self._knot_v, self._knot_p),
                        self.gauss_p_clip, 1.0 - self.gauss_p_clip)
            out[ok] = norm_ppf(p)
        return out

    def _from_rep(self, rep) -> np.ndarray:
        rep = np.asarray(rep, dtype=np.float64)
        out = np.full(rep.shape, np.nan)
        ok = np.isfinite(rep)
        if not ok.any():
            return out
        r = rep[ok]
        c = np.exp(r) if self.kind == "log" else np.interp(norm_cdf(r), self._knot_p, self._knot_v)
        out[ok] = np.clip(c, self.csi_min, self.csi_max)
        return out

    def _clearsky_rep(self) -> float:
        return float(self._to_rep(np.array([1.0]))[0])

    def fit(self, csi_train) -> "Representation":
        v = np.asarray(csi_train, dtype=np.float64)
        v = v[np.isfinite(v)]
        if v.size < 10:
            raise ValueError("Representation.fit needs at least 10 valid values")
        if self.kind == "gauss":
            levels = (np.arange(self.gauss_knots) + 0.5) / self.gauss_knots
            knots = np.quantile(np.clip(v, self.csi_min, self.csi_max), levels)
            kv, idx = np.unique(knots, return_index=True)   # strictly increasing abscissa
            kp = levels[idx]
            if kv[0] > self.csi_min:                         # anchor ends at the physical bounds
                kv, kp = np.concatenate([[self.csi_min], kv]), np.concatenate([[0.0], kp])
            if kv[-1] < self.csi_max:
                kv, kp = np.concatenate([kv, [self.csi_max]]), np.concatenate([kp, [1.0]])
            self._knot_v, self._knot_p = kv, kp
        rep = self._to_rep(v)
        rep = rep[np.isfinite(rep)]
        self.scale = float(rep.std() + 1e-8)
        self.center = self._clearsky_rep() if self.center_mode == "clearsky" else float(rep.mean())
        return self

    def encode(self, csi) -> np.ndarray:
        return ((self._to_rep(csi) - self.center) / self.scale).astype(np.float32)

    def decode(self, z) -> np.ndarray:
        rep = np.asarray(z, dtype=np.float64) * self.scale + self.center
        return self._from_rep(rep).astype(np.float32)

    def clearsky_code(self) -> float:
        # Model-space code of CSI = 1.
        return float((self._clearsky_rep() - self.center) / self.scale)

    def state(self) -> dict[str, Any]:
        return {"kind": self.kind, "center_mode": self.center_mode,
                "csi_min": self.csi_min, "csi_max": self.csi_max,
                "center": self.center, "scale": self.scale,
                "gauss_knots": self.gauss_knots, "gauss_p_clip": self.gauss_p_clip,
                "knot_v": self._knot_v, "knot_p": self._knot_p}

    @classmethod
    def from_state(cls, s: dict[str, Any]) -> "Representation":
        r = cls(s["kind"], s["center_mode"], s["csi_min"], s["csi_max"],
                {"gauss_knots": s["gauss_knots"], "gauss_p_clip": s["gauss_p_clip"]})
        r.center, r.scale = float(s["center"]), float(s["scale"])
        r._knot_v = None if s["knot_v"] is None else np.asarray(s["knot_v"], dtype=np.float64)
        r._knot_p = None if s["knot_p"] is None else np.asarray(s["knot_p"], dtype=np.float64)
        return r


def make_representation(cfg: dict[str, Any], kind: str) -> Representation:
    d = cfg["data"]
    return Representation(kind, d.get("center", "clearsky"), d["csi_min"], d["csi_max"],
                          cfg.get("representation"))
