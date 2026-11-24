# Standard-normal CDF and inverse CDF in pure NumPy (no SciPy dependency).
#
# The CDF uses the Abramowitz & Stegun 7.1.26 erf approximation (abs. error
# < 1.5e-7). The inverse uses Acklam's rational approximation refined by one
# Halley step.
from __future__ import annotations

import numpy as np


def norm_cdf(z) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    x = np.abs(z) / np.sqrt(2.0)
    t = 1.0 / (1.0 + 0.3275911 * x)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741
                + t * (-1.453152027 + t * 1.061405429))))
    erf = 1.0 - poly * np.exp(-x * x)
    return 0.5 * (1.0 + np.sign(z) * erf)


_A = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
      1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
_B = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
      6.680131188771972e+01, -1.328068155288572e+01]
_C = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
      -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
_D = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
      3.754408661907416e+00]


def norm_ppf(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-300, 1.0 - 1e-16)
    x = np.empty_like(p)
    plow, phigh = 0.02425, 1.0 - 0.02425
    lo, hi = p < plow, p > phigh
    mid = ~(lo | hi)

    def tail(q):
        return ((((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) /
                ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0))

    if lo.any():
        x[lo] = tail(np.sqrt(-2.0 * np.log(p[lo])))
    if hi.any():
        x[hi] = -tail(np.sqrt(-2.0 * np.log(1.0 - p[hi])))
    if mid.any():
        q = p[mid] - 0.5
        r = q * q
        x[mid] = ((((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q /
                  (((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0))
    e = norm_cdf(x) - p
    u = e * np.sqrt(2.0 * np.pi) * np.exp(0.5 * x * x)
    return x - u / (1.0 + 0.5 * x * u)
