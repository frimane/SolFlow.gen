# Leak-free fit of the Gaussian temporal copula for DeepQuantile.
#
# The copula consumes observations (to compute the PIT), so it may only be fitted
# on the validation windows the DeepQuantile checkpoint was early-stopped on. Those
# windows are the chronological tail of the training stations' data. Before fitting
# we check, and raise if any check fails, that they are disjoint from the test
# windows, share no station with them, precede the test period, and that the
# checkpoint belongs to the requested fold.
from __future__ import annotations

import numpy as np

from ..data.archive import conditions, select
from ..models.copula import GaussianTemporalCopula


def assert_leak_free(model, W, val_idx, test_idx, fold) -> None:
    meta = model.fold_metadata or {}
    if meta.get("fold_id") is not None and int(meta["fold_id"]) != int(fold):
        raise ValueError(f"checkpoint fold_id={meta['fold_id']} does not match requested fold={fold}")
    saved_test = meta.get("test_window_indices")
    if saved_test is not None and len(saved_test) and not np.array_equal(
            np.sort(np.asarray(saved_test, int)), np.sort(np.asarray(test_idx, int))):
        raise ValueError("the DeepQuantile checkpoint was trained for different test windows")
    if np.intersect1d(val_idx, test_idx).size:
        raise ValueError("copula leakage: validation and test windows overlap")
    saved_train = meta.get("train_window_indices")
    if saved_train is not None and len(saved_train) and np.intersect1d(np.asarray(saved_train, int), val_idx).size:
        raise ValueError("copula fit windows overlap the DeepQuantile training windows")
    if "site_id" in W:
        sid = np.asarray(W["site_id"])
        if np.intersect1d(sid[val_idx], sid[test_idx]).size and meta.get("scheme") in (
                "strict_spatiotemporal_holdout", "loso"):
            raise ValueError("copula leakage: a test station appears in the fitting windows")
    dates = np.asarray(W["date_ord"])
    if int(dates[val_idx].max()) >= int(dates[test_idx].min()):
        raise ValueError("copula leakage: fitting windows do not all precede the test period")


def fit_deepquantile_copula(dq, W, test_idx, fold, chunk: int = 1024) -> GaussianTemporalCopula:
    meta = dq.fold_metadata or {}
    val_idx = np.asarray(meta.get("validation_window_indices", []), dtype=int)
    if not len(val_idx):
        raise ValueError("the checkpoint stores no validation windows (trained with split-mode all?); "
                         "a leak-free copula cannot be fitted")
    test_idx = np.asarray(test_idx, dtype=int)
    assert_leak_free(dq, W, val_idx, test_idx, fold)
    view = select(W, val_idx)
    cond, truth, mask = conditions(view), view["fut_csi"], view["fut_mask"]

    def chunks():
        for c0 in range(0, len(val_idx), int(chunk)):
            sl = slice(c0, min(len(val_idx), c0 + int(chunk)))
            curves = dq.predict_quantiles({k: v[sl] for k, v in cond.items()})
            yield curves, truth[sl], mask[sl]

    copula = GaussianTemporalCopula.fit(chunks(), dq.q_levels)
    copula.info.update({
        "fold": int(fold), "n_validation_windows": int(len(val_idx)), "n_test_windows": int(len(test_idx)),
        "validation_years": (np.unique(np.asarray(W["forecast_year"])[val_idx]).astype(int).tolist()
                             if "forecast_year" in W else []),
        "test_years": (np.unique(np.asarray(W["forecast_year"])[test_idx]).astype(int).tolist()
                       if "forecast_year" in W else []),
        "leakage_checks": "passed: validation and test windows disjoint; fitting dates precede test dates"})
    return copula
