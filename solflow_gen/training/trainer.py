# Training driver: split the window archive, fit the representation and the prior
# on training rows only, train the Flow and/or DeepQuantile per fold, and write
# checkpoints plus a manifest. Everything a checkpoint needs to be re-evaluated or
# audited (which windows it trained, validated and was tested on) is stored in it.
from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..config import config_hash, validate_config
from ..data.archive import horizon_days, load_windows, select
from ..data.splits import carve_val, make_folds
from ..models.checkpoint import save_model
from ..models.deepquantile import DeepQuantile
from ..models.flow import GeometryFlow
from ..models.prior import SeasonalClearSkyPrior
from ..transforms.representation import make_representation
from ..utils import resolve_device, set_seed, sha256_file, write_json

FAMILIES = ("flow", "deepquantile")


def _prepare_fold(cfg, W, train_idx, test_idx):
    # Carve the validation tail out of the training windows (purged by day interval).
    tr, va = carve_val(cfg, train_idx, W)
    if len(tr) < 10 or len(va) < 1:
        raise ValueError(f"fold has too few windows to train ({len(tr)} train, {len(va)} validation)")
    rows = np.concatenate([tr, va])
    data = select(W, rows)
    return {"train": tr, "val": va, "test": test_idx, "rows": rows, "data": data,
            "local_train": np.arange(len(tr)), "local_val": np.arange(len(tr), len(rows))}


def _fit_representation_and_prior(cfg, split, K, n_days):
    d, lt = split["data"], split["local_train"]
    mask = np.asarray(d["fut_mask"], bool)[lt]
    csi = np.asarray(d["fut_csi"])[lt]
    rep = make_representation(cfg, cfg["representation"]["kind"]).fit(csi[mask])
    prior = SeasonalClearSkyPrior(cfg, K, n_days)
    prior.fit(rep.encode(csi), mask, rep.clearsky_code(), np.asarray(d["date_ord"])[lt])
    return rep, prior


def _metadata(cfg, split, fold_id, scheme):
    return {"fold_id": None if fold_id is None else int(fold_id), "scheme": scheme,
            "train_window_indices": split["train"].astype(int).tolist(),
            "validation_window_indices": split["val"].astype(int).tolist(),
            "test_window_indices": [] if split["test"] is None else np.asarray(split["test"], int).tolist(),
            "val_frac": float(cfg["split"]["val_frac"])}


def train_models(cfg: dict[str, Any], families=FAMILIES, fold: int | None = None) -> dict[str, Any]:
    # Train every requested family on every fold (or one fold). split.train_mode="all"
    # trains one deployment model on all windows instead and names it *_general.pt.
    validate_config(cfg)
    bad = [f for f in families if f not in FAMILIES]
    if bad:
        raise ValueError(f"unknown model families {bad}; choose from {FAMILIES}")
    W = load_windows(cfg["paths"]["windows"])
    K, n_days = horizon_days(W)
    if n_days != int(cfg["task"]["forecast_days"]):
        raise ValueError(f"archive horizon is {n_days} day(s) but task.forecast_days="
                         f"{cfg['task']['forecast_days']}; rebuild the windows")
    if cfg["model"].get("condition_site", True) and "site_coords" not in W:
        print("  note: archive has no site_coords; the models are trained without site conditioning")
    if cfg["model"].get("condition_clearsky_ghi", True) and "fut_ghi_cs" not in W:
        print("  note: archive has no fut_ghi_cs; the models are trained without clear-sky GHI")
    device = resolve_device(cfg["train"].get("device", "auto"))
    out_dir = Path(cfg["paths"]["models_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    scheme = cfg["split"]["scheme"]
    if cfg["split"]["train_mode"] == "all":
        jobs = [(None, np.arange(len(W["fut_csi"])), None)]
    else:
        folds = make_folds(cfg, W)
        if fold is not None and not 0 <= int(fold) < len(folds):
            raise ValueError(f"fold must be between 0 and {len(folds) - 1}")
        jobs = [(i, tr, te) for i, (tr, te) in enumerate(folds) if fold is None or i == int(fold)]

    chash = config_hash(cfg)
    write_json(out_dir / "config.json", cfg)
    entries = []
    for fold_id, train_idx, test_idx in jobs:
        label = "general" if fold_id is None else f"fold{fold_id}"
        split = _prepare_fold(cfg, W, train_idx, test_idx)
        print(f"[train] {label}: {len(split['train'])} train / {len(split['val'])} validation"
              f"{'' if test_idx is None else f' / {len(test_idx)} test'} windows", flush=True)
        rep, prior = _fit_representation_and_prior(cfg, split, K, n_days)
        for family in families:
            seed = int(cfg["seed"]) + (0 if fold_id is None else int(fold_id))
            set_seed(seed)
            rng = np.random.default_rng(seed)
            started = time.time()
            fcfg = copy.deepcopy(cfg)
            if family == "flow":
                model = GeometryFlow(fcfg, K, n_days, rep, prior, device)
                model.fit(split["data"], split["local_train"], split["local_val"], rng=rng)
            else:
                fcfg["model"]["condition_regime"] = False      # the baseline never sees regimes
                model = DeepQuantile(fcfg, K, n_days, rep, device)
                model.fit(split["data"], split["local_train"], split["local_val"], rng=rng)
            model.fold_metadata = _metadata(cfg, split, fold_id, scheme)
            path = save_model(model, out_dir / f"{family}_{label}.pt",
                              extra={"config_hash": chash, "seed": seed,
                                           "noon_col": int(W["noon_col"]) if "noon_col" in W else None})
            entries.append({"family": family, "fold": fold_id, "path": path.name,
                            "sha256": sha256_file(path), "best_epoch": model.best_epoch,
                            "epochs_trained": model.epochs_trained,
                            "best_val_score": model.best_val_score,
                            "selection_metric": model.selection_metric,
                            "seconds": round(time.time() - started, 1)})
            print(f"[train] saved {path}", flush=True)
    manifest = {"config_hash": chash, "scheme": scheme, "windows": str(Path(cfg["paths"]["windows"]).resolve()),
                "device": device, "models": entries}
    write_json(out_dir / "manifest.json", manifest)
    return manifest
