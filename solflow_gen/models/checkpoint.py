# Checkpoint files. A checkpoint holds the network weights (tensors) and everything
# else as plain Python types, so it can be read back with
# torch.load(weights_only=True): no pickled code is ever executed on load.
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from ..lazy import torch_nn
from ..transforms.representation import Representation
from ..utils import resolve_device, to_plain
from .deepquantile import DeepQuantile
from .flow import GeometryFlow
from .prior import SeasonalClearSkyPrior
from .regimes import MarkovRegimeSampler, SiteRegimeModel

FORMAT = "solflow_gen_v1"
MAX_BYTES = 4 * 1024 ** 3


def save_model(model, path: str | Path, extra: dict[str, Any] | None = None) -> Path:
    torch, _ = torch_nn()
    path = Path(path)
    payload = {
        "format": FORMAT, "family": model.family, "cfg": to_plain(model.cfg),
        "K": model.K, "n_days": model.n_days, "pad_fill": model.pad_fill,
        "gcs_scale": model.gcs_scale,
        "site_vec": None if model.site_vec is None else model.site_vec.tolist(),
        "rep": to_plain(model.rep.state()),
        "regime_proportions": model.regime_proportions,
        "markov_sampler": None if model.markov_sampler is None else to_plain(model.markov_sampler.state()),
        "site_regime_model": None if model.site_regime_model is None else to_plain(model.site_regime_model.state()),
        "best_epoch": model.best_epoch, "best_val_score": model.best_val_score,
        "best_val_components": to_plain(model.best_val_components),
        "epochs_trained": model.epochs_trained, "selection_metric": model.selection_metric,
        "fold_metadata": to_plain(model.fold_metadata), "extra": to_plain(extra or {}),
        "net_state": {k: v.detach().cpu() for k, v in model.net.state_dict().items()},
    }
    if model.family == "flow":
        payload["prior"] = to_plain(model.prior.state())
    else:
        payload["q_levels"] = model.q_levels.tolist()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path


def load_model(path: str | Path, device: str = "auto"):
    torch, _ = torch_nn()
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    if path.suffix.lower() not in {".pt", ".pth"}:
        raise ValueError("checkpoint must have a .pt or .pth extension")
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("checkpoint exceeds the 4 GB safety limit")
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(ck, dict) or ck.get("format") != FORMAT:
        raise ValueError(f"{path} is not a {FORMAT} checkpoint")
    device = resolve_device(device)
    cfg, rep = ck["cfg"], Representation.from_state(ck["rep"])
    if ck["family"] == "flow":
        prior = SeasonalClearSkyPrior.from_state(ck["prior"], cfg)
        model = GeometryFlow(cfg, ck["K"], ck["n_days"], rep, prior, device)
    elif ck["family"] == "deepquantile":
        model = DeepQuantile(cfg, ck["K"], ck["n_days"], rep, device, n_quantiles=len(ck["q_levels"]))
    else:
        raise ValueError(f"unknown model family {ck['family']!r}")
    model.pad_fill, model.gcs_scale = float(ck["pad_fill"]), ck["gcs_scale"]
    model.site_vec = None if ck["site_vec"] is None else np.asarray(ck["site_vec"], np.float32)
    model.regime_proportions = ck["regime_proportions"]
    model.markov_sampler = MarkovRegimeSampler.from_state(ck["markov_sampler"])
    model.site_regime_model = SiteRegimeModel.from_state(ck["site_regime_model"])
    model.best_epoch, model.best_val_score = ck["best_epoch"], ck["best_val_score"]
    model.best_val_components = dict(ck["best_val_components"])
    model.epochs_trained, model.selection_metric = int(ck["epochs_trained"]), ck["selection_metric"]
    model.fold_metadata = ck["fold_metadata"]
    model.extra = ck.get("extra", {})
    model._build_net(len(ck["q_levels"]) if ck["family"] == "deepquantile" else 1)
    model.net.load_state_dict(ck["net_state"])
    model.net.eval()
    return model
