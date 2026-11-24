# Machinery shared by the Flow and the DeepQuantile baseline: conditioning
# channels, device caching, EMA weights, validation-row selection and the ODE
# integrator. Both models use identical conditioning, so any difference in
# results comes from the model family and loss, not from the inputs.
from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from ..lazy import torch_nn
from .network import build_network
from .regimes import (MarkovRegimeSampler, SiteRegimeModel, allocate_regimes,
                      regime_proportions)


class EMAWeights:
    # Exponential moving average of the weights, with the usual warm-up ramp so a
    # short run is not dragged down by the random initialisation.
    def __init__(self, net, decay: float):
        self.decay = float(decay)
        self.step = 0
        self.shadow = {k: v.detach().clone() for k, v in net.state_dict().items()}

    def update(self, net) -> None:
        torch, _ = torch_nn()
        self.step += 1
        d = min(self.decay, (1.0 + self.step) / (10.0 + self.step))
        with torch.no_grad():
            for k, v in net.state_dict().items():
                if v.dtype.is_floating_point:
                    self.shadow[k].mul_(d).add_(v.detach(), alpha=1.0 - d)
                else:
                    self.shadow[k].copy_(v)

    def swap_in(self, net) -> dict:
        # Load the EMA weights into the net and return the raw weights to restore later.
        backup = {k: v.detach().clone() for k, v in net.state_dict().items()}
        net.load_state_dict(self.shadow)
        return backup


def make_ema(cfg, net):
    d = float(cfg["train"].get("ema", 0.0) or 0.0)
    return EMAWeights(net, d) if 0.0 < d < 1.0 else None


def select_monitor_rows(val_idx, site_coords, regimes, limit: int, seed: int) -> np.ndarray:
    # Pick at most limit validation rows for early stopping. Rows are grouped by
    # (site, regime) and one row per group is drawn first, so every site and sky
    # regime is represented before the remaining slots are filled at random.
    rows = np.asarray(val_idx, dtype=int)
    if limit <= 0 or len(rows) <= limit:
        return rows
    groups: dict[Any, list[int]] = {}
    for r in rows:
        site = None if site_coords is None else tuple(np.round(np.asarray(site_coords)[r], 6))
        reg = None if regimes is None else int(np.asarray(regimes)[r])
        groups.setdefault((site, reg), []).append(int(r))
    chooser = np.random.default_rng(int(seed))
    members = list(groups.values())
    if len(members) > limit:
        ids = chooser.choice(len(members), size=limit, replace=False)
        picked = [int(chooser.choice(members[int(i)])) for i in ids]
    else:
        picked = [int(chooser.choice(m)) for m in members]
        taken = set(picked)
        rest = np.array([r for r in rows if int(r) not in taken], dtype=int)
        if len(picked) < limit and len(rest):
            extra = chooser.choice(rest, size=min(limit - len(picked), len(rest)), replace=False)
            picked.extend(int(e) for e in extra)
    return np.array(sorted(set(picked)), dtype=int)


class GeneratorBase:
    def __init__(self, cfg: dict[str, Any], K: int, n_days: int, rep, device: str):
        self.cfg = cfg
        self.K, self.n_days = int(K), int(n_days)
        self.H = self.K * self.n_days
        self.rep = rep
        self.device = device
        self.net = None
        # Padded (night) steps carry the clear-sky code: finite and in-distribution.
        self.pad_fill = float(rep.clearsky_code())
        self.gcs_scale: float | None = None
        self.site_vec: np.ndarray | None = None
        self.regime_enabled = bool(cfg["model"].get("condition_regime", False))
        self.n_regimes = int(cfg["model"].get("n_regimes", 4)) if self.regime_enabled else 0
        self.regime_proportions: list[float] | None = None
        self.markov_sampler: MarkovRegimeSampler | None = None
        self.site_regime_model: SiteRegimeModel | None = None
        # Training record, written into the checkpoint.
        self.best_epoch: int | None = None
        self.best_val_score: float | None = None
        self.best_val_components: dict[str, float] = {}
        self.selection_metric = ""
        self.epochs_trained = 0
        self.fold_metadata: dict[str, Any] | None = None

    def _t(self, a):
        torch, _ = torch_nn()
        return torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=self.device)

    def _b(self, a):
        torch, _ = torch_nn()
        return torch.as_tensor(np.ascontiguousarray(a), dtype=torch.bool, device=self.device)

    def _fill(self, x, mask):
        return np.where(mask, np.nan_to_num(x, nan=self.pad_fill), self.pad_fill).astype(np.float32)

    @property
    def n_cond(self) -> int:
        # cos(zenith), mask, day fraction, five time channels, regime one-hots,
        # clear-sky GHI and the site vector (the last two only when frozen at fit time).
        n = 3 + 5 + self.n_regimes + (1 if self.gcs_scale is not None else 0)
        return n + (0 if self.site_vec is None else int(self.site_vec.shape[0]))

    def _freeze_conditioning(self, W, fit_rows) -> None:
        # Fix the clear-sky GHI scale and the site vector from the fitting rows only.
        self.gcs_scale = None
        if self.cfg["model"].get("condition_clearsky_ghi", True) and "fut_ghi_cs" in W:
            g = np.asarray(W["fut_ghi_cs"], np.float64)[fit_rows]
            v = g[np.asarray(W["fut_mask"], bool)[fit_rows]]
            v = v[np.isfinite(v)]
            if v.size:
                self.gcs_scale = max(1.0, float(np.percentile(v, 99)))
        self.site_vec = None
        if self.cfg["model"].get("condition_site", True) and "site_coords" in W:
            sc = np.asarray(W["site_coords"], np.float32)[fit_rows]
            if sc.ndim == 2 and sc.shape[1] == 4 and np.isfinite(sc).all():
                # Mean position is only the fallback for generation without coordinates.
                self.site_vec = sc.mean(axis=0).astype(np.float32)

    def conditioning(self, fut_zen, fut_mask, fut_ghi_cs=None, date_ord=None,
                     site_coords=None, regime=None) -> np.ndarray:
        # Build the [N, C, H] conditioning tensor. Every channel is known before the
        # day happens (geometry, calendar, site, optional regime label).
        zen = np.asarray(fut_zen)
        n, h = zen.shape
        if h != self.H:
            raise ValueError(f"horizon {h} does not match the model horizon {self.H}")
        ch = [np.cos(np.deg2rad(np.nan_to_num(zen, nan=90.0))).astype(np.float32),
              np.asarray(fut_mask).astype(np.float32),
              np.broadcast_to(np.repeat((np.arange(self.n_days) + 0.5) / self.n_days, self.K), (n, h))]
        if date_ord is None:
            raise ValueError("date_ord is required for the calendar channels")
        ords = np.asarray(date_ord).reshape(-1)
        if len(ords) != n:
            raise ValueError(f"date_ord has {len(ords)} entries for {n} rows")
        ang = 2.0 * np.pi * (ords.astype(np.int64) % 365).astype(np.float32) / 365.0
        lead = np.linspace(0.0, 1.0, h, dtype=np.float32)
        solar = 2.0 * np.pi * (lead - 0.5)
        ch += [np.repeat(np.sin(ang)[:, None], h, axis=1), np.repeat(np.cos(ang)[:, None], h, axis=1),
               np.broadcast_to(np.sin(solar), (n, h)), np.broadcast_to(np.cos(solar), (n, h)),
               np.broadcast_to(lead, (n, h))]
        if self.regime_enabled:
            if regime is None:
                raise ValueError("a regime-conditioned model needs regime labels")
            rid = np.asarray(regime, np.int64).reshape(-1)
            if len(rid) != n or np.any((rid < 0) | (rid >= self.n_regimes)):
                raise ValueError(f"regime must hold {n} labels in [0, {self.n_regimes - 1}]")
            onehot = np.eye(self.n_regimes, dtype=np.float32)[rid]
            ch += [np.repeat(onehot[:, j:j + 1], h, axis=1) for j in range(self.n_regimes)]
        if self.gcs_scale is not None:
            if fut_ghi_cs is None:
                raise ValueError("this model was trained with clear-sky GHI; fut_ghi_cs is required")
            ch.append(np.nan_to_num(np.asarray(fut_ghi_cs, np.float32) / self.gcs_scale, nan=0.0))
        if self.site_vec is not None:
            if site_coords is None:
                warnings.warn("site_coords not supplied; using the mean training site", RuntimeWarning)
                sc = np.tile(self.site_vec, (n, 1))
            else:
                sc = np.asarray(site_coords, np.float32)
                if sc.shape != (n, len(self.site_vec)):
                    raise ValueError(f"site_coords shape {sc.shape} != {(n, len(self.site_vec))}")
            ch += [np.repeat(sc[:, j:j + 1], h, axis=1) for j in range(sc.shape[1])]
        return np.stack(ch, axis=1).astype(np.float32)

    def _fit_regime_samplers(self, labels, fit_rows, site_coords, date_ord) -> None:
        # Learn proportions and transitions from the fitting rows only.
        rc = self.cfg["model"].get("regime", {})
        fit_labels = np.asarray(labels, np.int64)[fit_rows]
        self.regime_proportions = regime_proportions(fit_labels, self.n_regimes).tolist()
        if site_coords is not None and rc.get("site_conditioned", True):
            self.site_regime_model = SiteRegimeModel.fit(
                fit_labels, np.asarray(site_coords)[fit_rows], self.n_regimes,
                alpha_proportions=float(rc.get("site_alpha_proportions", 12.0)),
                alpha_transition=float(rc.get("site_alpha_transition", 12.0)),
                coordinate_scale=rc.get("coordinate_scale"),
                date_ord=None if date_ord is None else np.asarray(date_ord)[fit_rows])
            self.regime_proportions = self.site_regime_model.global_proportions.tolist()
        if rc.get("markov_enabled", True):
            self.markov_sampler = MarkovRegimeSampler.from_labels(
                fit_labels, self.n_regimes, smoothing=float(rc.get("markov_smoothing", 1.0)))

    def sample_regimes(self, n_rows, rng, site_coords=None, date_ord=None) -> np.ndarray:
        # Independent regime draws for rows whose true regime is unknown
        # (validation rows during training, every row at generation time).
        if self.site_regime_model is not None and site_coords is not None:
            return self.site_regime_model.sample_independent(site_coords, rng, day_ord=date_ord)
        p = self.regime_proportions or np.full(self.n_regimes, 1.0 / self.n_regimes)
        return allocate_regimes(n_rows, p, rng)

    def _integrate(self, x, cond, ok, steps: int):
        torch, _ = torch_nn()
        sampler = str(self.cfg["train"].get("sampler", "heun")).lower()
        if sampler not in ("euler", "midpoint", "heun"):
            raise ValueError(f"unknown sampler {sampler!r}")
        dt = 1.0 / steps
        b = x.shape[0]

        def vel(state, ts):
            t = torch.full((b,), ts, device=self.device, dtype=state.dtype)
            return self.net(state, t, cond, ok)[:, 0]      # single output channel: [B, H]

        for s in range(steps):
            t0 = s * dt
            if sampler == "euler":
                x = x + dt * vel(x, t0)
            elif sampler == "midpoint":
                x = x + dt * vel(x + 0.5 * dt * vel(x, t0), t0 + 0.5 * dt)
            else:
                k1 = vel(x, t0)
                x = x + 0.5 * dt * (k1 + vel(x + dt * k1, t0 + dt))
        return x

    def _build_net(self, n_out: int = 1):
        self.net = build_network(self.cfg, self.H, self.n_cond, n_out=n_out).to(self.device)
        return self.net
