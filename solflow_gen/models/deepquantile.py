# DeepQuantile: the baseline the Flow is compared against. It uses the same
# backbone and the same conditioning as the Flow but predicts M marginal quantiles
# per step with a pinball loss. It has no model of dependence between steps, so
# whole-day scenarios need an explicit coupling: independent uniforms, or the
# Gaussian temporal copula fitted on validation data (see evaluation/copula_fit.py).
from __future__ import annotations

from typing import Any

import numpy as np

from ..lazy import torch_nn
from .base import GeneratorBase, make_ema, select_monitor_rows


def quantile_curves_to_scenarios(curves, q_levels, n_scenarios, rng=None, copula=None):
    # Turn quantile curves [N, M, H] into scenarios [N, S, H] by inverting each step's
    # piecewise-linear CDF at uniforms. copula=None gives independent uniforms across
    # steps (marginals right, no temporal structure); a fitted copula supplies
    # correlated uniforms instead.
    curves = np.asarray(curves, dtype=float)
    levels = np.asarray(q_levels, dtype=float)
    n_scenarios = int(n_scenarios)
    if curves.ndim != 3 or levels.ndim != 1 or curves.shape[1] != len(levels):
        raise ValueError("curves must be [rows, quantiles, horizon] and match q_levels")
    if len(levels) < 1 or not np.isfinite(levels).all() or np.any(np.diff(levels) <= 0):
        raise ValueError("q_levels must be finite and strictly increasing")
    if n_scenarios < 1:
        raise ValueError("n_scenarios must be positive")
    rng = rng or np.random.default_rng()
    n_rows, n_q, horizon = curves.shape
    if copula is None:
        uniforms = rng.random((n_rows, n_scenarios, horizon))
    else:
        if int(copula.horizon) != horizon:
            raise ValueError(f"copula horizon {copula.horizon} != curve horizon {horizon}")
        uniforms = copula.uniforms(n_rows, n_scenarios, rng)
    if n_q == 1:
        return np.broadcast_to(curves[:, :1, :], (n_rows, n_scenarios, horizon)).astype(np.float32, copy=True)
    out = np.empty((n_rows, n_scenarios, horizon), dtype=np.float32)
    for step in range(horizon):
        u = np.clip(uniforms[:, :, step], levels[0], levels[-1])
        upper = np.clip(np.searchsorted(levels, u, side="right"), 1, n_q - 1)
        lower = upper - 1
        q = curves[:, :, step]
        lo, hi = np.take_along_axis(q, lower, axis=1), np.take_along_axis(q, upper, axis=1)
        out[:, :, step] = lo + (u - levels[lower]) / (levels[upper] - levels[lower]) * (hi - lo)
    return out


class DeepQuantile(GeneratorBase):
    family = "deepquantile"

    def __init__(self, cfg: dict[str, Any], K: int, n_days: int, rep, device: str, n_quantiles: int | None = None):
        super().__init__(cfg, K, n_days, rep, device)
        # The baseline is never regime-conditioned: it is the plain reference.
        self.regime_enabled, self.n_regimes = False, 0
        m = int(n_quantiles or cfg["baseline"]["n_quantiles"])
        self.q_levels = (np.arange(m, dtype=np.float32) + 0.5) / m

    def fit(self, data: dict[str, np.ndarray], train_idx, val_idx=None, rng=None):
        torch, _ = torch_nn()
        tc = self.cfg["train"]
        rng = rng or np.random.default_rng(self.cfg["seed"])
        tr = np.asarray(train_idx, dtype=int)
        va = np.asarray(val_idx if val_idx is not None else [], dtype=int)
        do_es = bool(tc.get("early_stopping", True)) and len(va) > 0
        fm = np.asarray(data["fut_mask"], bool)
        dates = np.asarray(data["date_ord"]).reshape(-1)
        sc = data.get("site_coords")
        m = len(self.q_levels)

        self._freeze_conditioning(data, tr)
        self._build_net(m)
        opt = torch.optim.AdamW(self.net.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
        y = self._fill(self.rep.encode(data["fut_csi"]), fm)
        cond = self.conditioning(data["fut_zen"], fm, data.get("fut_ghi_cs"), dates, sc, None)
        q_t = self._t(self.q_levels)[None, :, None]

        def pinball(rows):
            b = len(rows)
            x0 = torch.zeros(b, self.H, device=self.device)
            t = torch.full((b,), 0.5, device=self.device)
            ok = self._b(fm[rows])
            q = self.net(x0, t, self._t(cond[rows]), ok)
            e = self._t(y[rows])[:, None, :] - q
            pin = torch.maximum(q_t * e, (q_t - 1.0) * e)
            w = ok.float()[:, None, :]
            return (pin * w).sum() / (w.sum() * pin.shape[1]).clamp_min(1.0)

        monitor = (select_monitor_rows(va, sc, None, int(tc.get("es_max_rows", 512)),
                                       int(tc.get("es_seed", self.cfg["seed"]))) if do_es else None)
        self.selection_metric = "validation_pinball" if do_es else "none"
        warmup = max(0, int(tc.get("warmup_epochs", 0)))
        patience = int(tc.get("patience", 15))
        min_delta = float(tc.get("min_delta", 1e-4))
        ema = make_ema(self.cfg, self.net)
        best, bad, best_state = np.inf, 0, None
        self.best_epoch, self.best_val_score, self.best_val_components = None, None, {}
        bs = int(tc["batch_size"])
        for ep in range(int(tc["epochs"])):
            self.epochs_trained = ep + 1
            self.net.train()
            running = torch.zeros((), device=self.device)
            n_batches = 0
            perm = rng.permutation(tr)
            for s0 in range(0, len(perm), bs):
                loss = pinball(perm[s0:s0 + bs])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                if ema is not None:
                    ema.update(self.net)
                running += loss.detach()
                n_batches += 1
            if tc.get("verbose", 1) >= 1:
                print(f"  [deepquantile] epoch {ep + 1}/{tc['epochs']} "
                      f"pinball={float(running) / max(n_batches, 1):.5f}", flush=True)
            if not (do_es and ep + 1 > warmup and (ep + 1 - warmup) % int(tc.get("es_every", 2)) == 0):
                continue
            raw = ema.swap_in(self.net) if ema is not None else None
            self.net.eval()
            with torch.no_grad():
                score = float(pinball(monitor))
            improved = score < best - min_delta
            if improved:
                best, bad = score, 0
                self.best_epoch, self.best_val_score = ep + 1, score
                self.best_val_components = {"validation_pinball": score}
                best_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
            if tc.get("verbose", 1) >= 1:
                print(f"  [deepquantile] validation epoch={ep + 1} pinball={score:.6f} "
                      f"best={best:.6f} improved={improved} patience={bad}/{patience}", flush=True)
            if raw is not None:
                self.net.load_state_dict(raw)
            if bad >= patience:
                if tc.get("verbose", 1) >= 1:
                    print(f"  [deepquantile] early stop at epoch {ep + 1}")
                break
        if best_state is not None and tc.get("restore_best", True):
            self.net.load_state_dict(best_state)
        elif ema is not None:
            ema.swap_in(self.net)
        self.net.eval()
        return self

    def predict_quantiles(self, cond: dict[str, np.ndarray]) -> np.ndarray:
        # Quantile curves [N, M, H] in CSI space, sorted so they are monotone in level.
        torch, _ = torch_nn()
        zen = np.asarray(cond["fut_zen"])
        fm = np.asarray(cond["fut_mask"], bool)
        c = self.conditioning(zen, fm, cond.get("fut_ghi_cs"), cond["date_ord"],
                              cond.get("site_coords"), None)
        n, m = len(zen), len(self.q_levels)
        out = np.empty((n, m, self.H), np.float32)
        chunk = max(1, int(self.cfg["train"].get("sample_chunk", 4096)) // m)
        self.net.eval()
        with torch.no_grad():
            for c0 in range(0, n, chunk):
                sl = slice(c0, min(n, c0 + chunk))
                k = sl.stop - sl.start
                q = self.net(torch.zeros(k, self.H, device=self.device),
                             torch.full((k,), 0.5, device=self.device),
                             self._t(c[sl]), self._b(fm[sl]))
                q, _ = torch.sort(q, dim=1)
                out[sl] = self.rep.decode(q.cpu().numpy())
        return out

    def generate(self, cond, n_ensemble: int = 100, rng=None, copula=None) -> np.ndarray:
        # Scenarios [N, n_ensemble, H]; independent across steps unless a copula is given.
        rng = rng or np.random.default_rng()
        return quantile_curves_to_scenarios(self.predict_quantiles(cond), self.q_levels,
                                            n_ensemble, rng=rng, copula=copula)
