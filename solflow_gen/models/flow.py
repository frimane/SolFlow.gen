# GeometryFlow: a conditional flow-matching generator of whole-day clear-sky-index
# trajectories. It sees only sun geometry, the calendar, clear-sky irradiance, the
# site and (optionally) a sky-regime label; it never sees measured history.
#
# Training draws a source trajectory x0 from the seasonal prior, picks a random
# flow time t, forms x_t = (1 - t) x0 + t x1 on the straight line to the observed
# day x1, and regresses the net onto the velocity x1 - x0 (night steps masked).
# Generation integrates the learned ODE from a fresh prior draw.
from __future__ import annotations

from typing import Any

import numpy as np

from ..evaluation.scores import joint_validation_score, window_joint_scores
from ..lazy import torch_nn
from .base import GeneratorBase, make_ema, select_monitor_rows
from .regimes import classify_regimes


class GeometryFlow(GeneratorBase):
    family = "flow"

    def __init__(self, cfg: dict[str, Any], K: int, n_days: int, rep, prior, device: str):
        super().__init__(cfg, K, n_days, rep, device)
        self.prior = prior

    def fit(self, data: dict[str, np.ndarray], train_idx, val_idx=None, regime=None, rng=None):
        # data holds the rows needed for fitting (train and validation rows);
        # train_idx / val_idx index into it. regime gives one label per row, computed
        # from observed CSI; only the training labels are used as truth, validation rows
        # get labels sampled from the training statistics (no peeking at their targets).
        torch, _ = torch_nn()
        tc = self.cfg["train"]
        rng = rng or np.random.default_rng(self.cfg["seed"])
        tr = np.asarray(train_idx, dtype=int)
        va = np.asarray(val_idx if val_idx is not None else [], dtype=int)
        do_es = bool(tc.get("early_stopping", True)) and len(va) > 0
        fm = np.asarray(data["fut_mask"], bool)
        dates = np.asarray(data["date_ord"]).reshape(-1)
        sc = data.get("site_coords")

        self._freeze_conditioning(data, tr)
        labels = None
        if self.regime_enabled:
            if regime is None:
                regime, _ = classify_regimes(data["fut_csi"], fm, self.cfg["model"].get("regime"))
            labels = np.asarray(regime, np.int64).copy()
            self._fit_regime_samplers(labels, tr, sc, dates)
            if len(va):
                labels[va] = self.sample_regimes(
                    len(va), np.random.default_rng(int(self.cfg.get("seed", 0)) + 44017),
                    site_coords=None if sc is None else np.asarray(sc)[va], date_ord=dates[va])

        self._build_net(1)
        opt = torch.optim.AdamW(self.net.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
        x1 = self._fill(self.rep.encode(data["fut_csi"]), fm)
        cond = self.conditioning(data["fut_zen"], fm, data.get("fut_ghi_cs"), dates, sc, labels)

        # Keep the static tensors on the device so each batch only ships the prior draw.
        try:
            X1, CND, OK = self._t(x1), self._t(cond), self._b(fm)
            FW = OK.float()
            cached = True
        except RuntimeError as exc:
            import warnings
            warnings.warn(f"could not cache training tensors on the device ({exc}); "
                          "falling back to per-batch transfers", RuntimeWarning)
            cached = False
        use_amp = str(tc.get("amp", "off")).lower() == "bf16" and str(self.device).startswith("cuda")

        monitor = (select_monitor_rows(va, sc, labels, int(tc.get("es_max_rows", 512)),
                                       int(tc.get("es_seed", self.cfg["seed"]))) if do_es else None)
        self.selection_metric = "joint_energy_variogram" if do_es else "none"
        es_steps = int(tc.get("es_sampling_steps") or tc["n_sampling_steps"])
        warmup = max(0, int(tc.get("warmup_epochs", 0)))
        patience = int(tc.get("patience", 15))
        smooth = min(max(float(tc.get("es_smoothing", 0.0)), 0.0), 1.0)
        min_delta = float(tc.get("min_delta", 1e-4))

        ema = make_ema(self.cfg, self.net)
        best, bad, score_ema, best_state = np.inf, 0, None, None
        self.best_epoch, self.best_val_score, self.best_val_components = None, None, {}
        bs = int(tc["batch_size"])
        for ep in range(int(tc["epochs"])):
            self.epochs_trained = ep + 1
            self.net.train()
            running = torch.zeros((), device=self.device)
            n_batches = 0
            perm = rng.permutation(tr)
            for s0 in range(0, len(perm), bs):
                b = perm[s0:s0 + bs]
                x0 = self._t(self.prior.sample(len(b), rng, fm[b], dates[b]))
                tt = self._t(rng.random(len(b)))
                if cached:
                    bt = torch.as_tensor(b, device=self.device)
                    x1b, cb, okb, wb = X1[bt], CND[bt], OK[bt], FW[bt]
                else:
                    x1b, cb, okb = self._t(x1[b]), self._t(cond[b]), self._b(fm[b])
                    wb = okb.float()
                xt = (1 - tt)[:, None] * x0 + tt[:, None] * x1b
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                    v = self.net(xt, tt, cb, okb)[:, 0]
                    loss = (((v.float() - (x1b - x0)) ** 2) * wb).sum() / wb.sum().clamp_min(1.0)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                if ema is not None:
                    ema.update(self.net)
                running += loss.detach()
                n_batches += 1
            if tc.get("verbose", 1) >= 1:
                print(f"  epoch {ep + 1}/{tc['epochs']} loss={float(running) / max(n_batches, 1):.5f}",
                      flush=True)

            if not (do_es and ep + 1 > warmup and (ep + 1 - warmup) % int(tc.get("es_every", 2)) == 0):
                continue
            raw = ema.swap_in(self.net) if ema is not None else None
            pred = self._sample_cached(cond[monitor], fm[monitor], dates[monitor],
                                       int(tc.get("es_members", 100)), es_steps,
                                       np.random.default_rng(int(tc.get("es_seed", self.cfg["seed"]))))
            target = np.asarray(data["fut_csi"])[monitor]
            valid = fm[monitor]
            vals = target[valid]
            vals = vals[np.isfinite(vals)]
            if not vals.size:
                raise ValueError("early stopping has no valid validation targets")
            es_rows, vs_rows = window_joint_scores(pred, target, valid)
            es_rows, vs_rows = es_rows[np.isfinite(es_rows)], vs_rows[np.isfinite(vs_rows)]
            if not len(es_rows) or not len(vs_rows):
                raise ValueError("early stopping produced no finite energy / variogram scores")
            scale = max(float(np.mean(np.abs(vals))), 1e-6)
            es, vs = float(es_rows.mean()) / scale, float(vs_rows.mean()) / scale
            score = joint_validation_score(es, vs)
            score_ema = score if score_ema is None else smooth * score + (1 - smooth) * score_ema
            improved = score_ema < best - min_delta
            if improved:
                best, bad = score_ema, 0
                self.best_epoch, self.best_val_score = ep + 1, float(score_ema)
                self.best_val_components = {"fair_energy_score_normalized": es,
                                            "variogram_score_normalized": vs}
                best_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
            if tc.get("verbose", 1) >= 1:
                print(f"  validation epoch={ep + 1} score={score:.6f} ema={score_ema:.6f} "
                      f"best={best:.6f} improved={improved} patience={bad}/{patience} "
                      f"energy={es:.6f} variogram={vs:.6f}", flush=True)
            if raw is not None:
                self.net.load_state_dict(raw)
            if bad >= patience:
                if tc.get("verbose", 1) >= 1:
                    print(f"  early stop at epoch {ep + 1} (smoothed score {score_ema:.5f})")
                break

        # Deploy the best validated weights; without validation, the EMA (or raw) weights.
        if best_state is not None and tc.get("restore_best", True):
            self.net.load_state_dict(best_state)
        elif ema is not None:
            ema.swap_in(self.net)
        self.net.eval()
        return self

    def _sample_cached(self, cond, fm, dates, M, steps, rng):
        # Sample M members for each row of an already-built conditioning tensor.
        torch, _ = torch_nn()
        n = len(cond)
        out = np.empty((n, M, self.H), np.float32)
        chunk = max(1, int(self.cfg["train"].get("sample_chunk", 4096)) // M)
        self.net.eval()
        with torch.no_grad():
            for c0 in range(0, n, chunk):
                sl = slice(c0, min(n, c0 + chunk))
                k = sl.stop - sl.start
                x0 = self.prior.sample(k * M, rng, np.repeat(fm[sl], M, axis=0),
                                       np.repeat(dates[sl], M))
                x = self._integrate(self._t(x0), self._t(cond[sl]).repeat_interleave(M, dim=0),
                                    self._b(fm[sl]).repeat_interleave(M, dim=0), steps)
                out[sl] = self.rep.decode(x.cpu().numpy()).reshape(k, M, self.H)
        return out

    def generate(self, cond: dict[str, np.ndarray], n_ensemble: int = 50, rng=None,
                 regime=None, steps: int | None = None) -> np.ndarray:
        # cond: fut_zen, fut_mask, date_ord (+ fut_ghi_cs and site_coords when the model
        # was trained with them). Returns CSI scenarios [N, n_ensemble, H].
        torch, _ = torch_nn()
        rng = rng or np.random.default_rng()
        M = int(n_ensemble)
        if M < 1:
            raise ValueError("n_ensemble must be positive")
        zen = np.asarray(cond["fut_zen"])
        fm = np.asarray(cond["fut_mask"], bool)
        dates = np.asarray(cond["date_ord"]).reshape(-1)
        gcs, sc = cond.get("fut_ghi_cs"), cond.get("site_coords")
        n = len(zen)
        steps = int(steps or self.cfg["train"]["n_sampling_steps"])

        labels = None
        if self.regime_enabled:
            if regime is None:
                labels = self.sample_regimes(
                    n * M, rng, site_coords=None if sc is None else np.repeat(np.asarray(sc), M, axis=0),
                    date_ord=np.repeat(dates, M))
            else:
                given = np.asarray(regime, np.int64).reshape(-1)
                if len(given) == n:
                    labels = np.repeat(given, M)
                elif len(given) == n * M:
                    labels = given
                else:
                    raise ValueError(f"regime must hold {n} or {n * M} labels")

        pick = lambda a, ix: None if a is None else np.asarray(a)[ix]
        out = np.empty((n, M, self.H), np.float32)
        chunk = max(1, int(self.cfg["train"].get("sample_chunk", 4096)) // M)
        self.net.eval()
        with torch.no_grad():
            for c0 in range(0, n, chunk):
                sl = slice(c0, min(n, c0 + chunk))
                k = sl.stop - sl.start
                if self.regime_enabled:
                    rows = np.repeat(np.arange(sl.start, sl.stop), M)
                    c = self.conditioning(zen[rows], fm[rows], pick(gcs, rows), dates[rows],
                                          pick(sc, rows), labels[sl.start * M:sl.stop * M])
                    cond_t, ok_t = self._t(c), self._b(fm[rows])
                else:
                    c = self.conditioning(zen[sl], fm[sl], pick(gcs, sl), dates[sl], pick(sc, sl), None)
                    cond_t = self._t(c).repeat_interleave(M, dim=0)
                    ok_t = self._b(fm[sl]).repeat_interleave(M, dim=0)
                x0 = self.prior.sample(k * M, rng, np.repeat(fm[sl], M, axis=0), np.repeat(dates[sl], M))
                x = self._integrate(self._t(x0), cond_t, ok_t, steps)
                out[sl] = self.rep.decode(x.cpu().numpy()).reshape(k, M, self.H)
        return out

    def generate_sequence(self, cond, n_ensemble: int = 50, rng=None, initial_state=None):
        # Chronologically ordered rows get a connected regime sequence (site/season
        # Markov chain), then one ensemble per day. Returns (scenarios, labels).
        if not self.regime_enabled:
            raise ValueError("sequence generation needs a regime-conditioned model (model.condition_regime)")
        rng = rng or np.random.default_rng()
        sc = cond.get("site_coords")
        dates = np.asarray(cond["date_ord"]).reshape(-1)
        n = len(dates)
        if self.site_regime_model is not None and sc is not None:
            labels = self.site_regime_model.sample_sequence(sc, rng, initial_state=initial_state, day_ord=dates)
        else:
            sampler = self.markov_sampler
            if sampler is None:
                from .regimes import MarkovRegimeSampler
                p = np.asarray(self.regime_proportions or np.full(self.n_regimes, 1.0 / self.n_regimes))
                sampler = MarkovRegimeSampler(np.tile(p, (self.n_regimes, 1)), p)
            labels = sampler.sample(n, rng, initial_state=initial_state)
        return self.generate(cond, n_ensemble, rng, regime=labels), labels
