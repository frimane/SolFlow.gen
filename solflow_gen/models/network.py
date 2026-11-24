# Backbone shared by the Flow (velocity field) and the DeepQuantile baseline.
#
# A dilated, non-causal temporal convolutional network over the day:
#   - input: the state x [B, H] plus conditioning channels [B, C, H]
#     (geometry, mask, time, site, ...)
#   - residual blocks of conv, GroupNorm, ReLU, FiLM, conv, GroupNorm with a skip
#     connection; FiLM is driven by a sinusoidal embedding of the flow time t
#   - one self-attention layer with a feed-forward net in the middle of the stack;
#     night steps are never used as keys
#   - output: n_out channels per step (1 for the Flow velocity, M for the quantiles)
#
# Both models use this exact network, so differences between them come from the
# generative approach and the loss, not from the architecture.
from __future__ import annotations

import math
from typing import Any

from ..lazy import torch_nn


def plan_dilations(mcfg: dict[str, Any], H: int) -> tuple[list[int], int, bool]:
    # Dilations 1, 2, 4, ... and the receptive field they give.
    #
    # n_blocks="auto" picks the smallest count whose receptive field covers
    # the horizon (up to max_blocks); an integer is used as given.
    # Returns (dilations, receptive_field, covers_horizon).
    ks = int(mcfg.get("kernel_size", 3))
    if ks % 2 != 1:
        raise ValueError("model.kernel_size must be odd so padding preserves length")
    max_d, max_b = int(mcfg.get("max_dilation", 64)), int(mcfg.get("max_blocks", 8))
    n_cfg = mcfg.get("n_blocks", "auto")
    rf_of = lambda dils: 1 + sum(2 * (ks - 1) * d for d in dils)
    if n_cfg == "auto":
        dils: list[int] = []
        while rf_of(dils) < H and len(dils) < max_b:
            dils.append(min(2 ** len(dils), max_d))
    else:
        dils = [min(2 ** i, max_d) for i in range(int(n_cfg))]
    rf = rf_of(dils)
    return dils, rf, rf >= H


def build_network(cfg: dict[str, Any], H: int, n_cond: int, n_out: int = 1):
    # Construct the backbone. n_cond is the number of conditioning channels.
    torch, nn = torch_nn()
    m = cfg["model"]
    hidden, temb, cemb = int(m["hidden"]), int(m["time_embed_dim"]), int(m["cond_embed_dim"])
    drop, ks = float(m["dropout"]), int(m.get("kernel_size", 3))
    heads, use_attn = int(m.get("n_heads", 4)), bool(m.get("attention", True))
    dils, rf, covered = plan_dilations(m, H)
    if not covered and not use_attn:
        raise ValueError(f"receptive field {rf} < horizon {H} and attention is off; raise "
                         "model.max_blocks / model.max_dilation or enable model.attention")
    if use_attn and hidden % heads:
        raise ValueError(f"model.hidden={hidden} must be divisible by model.n_heads={heads}")

    def sinusoidal(t, dim):
        half = dim // 2
        freqs = torch.exp(torch.linspace(0.0, math.log(10000.0), half, device=t.device))
        ang = t[:, None] * freqs[None, :]
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)

    class FiLM(nn.Module):
        def __init__(self, ch, cond):
            super().__init__()
            self.g, self.b = nn.Linear(cond, ch), nn.Linear(cond, ch)

        def forward(self, x, c):
            return x * (1 + self.g(c)[:, :, None]) + self.b(c)[:, :, None]

    class Block(nn.Module):
        def __init__(self, ch, cond, d):
            super().__init__()
            pad = d * (ks - 1) // 2
            self.c1 = nn.Conv1d(ch, ch, ks, padding=pad, dilation=d)
            self.c2 = nn.Conv1d(ch, ch, ks, padding=pad, dilation=d)
            self.n1, self.n2 = nn.GroupNorm(1, ch), nn.GroupNorm(1, ch)
            self.film = FiLM(ch, cond)
            self.drop = nn.Dropout(drop)

        def forward(self, x, c):
            y = torch.relu(self.n1(self.c1(x)))
            y = self.drop(self.film(y, c))
            return torch.relu(x + self.n2(self.c2(y)))

    class HorizonAttention(nn.Module):
        # Self-attention along the horizon; padded steps are never used as keys.

        def __init__(self, ch):
            super().__init__()
            self.pos = nn.Parameter(torch.randn(1, H, ch) * 0.02)
            self.ln_a, self.ln_f = nn.LayerNorm(ch), nn.LayerNorm(ch)
            self.attn = nn.MultiheadAttention(ch, heads, dropout=drop, batch_first=True)
            self.ffn = nn.Sequential(nn.Linear(ch, 4 * ch), nn.GELU(), nn.Linear(4 * ch, ch))

        def forward(self, h, ok):
            # h [B, C, H]; ok [B, H] bool, True = valid. A row without any valid
            # step would make the softmax divide by zero (NaN), so such rows get
            # one dummy valid key; their output is then simply uninformative.
            y = h.transpose(1, 2) + self.pos
            pad = ~ok
            none_valid = ok.sum(dim=1) == 0
            pad = pad.clone()
            pad[:, 0] = torch.where(none_valid, torch.zeros_like(pad[:, 0]), pad[:, 0])
            z = self.ln_a(y)
            y = y + self.attn(z, z, z, key_padding_mask=pad, need_weights=False)[0]
            y = y + self.ffn(self.ln_f(y))
            return y.transpose(1, 2)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_proj = nn.Conv1d(1 + n_cond, hidden, 1)
            self.time_mlp = nn.Sequential(nn.Linear(temb, cemb), nn.SiLU(), nn.Linear(cemb, cemb))
            self.blocks = nn.ModuleList([Block(hidden, cemb, d) for d in dils])
            self.attn = HorizonAttention(hidden) if use_attn else None
            self.attn_after = max(0, len(dils) // 2 - 1)     # insert midway through the stack
            self.out = nn.Conv1d(hidden, n_out, 1)
            self.receptive_field, self._temb = rf, temb

        def forward(self, x, t, cond, ok):
            # x [B, H], t [B], cond [B, n_cond, H], ok [B, H] -> [B, n_out, H].
            h = self.in_proj(torch.cat([x.unsqueeze(1), cond], dim=1))
            c = self.time_mlp(sinusoidal(t, self._temb))
            for i, blk in enumerate(self.blocks):
                h = blk(h, c)
                if self.attn is not None and i == self.attn_after:
                    h = self.attn(h, ok)
            return self.out(h)

    return Net()
