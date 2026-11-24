# Configuration: one nested dictionary controls every behaviour.
#
# DEFAULT_CONFIG holds the defaults. A user JSON/YAML file is deep-merged over
# it, so a config file only has to list the keys it changes.
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

SPLIT_SCHEMES = ("holdout", "forward", "blocked_cv", "loso", "strict_spatiotemporal_holdout")
REPRESENTATIONS = ("gauss", "log")

DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 1990,

    "paths": {
        "raw_glob": None,                    # single-site CSV glob (CLI --input)
        "windows": "data/windows.npz",       # preprocessing output
        "models_dir": "runs/models",         # training output
    },

    # Coordinates are required whenever model.condition_site is true.
    "site": {
        "name": "site0",
        "latitude": None,                    # degrees north
        "longitude": None,                   # degrees east (west negative)
        "altitude": None,                    # metres
        "altitude_scale_m": 2000.0,          # altitude feature = altitude / scale
    },

    # Each entry: {"name", "latitude", "longitude", "altitude", "raw_glob"}.
    # When non-empty (and --multi-site is used) it replaces site.
    "sites": [],

    "data": {
        # source column names in the CSV files
        "col_datetime": "datetime",
        "col_ghi": "ghi_measured",
        "col_ghi_cs": "ghi_clearsky",
        "col_zenith": "zenith",

        # cadence of the model grid, and of the raw files
        "resolution_min": 10,
        "source_resolution_min": 1,
        "resample": True,                    # average raw rows onto the model grid
        "resample_min_frac": 0.6,            # keep a bin if >= this share of raw rows exist

        # physical limits of the clear-sky index
        "clearsky_floor": 50.0,              # W/m^2; CSI undefined below this clear-sky GHI
        "zenith_daylight_max": 85.0,         # degrees; daylight window
        "csi_min": 0.02,
        "csi_max": 1.80,
        "center": "clearsky",                # "clearsky": CSI = 1 maps to 0 in model space; or "mean"
        "drop_suspect_zero": True,           # zero CSI in daylight = sensor dropout
        "min_daylight_steps": 12,            # drop days with fewer valid daylight steps
    },

    # A training example is forecast_days consecutive calendar days.
    "task": {"forecast_days": 1},

    "split": {
        "scheme": "holdout",                 # see SPLIT_SCHEMES
        "train_mode": "folds",               # "folds" (evaluation) or "all" (deployment fit)
        "n_folds": 4,                        # blocked_cv / forward
        "holdout": {"train_end": None, "train_frac": 0.8},
        # strict_spatiotemporal_holdout: train on other stations in train_years,
        # test on held-out station groups in test_years.
        "spatial_test_stations": 10,
        "spatial_repeats": 4,
        "spatial_seed": 1990,
        "train_years": None,                 # default: all years but the last
        "test_years": None,                  # default: the last year
        "val_frac": 0.1,                     # chronological tail of train used for early stopping
    },

    "representation": {
        "kind": "log",                       # "log" or "gauss" (quantile-mapped to a standard normal)
        "gauss_knots": 1001,
        "gauss_p_clip": 1e-6,
    },

    # Seasonal clear-sky prior: x0 = seasonal-mean profile + correlated noise.
    "prior": {
        "kernel": "matern32",                # matern12 | matern32 | rbf
        "jitter": 1.0e-5,
        "col_std_floor_frac": 0.10,
        "enhancement": {                     # optional one-sided cloud-enhancement burst
            "weight": 0.0,                   # probability of a burst per sample; 0 disables
            "length_scale": 1.0,
            "amplitude_frac": 0.5,
        },
    },

    "model": {
        "hidden": 128,
        "n_blocks": "auto",                  # "auto": smallest receptive field covering the horizon
        "max_blocks": 8,
        "max_dilation": 64,
        "kernel_size": 3,
        "time_embed_dim": 64,
        "cond_embed_dim": 128,
        "dropout": 0.1,
        "attention": True,                   # one self-attention layer along the horizon
        "n_heads": 4,

        # conditioning channels (all forecast-time-known; none observed)
        "condition_clearsky_ghi": True,      # clear-sky GHI profile
        "condition_time_features": True,     # annual phase, solar-time phase, lead position
        "condition_site": True,              # ECEF unit-sphere coordinates + scaled altitude
        "condition_regime": False,           # four-state sky-regime variable (Flow only)
        "n_regimes": 4,
        "regime": {
            "clear_mean_min": 0.90,
            "clear_std_max": 0.08,
            "overcast_mean_max": 0.45,
            "high_variability_std_min": 0.22,
            "clear_csi_min": 0.90,
            "markov_enabled": True,
            "markov_smoothing": 1.0,
            "site_conditioned": True,
            "site_alpha_proportions": 12.0,
            "site_alpha_transition": 12.0,
            "coordinate_scale": None,
        },
    },

    "train": {
        "epochs": 250,
        "batch_size": 256,
        "lr": 1.0e-3,
        "weight_decay": 1.0e-5,
        "ema": 0.999,                        # EMA of weights (0 disables)
        "amp": "off",                        # "off" | "bf16" (CUDA only)
        "device": "auto",                    # "auto" | "cpu" | "cuda"
        "verbose": 1,
        # sampling (Flow)
        "n_sampling_steps": 10,
        "sampler": "heun",                   # euler | midpoint | heun
        "sample_chunk": 4096,                # cap on rows*members per sampling chunk
        # early stopping
        "early_stopping": True,
        "warmup_epochs": 5,
        "patience": 15,
        "min_delta": 1.0e-4,
        "es_every": 2,
        "es_members": 100,                   # generated members per validation row (Flow)
        "es_sampling_steps": None,           # None: use n_sampling_steps
        "es_max_rows": 512,                  # monitored validation rows (0 = all)
        "es_seed": 1990,
        "es_smoothing": 0.3,
        "restore_best": True,
    },

    # DeepQuantile baseline (same backbone and conditioning as the Flow).
    "baseline": {"n_quantiles": 100},

    "eval": {
        "n_series": 100,                     # generated series per test window
        "histogram_bins": 50,
        "ks_subsample": 20000,
        "seed": 1990,
    },
}


def deep_update(base: dict[str, Any], update: Mapping[str, Any] | None) -> dict[str, Any]:
    # Merge nested mappings into base in place and return it.
    for key, value in (update or {}).items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def load_config(path: str | Path | None = None,
                overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    # Return a fresh configuration: defaults, then an optional file, then overrides.
    #
    # JSON is always supported. YAML needs the optional pyyaml dependency.
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path is not None:
        config_path = Path(path).expanduser()
        if not config_path.is_file():
            raise FileNotFoundError(f"configuration file not found: {config_path}")
        text = config_path.read_text(encoding="utf-8")
        suffix = config_path.suffix.lower()
        if suffix in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:
                raise RuntimeError(
                    "YAML configs need PyYAML: pip install 'solflow-gen[yaml]'") from exc
            user = yaml.safe_load(text) or {}
        elif suffix == ".json":
            user = json.loads(text)
        else:
            raise ValueError("configuration files must use .json, .yaml or .yml")
        if not isinstance(user, dict):
            raise ValueError("the configuration file must contain a mapping")
        deep_update(cfg, user)
    deep_update(cfg, overrides)
    return cfg


def config_hash(cfg: Mapping[str, Any]) -> str:
    # Stable SHA-256 of a resolved configuration.
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()


class ConfigError(ValueError):
    # Raised when a configuration is internally inconsistent.
    pass


def validate_config(cfg: Mapping[str, Any]) -> None:
    # Check a resolved configuration; raise ConfigError on the first problem.
    data, model, train = cfg["data"], cfg["model"], cfg["train"]
    split, prior = cfg["split"], cfg["prior"]

    res = int(data["resolution_min"])
    if res <= 0 or 1440 % res:
        raise ConfigError("data.resolution_min must be a positive divisor of 1440")
    if not 0.0 < float(data["csi_min"]) < float(data["csi_max"]):
        raise ConfigError("data.csi_min must be > 0 and smaller than data.csi_max")
    if data["center"] not in ("clearsky", "mean"):
        raise ConfigError("data.center must be 'clearsky' or 'mean'")
    if int(cfg["task"]["forecast_days"]) < 1:
        raise ConfigError("task.forecast_days must be >= 1")

    if split["scheme"] not in SPLIT_SCHEMES:
        raise ConfigError(f"split.scheme must be one of {SPLIT_SCHEMES}")
    if split["train_mode"] not in ("folds", "all"):
        raise ConfigError("split.train_mode must be 'folds' or 'all'")
    if not 0.0 < float(split["val_frac"]) < 1.0:
        raise ConfigError("split.val_frac must be in (0, 1)")
    if int(split["n_folds"]) < 2:
        raise ConfigError("split.n_folds must be >= 2")

    if cfg["representation"]["kind"] not in REPRESENTATIONS:
        raise ConfigError(f"representation.kind must be one of {REPRESENTATIONS}")
    if prior["kernel"] not in ("matern12", "matern32", "rbf"):
        raise ConfigError("prior.kernel must be matern12, matern32 or rbf")
    if not 0.0 <= float(prior["enhancement"]["weight"]) <= 1.0:
        raise ConfigError("prior.enhancement.weight must be in [0, 1]")

    if int(model["n_regimes"]) < 2:
        raise ConfigError("model.n_regimes must be >= 2")
    if not 0.0 <= float(model["dropout"]) < 1.0:
        raise ConfigError("model.dropout must be in [0, 1)")
    scale = model["regime"].get("coordinate_scale")
    if scale is not None and (len(scale) != 4 or any(float(v) <= 0 for v in scale)):
        raise ConfigError("model.regime.coordinate_scale needs four positive values")
    for key in ("site_alpha_proportions", "site_alpha_transition"):
        if float(model["regime"][key]) < 0:
            raise ConfigError(f"model.regime.{key} cannot be negative")

    if int(train["epochs"]) < 1 or int(train["batch_size"]) < 1:
        raise ConfigError("train.epochs and train.batch_size must be positive")
    if str(train["sampler"]).lower() not in ("euler", "midpoint", "heun"):
        raise ConfigError("train.sampler must be euler, midpoint or heun")
    if str(train["amp"]).lower() not in ("off", "bf16"):
        raise ConfigError("train.amp must be 'off' or 'bf16'")
    if int(train["es_every"]) < 1 or int(train["es_members"]) < 2 or int(train["es_max_rows"]) < 0:
        raise ConfigError("train.es_every >= 1, train.es_members >= 2 and train.es_max_rows >= 0 required")
    if train["es_sampling_steps"] is not None and int(train["es_sampling_steps"]) < 1:
        raise ConfigError("train.es_sampling_steps must be positive or null")
    if not 0.0 <= float(train["es_smoothing"]) <= 1.0:
        raise ConfigError("train.es_smoothing must be in [0, 1]")
    if not 0.0 <= float(train["ema"]) < 1.0:
        raise ConfigError("train.ema must be in [0, 1)")
    if int(cfg["baseline"]["n_quantiles"]) < 2:
        raise ConfigError("baseline.n_quantiles must be >= 2")
    if int(cfg["eval"]["n_series"]) < 2:
        raise ConfigError("eval.n_series must be >= 2")
