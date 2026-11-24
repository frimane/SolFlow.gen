# Command line interface: one solflow-gen command with a subcommand per step.
#
# preprocess  CSV files -> window archive (.npz)
# train       train SolFlow_gen and/or DeepQuantile
# evaluate    score the trained models on their held-out folds
# report      turn an evaluation JSON into tables and figures
# generate    sample scenarios for any site and dates from a checkpoint
# run         train + evaluate + report in one go
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .config import deep_update, load_config, validate_config


def _parse_set(items: list[str] | None) -> dict[str, Any]:
    # --set a.b=value (value is JSON when it parses, otherwise a plain string)
    out: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        key, raw = item.split("=", 1)
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return out


def _config(args, **extra) -> dict[str, Any]:
    over: dict[str, Any] = {}
    deep_update(over, _parse_set(getattr(args, "set", None)))
    deep_update(over, {k: v for k, v in extra.items() if v})
    cfg = load_config(getattr(args, "config", None), over)
    validate_config(cfg)
    return cfg


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", help="JSON or YAML configuration (defaults are used when omitted)")
    p.add_argument("--set", action="append", metavar="KEY=VALUE",
                   help="override any config value, e.g. --set train.epochs=50 (repeatable)")


def cmd_preprocess(args) -> None:
    from .data.preprocess import preprocess
    paths: dict[str, Any] = {}
    if args.input:
        paths["raw_glob"] = args.input
    if args.output:
        paths["windows"] = args.output
    site = {k: getattr(args, k) for k in ("latitude", "longitude", "altitude") if getattr(args, k) is not None}
    if args.name:
        site["name"] = args.name
    cfg = _config(args, paths=paths, site=site)
    preprocess(cfg, multi_site=args.multi_site)


def cmd_train(args) -> None:
    from .training import train_models
    split: dict[str, Any] = {}
    if args.split_mode:
        split["train_mode"] = args.split_mode
    train = {"epochs": args.epochs} if args.epochs else {}
    model = {} if args.regime is None else {"condition_regime": bool(args.regime)}
    cfg = _config(args, paths={k: v for k, v in (("windows", args.windows), ("models_dir", args.output)) if v},
                  split=split, train=train, model=model,
                  representation={"kind": args.representation} if args.representation else {})
    train_models(cfg, families=tuple(args.models), fold=args.fold)


def cmd_evaluate(args) -> None:
    from .evaluation.evaluate import evaluate_run
    cfg = _config(args)
    evaluate_run(args.windows or cfg["paths"]["windows"], args.models_dir or cfg["paths"]["models_dir"],
                 args.output, folds=args.fold, models=tuple(args.models), n_series=args.n_series,
                 copula_mode=args.copula)


def cmd_report(args) -> None:
    from .evaluation.report import build_report
    build_report(args.input, args.output, plots=not args.no_plots)


def cmd_generate(args) -> None:
    from .geometry import day_conditions
    from .models.checkpoint import load_model
    model = load_model(args.checkpoint, args.device)
    noon_col = (model.extra or {}).get("noon_col")
    if noon_col is None:
        raise SystemExit("this checkpoint does not record the day-grid layout (noon_col); retrain it")
    cfg = model.cfg
    cond = day_conditions(args.latitude, args.longitude, args.altitude, args.start, args.end or args.start,
                          model.K, int(noon_col), cfg["data"], forecast_days=model.n_days,
                          altitude_scale_m=float(cfg["site"].get("altitude_scale_m", 2000.0)))
    rng = np.random.default_rng(args.seed)
    if model.family == "flow" and model.regime_enabled and args.sequence:
        csi, labels = model.generate_sequence(cond, args.n_scenarios, rng)
        extra = {"regime": labels}
    else:
        csi, extra = model.generate(cond, args.n_scenarios, rng), {}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    ghi = csi * np.nan_to_num(cond["fut_ghi_cs"])[:, None, :]
    if out.suffix.lower() == ".csv":
        import pandas as pd
        n, s, h = csi.shape
        rows, scen, step = np.meshgrid(np.arange(n), np.arange(s), np.arange(h), indexing="ij")
        keep = np.broadcast_to(cond["fut_mask"][:, None, :], csi.shape)
        df = pd.DataFrame({"date_ord": cond["date_ord"][rows][keep], "scenario": scen[keep], "step": step[keep],
                           "csi": csi[keep], "ghi": ghi[keep]})
        df.to_csv(out, index=False)
    else:
        np.savez_compressed(out, csi=csi, ghi=ghi.astype(np.float32), **cond, K=model.K, **extra)
    print(f"wrote {out}: {csi.shape[0]} day(s) x {csi.shape[1]} scenarios x {csi.shape[2]} steps")


def cmd_run(args) -> None:
    from .evaluation.evaluate import evaluate_run
    from .evaluation.report import build_report
    from .training import train_models
    cfg = _config(args, paths={k: v for k, v in (("windows", args.windows), ("models_dir", args.output)) if v})
    out = Path(cfg["paths"]["models_dir"])
    if args.dry_run:
        print(f"would train {args.models} on {cfg['paths']['windows']} (scheme={cfg['split']['scheme']}) "
              f"-> {out}, then evaluate and write a report under {out / 'report'}")
        return
    train_models(cfg, families=tuple(args.models))
    evaluate_run(cfg["paths"]["windows"], out, out / "evaluation.json", models=tuple(args.models),
                 n_series=args.n_series, copula_mode=args.copula)
    build_report(out / "evaluation.json", out / "report")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="solflow-gen", description="Geometry-conditioned generator of "
                                "clear-sky-index days, with a DeepQuantile baseline.")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("preprocess", help="CSV files -> window archive")
    _add_common(s)
    s.add_argument("--input", help="glob of CSV files (single-site mode)")
    s.add_argument("--output", help="window archive to write (.npz)")
    s.add_argument("--multi-site", action="store_true", help="read every entry of `sites` in the config")
    s.add_argument("--name")
    for k in ("latitude", "longitude", "altitude"):
        s.add_argument(f"--{k}", type=float)
    s.set_defaults(func=cmd_preprocess)

    s = sub.add_parser("train", help="train the Flow and/or DeepQuantile")
    _add_common(s)
    s.add_argument("--windows")
    s.add_argument("--output", help="directory for checkpoints")
    s.add_argument("--models", nargs="+", default=["flow", "deepquantile"], choices=["flow", "deepquantile"])
    s.add_argument("--epochs", type=int)
    s.add_argument("--representation", choices=["log", "gauss"])
    s.add_argument("--regime", dest="regime", action="store_true", default=None,
                   help="condition the Flow on the 4-state sky regime")
    s.add_argument("--no-regime", dest="regime", action="store_false")
    s.add_argument("--split-mode", choices=["folds", "all"])
    s.add_argument("--fold", type=int, help="train a single fold")
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("evaluate", help="score trained models on held-out folds")
    _add_common(s)
    s.add_argument("--windows")
    s.add_argument("--models-dir")
    s.add_argument("--output", required=True, help="evaluation JSON to write")
    s.add_argument("--models", nargs="+", default=["flow", "deepquantile"], choices=["flow", "deepquantile"])
    s.add_argument("--fold", type=int, nargs="+")
    s.add_argument("--n-series", type=int)
    s.add_argument("--copula", choices=["gaussian", "none"], default="gaussian")
    s.set_defaults(func=cmd_evaluate)

    s = sub.add_parser("report", help="evaluation JSON -> tables and figures")
    s.add_argument("--input", required=True)
    s.add_argument("--output", required=True)
    s.add_argument("--no-plots", action="store_true")
    s.set_defaults(func=cmd_report)

    s = sub.add_parser("generate", help="sample scenarios for a site and dates")
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--latitude", type=float, required=True)
    s.add_argument("--longitude", type=float, required=True)
    s.add_argument("--altitude", type=float, required=True)
    s.add_argument("--start", required=True, help="first date, YYYY-MM-DD")
    s.add_argument("--end", help="last date (default: the first date)")
    s.add_argument("--n-scenarios", type=int, default=100)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--sequence", action="store_true", help="connected regime sequence (regime models only)")
    s.add_argument("--device", default="auto")
    s.add_argument("--output", required=True, help=".npz or .csv")
    s.set_defaults(func=cmd_generate)

    s = sub.add_parser("run", help="train + evaluate + report in one go")
    _add_common(s)
    s.add_argument("--windows")
    s.add_argument("--output", help="run directory")
    s.add_argument("--models", nargs="+", default=["flow", "deepquantile"], choices=["flow", "deepquantile"])
    s.add_argument("--n-series", type=int)
    s.add_argument("--copula", choices=["gaussian", "none"], default="gaussian")
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(func=cmd_run)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
