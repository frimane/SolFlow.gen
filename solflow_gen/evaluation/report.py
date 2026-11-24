# Turn an evaluation JSON into a readable summary: a Markdown table, a CSV, and (if
# matplotlib is installed) three small figures. It only reads the JSON, so a report
# can be rebuilt any time without the checkpoints or the data.
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

MODELS = (("flow", "SolFlow_gen (Flow)"),
          ("deepquantile_independent_scenarios", "DeepQuantile, independent"),
          ("deepquantile_gaussian_copula_scenarios", "DeepQuantile + Gaussian copula"))
# (label, section, key, lower_is_better)
METRICS = (
    ("Energy score", "joint_metrics", "energy_score"),
    ("Variogram score", "joint_metrics", "variogram_score"),
    ("Joint proxy", "transport_quality", "joint_transport_proxy"),
    ("KS values", "value_distribution", "ks_statistic"),
    ("KS ramps", "ramp_distribution", "ks_statistic"),
    ("KS daily total", "daily_total_distribution", "ks_statistic"),
    ("ACF MAE", "temporal_autocorrelation", "mae"),
    ("Covariance nMAE", "cross_step_covariance", "normalized_mae"),
    ("Roughness error", "per_trajectory_temporal_coherence", "roughness_error"),
    ("Regime mix MAE", "regimes", "proportion_mae"),
)
SHORT = {"flow": "Flow", "deepquantile_independent_scenarios": "DQ indep.",
         "deepquantile_gaussian_copula_scenarios": "DQ copula"}
COLORS = {"flow": "#1f6f8b", "deepquantile_independent_scenarios": "#c9772b",
          "deepquantile_gaussian_copula_scenarios": "#6a8d3c", "real": "#222222"}


def _get(d: dict[str, Any], *path):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return None
        d = d[p]
    return d


def _fmt(v) -> str:
    return "-" if v is None or v != v else f"{v:.4f}"


def table_rows(agg: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for key, name in MODELS:
        sec = agg.get(key)
        if not sec:
            continue
        row = {"model": name}
        for label, section, metric in METRICS:
            row[label] = _get(sec, section, metric)
        rows.append(row)
    mq = agg.get("marginal_quantile_metrics", {})
    for row, (key, _) in zip(rows, [m for m in MODELS if agg.get(m[0])]):
        model = "flow" if key == "flow" else "deepquantile"
        row["Pinball"] = _get(mq, model, "mean_pinball")
        row["Coverage error"] = _get(mq, model, "mean_absolute_coverage_error")
    return rows


def write_tables(payload: dict[str, Any], out: Path) -> list[dict[str, Any]]:
    rows = table_rows(payload["aggregate"])
    if not rows:
        raise ValueError("the evaluation JSON has no model sections to report")
    cols = list(rows[0])
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] + ["---:"] * (len(cols) - 1)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join([r["model"]] + [_fmt(r.get(c)) for c in cols[1:]]) + " |")
    n = payload["aggregate"].get("n_folds", len(payload.get("per_fold", [])))
    text = (f"# Evaluation summary\n\nMean over {n} fold(s). Lower is better for every column. "
            "Scenario columns compare generated days with held-out real days; Pinball and "
            "Coverage error compare marginal quantiles at identical levels.\n\n" + "\n".join(lines) + "\n")
    (out / "summary.md").write_text(text, encoding="utf-8")
    return rows


def _plots(payload: dict[str, Any], out: Path) -> list[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[report] matplotlib not installed; skipping figures")
        return []
    agg = payload["aggregate"]
    present = [(k, n) for k, n in MODELS if agg.get(k)]
    written = []
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})

    # 1. autocorrelation by lag: real days vs each model (lists are averaged over folds,
    # because the aggregate keeps scalars only)
    import numpy as np

    def mean_by_lag(key, field):
        rows = [f[key]["temporal_autocorrelation"][field] for f in payload.get("per_fold", [])
                if key in f and "temporal_autocorrelation" in f[key]]
        return np.nanmean(np.asarray(rows, float), axis=0) if rows else None

    ref_key = present[0][0]
    real_acf = mean_by_lag(ref_key, "real_mean_by_lag")
    if real_acf is not None:
        fig, ax = plt.subplots(figsize=(5.4, 3.4))
        lags = range(1, len(real_acf) + 1)
        ax.plot(lags, real_acf, "o-", color=COLORS["real"], label="real days")
        for k, n in present:
            acf = mean_by_lag(k, "synthetic_mean_by_lag")
            if acf is not None:
                ax.plot(lags, acf, "o-", color=COLORS[k], label=n)
        ax.set(xlabel="lag (steps)", ylabel="mean autocorrelation", title="Temporal dependence")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        p = out / "autocorrelation.png"
        fig.savefig(p, dpi=160)
        plt.close(fig)
        written.append(p)

    # 2. quantile calibration: coverage vs nominal level
    mq = agg.get("marginal_quantile_metrics", {})
    if mq.get("quantile_levels"):
        fig, ax = plt.subplots(figsize=(4.4, 4.0))
        lv = mq["quantile_levels"]
        ax.plot([0, 1], [0, 1], "--", color="#888888", lw=1)
        for model, name in (("flow", "SolFlow_gen (Flow)"), ("deepquantile", "DeepQuantile")):
            if model in mq:
                ax.plot(lv, mq[model]["coverage_by_quantile"], "-", color=COLORS["flow" if model == "flow"
                        else "deepquantile_independent_scenarios"], label=name)
        ax.set(xlabel="nominal level", ylabel="observed coverage", title="Marginal calibration")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        p = out / "calibration.png"
        fig.savefig(p, dpi=160)
        plt.close(fig)
        written.append(p)

    # 3. joint scores side by side
    rows = table_rows(agg)
    if rows:
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
        for ax, metric in zip(axes, ("Energy score", "Variogram score")):
            vals = [r.get(metric) for r in rows]
            ax.bar(range(len(rows)), [v if v is not None else 0 for v in vals],
                   color=[COLORS[k] for k, _ in present])
            ax.set_xticks(range(len(rows)), [SHORT[k] for k, _ in present], fontsize=8)
            ax.set_title(metric + " (lower is better)", fontsize=9)
        fig.tight_layout()
        p = out / "joint_scores.png"
        fig.savefig(p, dpi=160)
        plt.close(fig)
        written.append(p)
    return written


def build_report(eval_json: str | Path, out_dir: str | Path, plots: bool = True) -> Path:
    payload = json.loads(Path(eval_json).read_text(encoding="utf-8"))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_tables(payload, out)
    if plots:
        _plots(payload, out)
    print(f"[report] wrote {out / 'summary.md'}")
    return out
