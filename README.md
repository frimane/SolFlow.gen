<div align="center">

# ☀️ FranceSolFlow

### SolFlow — Conditional Flow Matching for probabilistic solar irradiance scenario generation

*Whole-day clear-sky-index trajectories, generated from solar geometry and site coordinates alone.*

<p>
  <img alt="Status" src="https://img.shields.io/badge/paper-under%20review-orange?style=flat-square">
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-flow%20matching-EE4C2C?style=flat-square&logo=pytorch&logoColor=white">
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-app-FF4B4B?style=flat-square&logo=streamlit&logoColor=white">
  <img alt="Data" src="https://img.shields.io/badge/data-Météo--France-005CA9?style=flat-square">
  <img alt="Stations" src="https://img.shields.io/badge/stations-90-2ea44f?style=flat-square">
  <a href="LICENSE"><img alt="License: GPL v3" src="https://img.shields.io/badge/license-GPLv3-blue?style=flat-square"></a>
</p>

[**Overview**](#-overview) · [**Demo**](#-demo) · [**Quick start**](#-quick-start) · [**Results**](#-results) · [**Stations map**](#-stations) · [**Citation**](#-citation) · [**License**](#-license) · [**Contact**](#-contact)

</div>

<!-- ───────────── HERO PLACEHOLDER ───────────── -->
<div align="center">

<!-- TODO: add a screenshot or GIF of the Streamlit app, then uncomment:
<img src="app_demo.gif" alt="SolFlow Streamlit app demo" width="90%">
-->

</div>

---

## ✨ Overview

Studies of **storage sizing** and **grid ramp-rate limits** need irradiance days that reproduce not only the *distribution* of values but also their *order in time*.

**SolFlow** is a flow-matching model that generates ensembles of complete daily trajectories of the clear-sky index (CSI) and converts them to GHI. **No measurements and no weather forecast are needed at generation time.**

| | |
|---|---|
| 🌅 **Full-day trajectories** | Not just a mean or a set of quantiles — realistic temporal structure (ramps, autocorrelation, roughness). |
| 🧭 **Physics-aware conditioning** | Solar geometry, calendar date, clear-sky GHI, and site position (latitude, longitude, altitude). |
| 📍 **Generalises to new sites** | Evaluated on 40 stations never seen during training, on a year never seen either. |
| 🖥️ **Interactive app** | Pick a point, a date range and an ensemble size; plot, inspect and download samples. |

> [!NOTE]
> The paper is **under review** (Frimane et al.). The full reference will be added here once available. Training code, trained weights and the Streamlit app are released.

## 🎬 Demo

<!-- TODO: add the Streamlit app here once it is ready.
     Options: live link badge, screenshot, or short GIF. -->

<div align="center">

[![Open the app](https://img.shields.io/badge/🚀_Live_app-coming_soon-lightgrey?style=for-the-badge)](#)

<!-- Example once deployed:
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://YOUR-APP.streamlit.app) -->

<!-- TODO: add screenshots, then uncomment:
<img src="screenshot_1.png" alt="App screenshot: ensemble of daily CSI trajectories" width="48%">
<img src="screenshot_2.png" alt="App screenshot: GHI scenarios and download" width="48%">
-->

</div>

## 🚀 Quick start

```bash
python -m pip install torch numpy pandas streamlit plotly pvlib
cd streamlit_app
python -m streamlit run app.py
```

> [!IMPORTANT]
> Start the app from **inside `streamlit_app/`**, which holds the app theme. The four trained checkpoints are in `streamlit_app/checkpoints/`.

> [!TIP]
> The checkpoints were trained with **McClear** (CAMS) clear-sky irradiance. The app draws its own clear-sky curve with **Ineichen** (pvlib, default) or **Haurwitz** to scale CSI into GHI. You may train, fine-tune or plot with any clear-sky model — choose one close to the one used for training.

## 🧠 How it works

```mermaid
flowchart LR
    A["Correlated noise<br/>around a seasonal<br/>clear-sky profile"] --> B["Learned velocity field<br/>(TCN + FiLM + attention)<br/>10 Heun steps"]
    C["Conditioning<br/>cos(zenith), daylight mask,<br/>day of year, time of day,<br/>clear-sky GHI, lat/lon/alt"] --> B
    B --> D["CSI trajectory<br/>(92 × 10 min)"]
    D --> E["× clear-sky GHI<br/>= GHI scenario"]
```

| Component | Details |
|---|---|
| **Method** | Conditional flow matching, integrated with 10 Heun steps |
| **Network** | Dilated temporal CNN, FiLM conditioning, one self-attention layer (hidden size 128, 4 heads) |
| **Inputs** | cos(zenith), daylight mask, time of year, position in day, clear-sky GHI, site position — *no measured history* |
| **Released weights** | 4 fold checkpoints in `streamlit_app/checkpoints/` — Gaussian CSI representation, no sky-regime conditioning |

## 🗂️ Data

| | |
|---|---|
| **Provider** | Météo-France |
| **Stations** | 90 ground stations · lat 41.9–51.0 °N · lon 4.4 °W–9.5 °E · altitude 1–1148 m (median ≈ 140 m) |
| **Measurements** | GHI every minute → 10-min grid; a step is kept if ≥ 60 % of its minutes are valid |
| **Clear sky** | McClear (CAMS) → CSI = GHI / clear-sky GHI |
| **Filters** | zenith < 85°, clear-sky GHI > 50 W/m², CSI ∈ [0.02, 1.8], suspect zeros dropped |
| **Day grid** | 92 steps (10 min) centred on solar noon |

## 🧪 Training & evaluation protocol

The split is **strict in both space and time**: test results are on stations *and* a year the model has never seen.

| | |
|---|---|
| **Train years** | 2017–2021 (2021 also used for validation / early stopping) |
| **Test year** | 2022 |
| **Folds** | 4 folds, 10 disjoint held-out stations each → **40 stations tested** |
| **Training stations** | the other 80 per fold, one checkpoint per fold |

| Fold | Test stations | Best epoch |
|:---:|:---:|:---:|
| 0 | 10 | 83 |
| 1 | 10 | 119 |
| 2 | 10 | 93 |
| 3 | 10 | 133 |

## 📊 Results

SolFlow is benchmarked against **DeepQuantile** (same backbone, pinball loss), whose marginal quantiles are turned into daily trajectories via **independent coupling** or a **Gaussian temporal copula**. Both models are assessed under Gaussian and logarithmic CSI representations; the ranking is the same for both.

*Test on 2022 · 10 unseen stations per fold · 4 folds.*

**Marginal accuracy — the two models are close.**
Mean pinball loss is **0.070** (SolFlow) vs **0.068** (DeepQuantile). SolFlow's empirical quantile frequencies stay within 2–3 % of nominal levels, and it obtains the lowest variogram score.

**Temporal structure — where SolFlow gains most.**

| Metric (lower is better) | **SolFlow** | DeepQuantile + Gaussian copula | DeepQuantile + independent |
|---|:---:|:---:|:---:|
| KS distance, largest daily ramp | **0.07** | 0.24 | 0.79 |
| Autocorrelation error, lags 1–10 (Gaussian) | **0.007** | 0.092 | 0.308 |
| Roughness error | **0.003 – 0.008** | 0.039 | 0.18 |

<div align="center">

| Largest-ramp distance | Autocorrelation error | Roughness error |
|:---:|:---:|:---:|
| **≈ 4× / 12×** larger for baselines | **13× / 45×** larger for baselines | **5–13× / 23–61×** larger for baselines |

</div>

> [!NOTE]
> **Sky-regime conditioning** lowers the largest-ramp distance further (0.07 → 0.04), at the cost of marginal accuracy and energy score.

## 📍 Stations

90 Météo-France ground stations spread across mainland France and Corsica (altitudes 1–1148 m).

<div align="center">
<img src="stations_map.png" alt="Map of the 90 Météo-France stations used in SolFlow" width="560">
</div>

## 📁 Repository layout

```text
SolFlow.gen/
├── solflow_gen/            # Python package
│   ├── data/               # data loading and preprocessing
│   ├── transforms/         # CSI representations (Gaussian, log)
│   ├── models/             # flow-matching network
│   ├── training/           # training loops
│   └── evaluation/         # metrics and benchmarks
├── configs/                # experiment configurations
├── streamlit_app/
│   ├── .streamlit/         # app theme
│   ├── checkpoints/        # 4 trained fold checkpoints
│   └── ui/                 # app interface components
├── pyproject.toml
├── LICENSE
└── README.md
```


## 📚 Citation

```bibtex
@unpublished{frimane_solflow,
  author = {Frimane, et al.},
  title  = {Conditional Flow matching for probabilistic solar irradiance scenario generation},
  note   = {Under review}
}
```

<!-- TODO: replace with the final reference when the paper is published. -->

## 🙏 Acknowledgements

Ground measurements from **Météo-France**; clear-sky irradiance from **McClear (CAMS)**; clear-sky models in the app from **pvlib**.

## 📄 License

This project is released under the **GNU General Public License v3.0** (GPL-3.0). See the [`LICENSE`](LICENSE) file for the full text.

## ✉️ Contact

**Azeddine Frimane** — [Azeddine.frimane@yahoo.com](mailto:Azeddine.frimane@yahoo.com)

Questions, bug reports and collaboration ideas are welcome — feel free to email me or open an issue.

<div align="center">
<sub>Built for storage sizing, ramp-rate studies, and anyone who needs solar days that behave like real ones. ☀️</sub>
</div>
