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
</p>

[**Overview**](#-overview) · [**Demo**](#-demo) · [**Quick start**](#-quick-start) · [**Results**](#-results) · [**Stations map**](#-stations) · [**Citation**](#-citation)

</div>

<!-- ───────────── HERO PLACEHOLDER ───────────── -->
<div align="center">

<!-- TODO: replace with a screenshot or GIF of the Streamlit app, e.g. docs/assets/app_demo.gif -->
<img src="docs/assets/app_demo.gif" alt="SolFlow Streamlit app demo" width="90%">

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

<img src="docs/assets/screenshot_1.png" alt="App screenshot: ensemble of daily CSI trajectories" width="48%">
<img src="docs/assets/screenshot_2.png" alt="App screenshot: GHI scenarios and download" width="48%">

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

90 Météo-France stations. Hover or click a marker for name and altitude.

**Legend:** 🔴 test fold 0 · 🔵 test fold 1 · 🟢 test fold 2 · 🟡 test fold 3 · ⚪ always used for training

```geojson
{"type":"FeatureCollection","features":[
{"type":"Feature","geometry":{"type":"Point","coordinates":[3.533,45.461]},"properties":{"id":0,"title":"ST-GERMAIN-L_HE","description":"Altitude 1070 m · train in all folds","altitude_m":1070,"test_fold":null,"marker-color":"#8b949e","marker-size":"small","marker-symbol":""}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[2.991,45.075]},"properties":{"id":1,"title":"COLTINES","description":"Altitude 979 m · train in all folds","altitude_m":979,"test_fold":null,"marker-color":"#8b949e","marker-size":"small","marker-symbol":""}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[-0.691,44.831]},"properties":{"id":2,"title":"BORDEAUX-MERIGNAC","description":"Altitude 47 m · test fold 2","altitude_m":47,"test_fold":2,"marker-color":"#3fb950","marker-size":"small","marker-symbol":"2"}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[0.601,43.689]},"properties":{"id":3,"title":"AUCH","description":"Altitude 122 m · train in all folds","altitude_m":122,"test_fold":null,"marker-color":"#8b949e","marker-size":"small","marker-symbol":""}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[3.965,43.576]},"properties":{"id":4,"title":"MONTPELLIER-AEROPORT","description":"Altitude 1 m · test fold 2","altitude_m":1,"test_fold":2,"marker-color":"#3fb950","marker-size":"small","marker-symbol":"2"}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[2.36,47.059]},"properties":{"id":5,"title":"BOURGES","description":"Altitude 161 m · train in all folds","altitude_m":161,"test_fold":null,"marker-color":"#8b949e","marker-size":"small","marker-symbol":""}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[0.314,46.594]},"properties":{"id":6,"title":"POITIERS-BIARD","description":"Altitude 125 m · test fold 2","altitude_m":125,"test_fold":2,"marker-color":"#3fb950","marker-size":"small","marker-symbol":"2"}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[-1.609,47.15]},"properties":{"id":7,"title":"NANTES-BOUGUENAIS","description":"Altitude 26 m · test fold 1","altitude_m":26,"test_fold":1,"marker-color":"#2f81f7","marker-size":"small","marker-symbol":"1"}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[-1.534,43.469]},"properties":{"id":8,"title":"BIARRITZ-PAYS-BASQUE","description":"Altitude 71 m · test fold 0","altitude_m":71,"test_fold":0,"marker-color":"#e5534b","marker-size":"small","marker-symbol":"0"}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[5.216,43.438]},"properties":{"id":9,"title":"MARIGNANE","description":"Altitude 9 m · train in all folds","altitude_m":9,"test_fold":null,"marker-color":"#8b949e","marker-size":"small","marker-symbol":""}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[6.221,48.688]},"properties":{"id":10,"title":"NANCY-ESSEY","description":"Altitude 212 m · test fold 1","altitude_m":212,"test_fold":1,"marker-color":"#2f81f7","marker-size":"small","marker-symbol":"1"}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[5.989,47.249]},"properties":{"id":11,"title":"BESANCON","description":"Altitude 307 m · train in all folds","altitude_m":307,"test_fold":null,"marker-color":"#8b949e","marker-size":"small","marker-symbol":""}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[6.452,48.211]},"properties":{"id":12,"title":"EPINAL","description":"Altitude 317 m · test fold 2","altitude_m":317,"test_fold":2,"marker-color":"#3fb950","marker-size":"small","marker-symbol":"2"}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[2.534,49.015]},"properties":{"id":13,"title":"ROISSY","description":"Altitude 108 m · train in all folds","altitude_m":108,"test_fold":null,"marker-color":"#8b949e","marker-size":"small","marker-symbol":""}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[3.149,45.787]},"properties":{"id":14,"title":"CLERMONT-FD","description":"Altitude 331 m · train in all folds","altitude_m":331,"test_fold":null,"marker-color":"#8b949e","marker-size":"small","marker-symbol":""}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[1.726,45.942]},"properties":{"id":15,"title":"BOURGANEUF","description":"Altitude 513 m · train in all folds","altitude_m":513,"test_fold":null,"marker-color":"#8b949e","marker-size":"small","marker-symbol":""}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[1.244,45.514]},"properties":{"id":16,"title":"ST_YRIEIX_LA_PE","description":"Altitude 404 m · train in all folds","altitude_m":404,"test_fold":null,"marker-color":"#8b949e","marker-size":"small","marker-symbol":""}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[-0.416,43.385]},"properties":{"id":17,"title":"PAU-UZEIN","description":"Altitude 183 m · train in all folds","altitude_m":183,"test_fold":null,"marker-color":"#8b949e","marker-size":"small","marker-symbol":""}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[4.406,43.857]},"properties":{"id":18,"title":"NIMES-COURBESSAC","description":"Altitude 59 m · train in all folds","altitude_m":59,"test_fold":null,"marker-color":"#8b949e","marker-size":"small","marker-symbol":""}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[-1.382,46.705]},"properties":{"id":19,"title":"LA_ROCHE_SUR_YON","description":"Altitude 88 m · train in all folds","altitude_m":88,"test_fold":null,"marker-color":"#8b949e","marker-size":"small","marker-symbol":""}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[4.949,45.721]},"properties":{"id":20,"title":"LYON-BRON","description":"Altitude 202 m · train in all folds","altitude_m":202,"test_fold":null,"marker-color":"#8b949e","marker-size":"small","marker-symbol":""}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[4.904,48.631]},"properties":{"id":21,"title":"ST-DIZIER","description":"Altitude 139 m · test fold 3","altitude_m":139,"test_fold":3,"marker-color":"#d29922","marker-size":"small","marker-symbol":"3"}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[0.727,47.444]},"properties":{"id":22,"title":"TOURS","description":"Altitude 108 m · test fold 0","altitude_m":108,"test_fold":0,"marker-color":"#e5534b","marker-size":"small","marker-symbol":"0"}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[1.473,45.147]},"properties":{"id":23,"title":"BRIVE","description":"Altitude 115 m · train in all folds","altitude_m":115,"test_fold":null,"marker-color":"#8b949e","marker-size":"small","marker-symbol":""}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[-0.614,47.479]},"properties":{"id":24,"title":"BEAUCOUZE","description":"Altitude 50 m · train in all folds","altitude_m":50,"test_fold":null,"marker-color":"#8b949e","marker-size":"small","marker-symbol":""}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[4.733,44.581]},"properties":{"id":25,"title":"MONTELIMAR","description":"Altitude 73 m · train in all folds","altitude_m":73,"test_fold":null,"marker-color":"#8b949e","marker-size":"small","marker-symbol":""}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[3.399,46.167]},"properties":{"id":26,"title":"VICHY-CHARMEIL","description":"Altitude 249 m · train in all folds","altitude_m":249,"test_fold":null,"marker-color":"#8b949e","marker-size":"small","marker-symbol":""}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[-4.383,48.45]},"properties":{"id":27,"title":"BREST-GUIPAVAS","description":"Altitude 94 m · test fold 1","altitude_m":94,"test_fold":1,"marker-color":"#2f81f7","marker-size":"small","marker-symbol":"1"}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[2.127,49.446]},"properties":{"id":28,"title":"BEAUVAIS-TILLE","description":"Altitude 89 m · test fold 3","altitude_m":89,"test_fold":3,"marker-color":"#d29922","marker-size":"small","marker-symbol":"3"}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[-0.456,49.18]},"properties":{"id":29,"title":"CAEN-CARPIQUET","description":"Altitude 67 m · test fold 1","altitude_m":67,"test_fold":1,"marker-color":"#2f81f7","marker-size":"small","marker-symbol":"1"}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[1.379,43.621]},"properties":{"id":30,"title":"TOULOUSE-BLAGNAC","description":"Altitude 151 m · train in all folds","altitude_m":151,"test_fold":null,"marker-color":"#8b949e","marker-size":"small","marker-symbol":""}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[5.313,45.364]},"properties":{"id":31,"title":"GRENOBLE-ST_GEOIRS","description":"Altitude 384 m · train in all folds","altitude_m":384,"test_fold":null,"marker-color":"#8b949e","marker-size":"small","marker-symbol":""}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[1.175,45.861]},"properties":{"id":32,"title":"LIMOGES-BELLEGARDE","description":"Altitude 402 m · train in all folds","altitude_m":402,"test_fold":null,"marker-color":"#8b949e","marker-size":"small","marker-symbol":""}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[3.805,46.051]},"properties":{"id":33,"title":"ST-NICOLAS","description":"Altitude 1022 m · train in all folds","altitude_m":1022,"test_fold":null,"marker-color":"#8b949e","marker-size":"small","marker-symbol":""}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[4.794,46.294]},"properties":{"id":34,"title":"MACON","description":"Altitude 219 m · test fold 1","altitude_m":219,"test_fold":1,"marker-color":"#2f81f7","marker-size":"small","marker-symbol":"1"}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[7.64,48.549]},"properties":{"id":35,"title":"STRASBOURG-ENTZHEIM","description":"Altitude 150 m · test fold 0","altitude_m":150,"test_fold":0,"marker-color":"#e5534b","marker-size":"small","marker-symbol":"0"}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[5.088,47.268]},"properties":{"id":36,"title":"DIJON-LONGVIC","description":"Altitude 219 m · train in all folds","altitude_m":219,"test_fold":null,"marker-color":"#8b949e","marker-size":"small","marker-symbol":""}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[2.01,48.774]},"properties":{"id":37,"title":"TRAPPES","description":"Altitude 167 m · train in all folds","altitude_m":167,"test_fold":null,"marker-color":"#8b949e","marker-size":"small","marker-symbol":""}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[3.764,45.075]},"properties":{"id":38,"title":"LE_PUY-LOUDES","description":"Altitude 833 m · train in all folds","altitude_m":833,"test_fold":null,"marker-color":"#8b949e","marker-size":"small","marker-symbol":""}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[-2.852,48.535]},"properties":{"id":39,"title":"ST_BRIEUC","description":"Altitude 135 m · test fold 3","altitude_m":135,"test_fold":3,"marker-color":"#d29922","marker-size":"small","marker-symbol":"3"}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[2.873,42.737]},"properties":{"id":40,"title":"PERPIGNAN","description":"Altitude 42 m · test fold 3","altitude_m":42,"test_fold":3,"marker-color":"#d29922","marker-size":"small","marker-symbol":"3"}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[2.397,48.718]},"properties":{"id":41,"title":"ORLY","description":"Altitude 86 m · train in all folds","altitude_m":86,"test_fold":null,"marker-color":"#8b949e","marker-size":"small","marker-symbol":""}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[0.11,48.445]},"properties":{"id":42,"title":"ALENCON","description":"Altitude 143 m · train in all folds","altitude_m":143,"test_fold":null,"marker-color":"#8b949e","marker-size":"small","marker-symbol":""}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[4.263,46.972]},"properties":{"id":43,"title":"AUTUN","description":"Altitude 303 m · test fold 3","altitude_m":303,"test_fold":3,"marker-color":"#d29922","marker-size":"small","marker-symbol":"3"}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[2.833,45.303]},"properties":{"id":44,"title":"MARCENAT","description":"Altitude 1075 m · train in all folds","altitude_m":1075,"test_fold":null,"marker-color":"#8b949e","marker-size":"small","marker-symbol":""}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[2.419,44.891]},"properties":{"id":45,"title":"AURILLAC","description":"Altitude 632 m · test fold 3","altitude_m":632,"test_fold":3,"marker-color":"#d29922","marker-size":"small","marker-symbol":"3"}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[4.02,48.325]},"properties":{"id":46,"title":"TROYES-BARBEREY","description":"Altitude 112 m · test fold 0","altitude_m":112,"test_fold":0,"marker-color":"#e5534b","marker-size":"small","marker-symbol":"0"}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[1.212,47.679]},"properties":{"id":47,"title":"BLOIS","description":"Altitude 119 m · test fold 3","altitude_m":119,"test_fold":3,"marker-color":"#d29922","marker-size":"small","marker-symbol":"3"}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[6.509,44.571]},"properties":{"id":48,"title":"EMBRUN","description":"Altitude 873 m · test fold 3","altitude_m":873,"test_fold":3,"marker-color":"#d29922","marker-size":"small","marker-symbol":"3"}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[2.116,43.915]},"properties":{"id":49,"title":"ALBI","description":"Altitude 172 m · test fold 0","altitude_m":172,"test_fold":0,"marker-color":"#e5534b","marker-size":"small","marker-symbol":"0"}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[5.878,45.641]},"properties":{"id":50,"title":"CHAMBERY-AIX","description":"Altitude 235 m · test fold 0","altitude_m":235,"test_fold":0,"marker-color":"#e5534b","marker-size":"small","marker-symbol":"0"}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[2.296,43.215]},"properties":{"id":51,"title":"CARCASSONNE","description":"Altitude 128 m · train in all folds","altitude_m":128,"test_fold":null,"marker-color":"#8b949e","marker-size":"small","marker-symbol":""}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[1.741,46.87]},"properties":{"id":52,"title":"CHATEAUROUX_DEOLS","description":"Altitude 158 m · test fold 1","altitude_m":158,"test_fold":1,"marker-color":"#2f81f7","marker-size":"small","marker-symbol":"1"}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[5.595,47.507]},"properties":{"id":53,"title":"CHARGEY-LES-GRAY","description":"Altitude 244 m · train in all folds","altitude_m":244,"test_fold":null,"marker-color":"#8b949e","marker-size":"small","marker-symbol":""}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[-2.076,48.585]},"properties":{"id":54,"title":"DINARD","description":"Altitude 65 m · train in all folds","altitude_m":65,"test_fold":null,"marker-color":"#8b949e","marker-size":"small","marker-symbol":""}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[-2.714,47.604]},"properties":{"id":55,"title":"VANNES-SENE","description":"Altitude 3 m · train in all folds","altitude_m":3,"test_fold":null,"marker-color":"#8b949e","marker-size":"small","marker-symbol":""}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[4.477,46.628]},"properties":{"id":56,"title":"MT-ST-VINCENT","description":"Altitude 601 m · test fold 2","altitude_m":601,"test_fold":2,"marker-color":"#3fb950","marker-size":"small","marker-symbol":"2"}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[9.485,42.541]},"properties":{"id":57,"title":"BASTIA","description":"Altitude 10 m · test fold 0","altitude_m":10,"test_fold":0,"marker-color":"#e5534b","marker-size":"small","marker-symbol":"0"}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[8.793,41.918]},"properties":{"id":58,"title":"AJACCIO","description":"Altitude 5 m · test fold 1","altitude_m":5,"test_fold":1,"marker-color":"#2f81f7","marker-size":"small","marker-symbol":"1"}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[-1.734,48.069]},"properties":{"id":59,"title":"RENNES-ST_JACQUES","description":"Altitude 36 m · train in all folds","altitude_m":36,"test_fold":null,"marker-color":"#8b949e","marker-size":"small","marker-symbol":""}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[5.058,44.084]},"properties":{"id":60,"title":"CARPENTRAS","description":"Altitude 98 m · test fold 1","altitude_m":98,"test_fold":1,"marker-color":"#2f81f7","marker-size":"small","marker-symbol":"1"}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[3.206,49.818]},"properties":{"id":61,"title":"ST_QUENTIN","description":"Altitude 98 m · test fold 1","altitude_m":98,"test_fold":1,"marker-color":"#2f81f7","marker-size":"small","marker-symbol":"1"}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[6.386,43.383]},"properties":{"id":62,"title":"LE_LUC","description":"Altitude 80 m · test fold 0","altitude_m":80,"test_fold":0,"marker-color":"#e5534b","marker-size":"small","marker-symbol":"0"}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[7.209,43.649]},"properties":{"id":63,"title":"NICE","description":"Altitude 2 m · test fold 2","altitude_m":2,"test_fold":2,"marker-color":"#3fb950","marker-size":"small","marker-symbol":"2"}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[0.595,44.172]},"properties":{"id":64,"title":"AGEN-LA_GARENNE","description":"Altitude 58 m · train in all folds","altitude_m":58,"test_fold":null,"marker-color":"#8b949e","marker-size":"small","marker-symbol":""}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[0.543,45.787]},"properties":{"id":65,"title":"MONTEMBOEUF","description":"Altitude 247 m · train in all folds","altitude_m":247,"test_fold":null,"marker-color":"#8b949e","marker-size":"small","marker-symbol":""}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[3.844,44.858]},"properties":{"id":66,"title":"LANDOS-CHARBON","description":"Altitude 1148 m · train in all folds","altitude_m":1148,"test_fold":null,"marker-color":"#8b949e","marker-size":"small","marker-symbol":""}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[1.956,50.96]},"properties":{"id":67,"title":"CALAIS-MARCK","description":"Altitude 2 m · test fold 0","altitude_m":2,"test_fold":0,"marker-color":"#e5534b","marker-size":"small","marker-symbol":"0"}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[0.0,43.188]},"properties":{"id":68,"title":"TARBES-LOURDES-PYRENEES","description":"Altitude 360 m · train in all folds","altitude_m":360,"test_fold":null,"marker-color":"#8b949e","marker-size":"small","marker-symbol":""}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[4.472,48.353]},"properties":{"id":69,"title":"MATHAUX-ETAPE","description":"Altitude 145 m · train in all folds","altitude_m":145,"test_fold":null,"marker-color":"#8b949e","marker-size":"small","marker-symbol":""}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[2.948,46.718]},"properties":{"id":70,"title":"LURCY-LEVIS_SA","description":"Altitude 225 m · train in all folds","altitude_m":225,"test_fold":null,"marker-color":"#8b949e","marker-size":"small","marker-symbol":""}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[1.107,43.005]},"properties":{"id":71,"title":"ST_GIRONS","description":"Altitude 414 m · test fold 2","altitude_m":414,"test_fold":2,"marker-color":"#3fb950","marker-size":"small","marker-symbol":"2"}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[1.377,44.028]},"properties":{"id":72,"title":"MONTAUBAN","description":"Altitude 106 m · train in all folds","altitude_m":106,"test_fold":null,"marker-color":"#8b949e","marker-size":"small","marker-symbol":""}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[0.524,44.823]},"properties":{"id":73,"title":"BERGERAC","description":"Altitude 49 m · train in all folds","altitude_m":49,"test_fold":null,"marker-color":"#8b949e","marker-size":"small","marker-symbol":""}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[-1.07,43.69]},"properties":{"id":74,"title":"DAX","description":"Altitude 31 m · test fold 3","altitude_m":31,"test_fold":3,"marker-color":"#d29922","marker-size":"small","marker-symbol":"3"}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[1.501,48.46]},"properties":{"id":75,"title":"CHARTRES","description":"Altitude 155 m · test fold 2","altitude_m":155,"test_fold":2,"marker-color":"#3fb950","marker-size":"small","marker-symbol":"2"}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[-4.161,47.973]},"properties":{"id":76,"title":"QUIMPER","description":"Altitude 82 m · test fold 0","altitude_m":82,"test_fold":0,"marker-color":"#e5534b","marker-size":"small","marker-symbol":"0"}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[0.366,43.127]},"properties":{"id":77,"title":"CAMPISTROUS","description":"Altitude 593 m · test fold 1","altitude_m":593,"test_fold":1,"marker-color":"#2f81f7","marker-size":"small","marker-symbol":"1"}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[3.02,44.118]},"properties":{"id":78,"title":"MILLAU","description":"Altitude 712 m · train in all folds","altitude_m":712,"test_fold":null,"marker-color":"#8b949e","marker-size":"small","marker-symbol":""}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[4.581,47.849]},"properties":{"id":79,"title":"CHATILLON_SEINE","description":"Altitude 262 m · train in all folds","altitude_m":262,"test_fold":null,"marker-color":"#8b949e","marker-size":"small","marker-symbol":""}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[2.68,48.61]},"properties":{"id":80,"title":"MELUN","description":"Altitude 91 m · train in all folds","altitude_m":91,"test_fold":null,"marker-color":"#8b949e","marker-size":"small","marker-symbol":""}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[-4.077,48.66]},"properties":{"id":81,"title":"SIBIRIL_S_A","description":"Altitude 58 m · test fold 2","altitude_m":58,"test_fold":2,"marker-color":"#3fb950","marker-size":"small","marker-symbol":"2"}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[5.99,44.062]},"properties":{"id":82,"title":"ST_AUBAN","description":"Altitude 458 m · train in all folds","altitude_m":458,"test_fold":null,"marker-color":"#8b949e","marker-size":"small","marker-symbol":""}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[4.643,49.783]},"properties":{"id":83,"title":"CHARLEVILLE-MEZ","description":"Altitude 145 m · test fold 2","altitude_m":145,"test_fold":2,"marker-color":"#3fb950","marker-size":"small","marker-symbol":"2"}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[-0.316,45.665]},"properties":{"id":84,"title":"COGNAC","description":"Altitude 30 m · test fold 3","altitude_m":30,"test_fold":3,"marker-color":"#d29922","marker-size":"small","marker-symbol":"3"}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[5.427,47.044]},"properties":{"id":85,"title":"TAVAUX_SA","description":"Altitude 195 m · train in all folds","altitude_m":195,"test_fold":null,"marker-color":"#8b949e","marker-size":"small","marker-symbol":""}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[2.483,44.41]},"properties":{"id":86,"title":"RODEZ-AVEYRON","description":"Altitude 578 m · train in all folds","altitude_m":578,"test_fold":null,"marker-color":"#8b949e","marker-size":"small","marker-symbol":""}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[0.194,47.946]},"properties":{"id":87,"title":"LE_MANS","description":"Altitude 51 m · train in all folds","altitude_m":51,"test_fold":null,"marker-color":"#8b949e","marker-size":"small","marker-symbol":""}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[2.338,48.822]},"properties":{"id":88,"title":"PARIS-MONTSOURIS","description":"Altitude 75 m · train in all folds","altitude_m":75,"test_fold":null,"marker-color":"#8b949e","marker-size":"small","marker-symbol":""}},
{"type":"Feature","geometry":{"type":"Point","coordinates":[1.866,43.694]},"properties":{"id":89,"title":"LAVAUR","description":"Altitude 131 m · train in all folds","altitude_m":131,"test_fold":null,"marker-color":"#8b949e","marker-size":"small","marker-symbol":""}}
]}
```

> [!TIP]
> The same data is available as [`stations.geojson`](stations.geojson) — GitHub renders it as an interactive map when opened directly, and you can load it in QGIS, `geopandas`, Kepler.gl or geojson.io.

<details>
<summary><b>📋 Full station table (90 rows)</b></summary>

<br>

`—` means the station was used for training in every fold.

| # | Station | Lat | Lon | Alt (m) | Test fold |
|--:|---|--:|--:|--:|:---:|
| 0 | `ST-GERMAIN-L_HE` | 45.461 | 3.533 | 1070 | — |
| 1 | `COLTINES` | 45.075 | 2.991 | 979 | — |
| 2 | `BORDEAUX-MERIGNAC` | 44.831 | -0.691 | 47 | 2 |
| 3 | `AUCH` | 43.689 | 0.601 | 122 | — |
| 4 | `MONTPELLIER-AEROPORT` | 43.576 | 3.965 | 1 | 2 |
| 5 | `BOURGES` | 47.059 | 2.360 | 161 | — |
| 6 | `POITIERS-BIARD` | 46.594 | 0.314 | 125 | 2 |
| 7 | `NANTES-BOUGUENAIS` | 47.150 | -1.609 | 26 | 1 |
| 8 | `BIARRITZ-PAYS-BASQUE` | 43.469 | -1.534 | 71 | 0 |
| 9 | `MARIGNANE` | 43.438 | 5.216 | 9 | — |
| 10 | `NANCY-ESSEY` | 48.688 | 6.221 | 212 | 1 |
| 11 | `BESANCON` | 47.249 | 5.989 | 307 | — |
| 12 | `EPINAL` | 48.211 | 6.452 | 317 | 2 |
| 13 | `ROISSY` | 49.015 | 2.534 | 108 | — |
| 14 | `CLERMONT-FD` | 45.787 | 3.149 | 331 | — |
| 15 | `BOURGANEUF` | 45.942 | 1.726 | 513 | — |
| 16 | `ST_YRIEIX_LA_PE` | 45.514 | 1.244 | 404 | — |
| 17 | `PAU-UZEIN` | 43.385 | -0.416 | 183 | — |
| 18 | `NIMES-COURBESSAC` | 43.857 | 4.406 | 59 | — |
| 19 | `LA_ROCHE_SUR_YON` | 46.705 | -1.382 | 88 | — |
| 20 | `LYON-BRON` | 45.721 | 4.949 | 202 | — |
| 21 | `ST-DIZIER` | 48.631 | 4.904 | 139 | 3 |
| 22 | `TOURS` | 47.444 | 0.727 | 108 | 0 |
| 23 | `BRIVE` | 45.147 | 1.473 | 115 | — |
| 24 | `BEAUCOUZE` | 47.479 | -0.614 | 50 | — |
| 25 | `MONTELIMAR` | 44.581 | 4.733 | 73 | — |
| 26 | `VICHY-CHARMEIL` | 46.167 | 3.399 | 249 | — |
| 27 | `BREST-GUIPAVAS` | 48.450 | -4.383 | 94 | 1 |
| 28 | `BEAUVAIS-TILLE` | 49.446 | 2.127 | 89 | 3 |
| 29 | `CAEN-CARPIQUET` | 49.180 | -0.456 | 67 | 1 |
| 30 | `TOULOUSE-BLAGNAC` | 43.621 | 1.379 | 151 | — |
| 31 | `GRENOBLE-ST_GEOIRS` | 45.364 | 5.313 | 384 | — |
| 32 | `LIMOGES-BELLEGARDE` | 45.861 | 1.175 | 402 | — |
| 33 | `ST-NICOLAS` | 46.051 | 3.805 | 1022 | — |
| 34 | `MACON` | 46.294 | 4.794 | 219 | 1 |
| 35 | `STRASBOURG-ENTZHEIM` | 48.549 | 7.640 | 150 | 0 |
| 36 | `DIJON-LONGVIC` | 47.268 | 5.088 | 219 | — |
| 37 | `TRAPPES` | 48.774 | 2.010 | 167 | — |
| 38 | `LE_PUY-LOUDES` | 45.075 | 3.764 | 833 | — |
| 39 | `ST_BRIEUC` | 48.535 | -2.852 | 135 | 3 |
| 40 | `PERPIGNAN` | 42.737 | 2.873 | 42 | 3 |
| 41 | `ORLY` | 48.718 | 2.397 | 86 | — |
| 42 | `ALENCON` | 48.445 | 0.110 | 143 | — |
| 43 | `AUTUN` | 46.972 | 4.263 | 303 | 3 |
| 44 | `MARCENAT` | 45.303 | 2.833 | 1075 | — |
| 45 | `AURILLAC` | 44.891 | 2.419 | 632 | 3 |
| 46 | `TROYES-BARBEREY` | 48.325 | 4.020 | 112 | 0 |
| 47 | `BLOIS` | 47.679 | 1.212 | 119 | 3 |
| 48 | `EMBRUN` | 44.571 | 6.509 | 873 | 3 |
| 49 | `ALBI` | 43.915 | 2.116 | 172 | 0 |
| 50 | `CHAMBERY-AIX` | 45.641 | 5.878 | 235 | 0 |
| 51 | `CARCASSONNE` | 43.215 | 2.296 | 128 | — |
| 52 | `CHATEAUROUX_DEOLS` | 46.870 | 1.741 | 158 | 1 |
| 53 | `CHARGEY-LES-GRAY` | 47.507 | 5.595 | 244 | — |
| 54 | `DINARD` | 48.585 | -2.076 | 65 | — |
| 55 | `VANNES-SENE` | 47.604 | -2.714 | 3 | — |
| 56 | `MT-ST-VINCENT` | 46.628 | 4.477 | 601 | 2 |
| 57 | `BASTIA` | 42.541 | 9.485 | 10 | 0 |
| 58 | `AJACCIO` | 41.918 | 8.793 | 5 | 1 |
| 59 | `RENNES-ST_JACQUES` | 48.069 | -1.734 | 36 | — |
| 60 | `CARPENTRAS` | 44.084 | 5.058 | 98 | 1 |
| 61 | `ST_QUENTIN` | 49.818 | 3.206 | 98 | 1 |
| 62 | `LE_LUC` | 43.383 | 6.386 | 80 | 0 |
| 63 | `NICE` | 43.649 | 7.209 | 2 | 2 |
| 64 | `AGEN-LA_GARENNE` | 44.172 | 0.595 | 58 | — |
| 65 | `MONTEMBOEUF` | 45.787 | 0.543 | 247 | — |
| 66 | `LANDOS-CHARBON` | 44.858 | 3.844 | 1148 | — |
| 67 | `CALAIS-MARCK` | 50.960 | 1.956 | 2 | 0 |
| 68 | `TARBES-LOURDES-PYRENEES` | 43.188 | 0.000 | 360 | — |
| 69 | `MATHAUX-ETAPE` | 48.353 | 4.472 | 145 | — |
| 70 | `LURCY-LEVIS_SA` | 46.718 | 2.948 | 225 | — |
| 71 | `ST_GIRONS` | 43.005 | 1.107 | 414 | 2 |
| 72 | `MONTAUBAN` | 44.028 | 1.377 | 106 | — |
| 73 | `BERGERAC` | 44.823 | 0.524 | 49 | — |
| 74 | `DAX` | 43.690 | -1.070 | 31 | 3 |
| 75 | `CHARTRES` | 48.460 | 1.501 | 155 | 2 |
| 76 | `QUIMPER` | 47.973 | -4.161 | 82 | 0 |
| 77 | `CAMPISTROUS` | 43.127 | 0.366 | 593 | 1 |
| 78 | `MILLAU` | 44.118 | 3.020 | 712 | — |
| 79 | `CHATILLON_SEINE` | 47.849 | 4.581 | 262 | — |
| 80 | `MELUN` | 48.610 | 2.680 | 91 | — |
| 81 | `SIBIRIL_S_A` | 48.660 | -4.077 | 58 | 2 |
| 82 | `ST_AUBAN` | 44.062 | 5.990 | 458 | — |
| 83 | `CHARLEVILLE-MEZ` | 49.783 | 4.643 | 145 | 2 |
| 84 | `COGNAC` | 45.665 | -0.316 | 30 | 3 |
| 85 | `TAVAUX_SA` | 47.044 | 5.427 | 195 | — |
| 86 | `RODEZ-AVEYRON` | 44.410 | 2.483 | 578 | — |
| 87 | `LE_MANS` | 47.946 | 0.194 | 51 | — |
| 88 | `PARIS-MONTSOURIS` | 48.822 | 2.338 | 75 | — |
| 89 | `LAVAUR` | 43.694 | 1.866 | 131 | — |

</details>

## 📁 Repository layout

```text
FranceSolFlow/
├── data/
│   └── stations.geojson        # station metadata + fold assignment
├── docs/assets/                # screenshots / GIF for this README
├── streamlit_app/
│   ├── app.py                  # Streamlit application
│   └── checkpoints/            # 4 trained fold checkpoints
└── README.md
```

<!-- TODO: adjust the tree to match the actual repo (training code folder, etc.) -->

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

<!-- TODO: add LICENSE section / badge once a license is chosen. -->

<div align="center">
<sub>Built for storage sizing, ramp-rate studies, and anyone who needs solar days that behave like real ones. ☀️</sub>
</div>
