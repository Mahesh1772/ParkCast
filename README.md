# 🚗 Singapore Carpark Availability Forecasting
### Weather-Aware IoT Data Science | EE4211 Data Science for IoT — NUS Group 21

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Best%20Model-orange?style=flat)](https://xgboost.readthedocs.io/)
[![MAPE](https://img.shields.io/badge/MAPE-1.28%25-brightgreen?style=flat)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Overview

This project investigates **how weather conditions influence carpark availability patterns across Singapore**, and whether integrating weather data can meaningfully improve short-term forecasting accuracy. Using real-world IoT sensor data from ~1,975 carparks island-wide, we built, benchmarked, and evaluated **8 distinct forecasting models** — from simple baselines to deep learning — on 1,488 hourly observations (July–August 2024).

**Best Result:** XGBoost Regressor — **MAPE 1.28%**, RMSE 0.0179 — outperforming the persistence baseline by **22.8%**.

---

## 🗂️ Repository Structure

```
├── EE4211_Project_Q2.ipynb           # Question 2: Forecasting (Ridge, RF, MLP + hyperparameter tuning)
├── EE4211_Project_Q3.ipynb           # Question 3: Group Project (XGBoost, LightGBM, LSTM, TCN, SARIMA/X)
├── data/
│   ├── carpark_july2024.csv          # Training set (744 hourly observations)
│   ├── carpark_august2024.csv        # Test set (744 hourly observations)
│   ├── weather_july2024.csv          # NEA weather data — July 2024
│   └── weather_august2024.csv        # NEA weather data — August 2024
├── models/
│   └── final_model_object.pkl        # Serialised best model (XGBoost) — callable via .predict(X)
├── reports/
│   ├── EE4211-Data-Science-for-IoT-Group-Project-Report.pdf
│   └── Group21_EE4211_Project_Notebook_Print_Preview.pdf
└── README.md
```

---

## 🔬 Research Questions

1. How strongly do weather variables (rainfall, temperature, humidity, wind speed) affect system-wide carpark demand at different times of day?
2. Can weather-aware ML models improve short-term (1-hour ahead) forecasts versus baseline time-series models?
3. Which modelling approaches (statistical, gradient boosting, deep learning) best capture weather–parking relationships?
4. What is the relative importance of weather features vs. temporal and lag features?

---

## 📦 Data Sources

| Source | Coverage | Variables |
|---|---|---|
| [Singapore Gov Carpark API](https://data.gov.sg/datasets/dca933a6-44e5-5d34-fe21-f28b8052fac6/view) | Jul–Aug 2024, ~1,975 carparks | `carpark_number`, `lots_available`, `total_lots`, `timestamp` |
| [NEA Weather (data.gov.sg)](https://data.gov.sg) | Jul–Aug 2024, hourly | `temperature (°C)`, `humidity (%)`, `wind_speed (m/s)`, `rainfall (mm)` |

**Key Data Stats:**
- ~1,975 unique carparks tracked (island-wide HDB + LTA)
- 744 hourly observations per month (July train / August test)
- Missing hours imputed: 42 (July, 5.6%) and 32 (August, 4.3%) via **forward-fill**
- Mean availability ratio: 57.3% (July) / 57.0% (August)

---

## 🧹 Data Preprocessing & IoT Quality Analysis

A systematic sensor quality audit was performed across the full carpark network, identifying **87 malfunctioning sensors** out of ~1,975 carparks:

| Fault Type | Count | Detection Method |
|---|---|---|
| Frozen sensors (zero variance) | 72 | Variance < 0.0001 threshold |
| Always empty (≥ 99% availability) | 9 | Min availability > 0.99 |
| Always full (≤ 1% availability) | 6 | Max availability < 0.01 |

Additional preprocessing steps:
- API update frequency characterised empirically: **~68-second refresh cycle** (not per-second as the timestamp parameter implies)
- Unique carpark count verified stable at **1,975** within a 10-minute window
- Timestamps localised to `Asia/Singapore` (UTC+8)
- Hourly aggregation: system-wide `availability_ratio = sum(lots_available) / sum(total_lots)`

---

## ⚙️ Feature Engineering

27 predictive features engineered across 5 categories:

| Category | Features |
|---|---|
| **Calendar / Temporal** | `hour_of_day`, `day_of_week`, `day_of_month`, `is_weekend`, `is_public_holiday`, `is_peak_hour` |
| **Cyclical Encoding** | `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos` |
| **Lag Features** | `lag_1h`, `lag_2h`, `lag_24h` (daily), `lag_168h` (weekly) |
| **Rolling Window** | `rolling_24h_mean`, `rolling_24h_std`, `rolling_168h_mean` |
| **Weather** | `temperature`, `humidity`, `wind_speed`, `rainfall_mm`, `is_rain`, `heavy_rain`, `is_hot`, `is_humid` |

**Top Pearson correlations with target (`availability_ratio`):**

| Feature | Correlation |
|---|---|
| temperature | +0.536 |
| humidity | −0.515 |
| wind_speed | +0.512 |
| rainfall_mm | +0.124 (weak linear, nonlinear captured by XGBoost) |

> Weather sensitivity peaks during late morning (09:00–12:00), with the highest single-feature correlation reaching **r = 0.209 at 11:00**.

---

## 🤖 Models Benchmarked

| Model | RMSE | MAE | MAPE |
|---|---|---|---|
| **XGBoost** ⭐ | **0.0179** | **0.0073** | **1.28%** |
| LightGBM | 0.0185 | 0.0074 | 1.30% |
| Persistence (Baseline) | 0.0232 | 0.0157 | 2.75% |
| SARIMA | 0.0333 | 0.0232 | 4.00% |
| SARIMAX | 0.0327 | 0.0227 | 3.91% |
| LSTM | 0.0405 | 0.0325 | 5.71% |
| TCN | 0.0625 | 0.0490 | 8.11% |
| Ridge Regression | — | — | — |

**Key Findings:**
- XGBoost achieves **22.8% lower RMSE** than the persistence baseline and **3.2% lower RMSE** than LightGBM
- Gradient boosting dominates — both XGBoost and LightGBM significantly outperform all other approaches
- Deep learning (LSTM, TCN) underperforms due to limited training data (576 samples after feature engineering)
- SARIMAX paradoxically underperforms SARIMA, suggesting linear frameworks cannot capture weather–parking nonlinearities

---

## 🛠️ Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn xgboost lightgbm statsmodels tensorflow matplotlib seaborn requests pickle5
```

### Run Forecasting Notebook (Q2)

```bash
jupyter notebook EE4211_Project_Q2.ipynb
```

Covers: Ridge Regression, Random Forest, MLP with hyperparameter tuning via `RandomizedSearchCV`.  
Final trained model saved as `final_model_object.pkl`.

### Load & Use the Final Model

```python
import pickle
import pandas as pd

with open("models/final_model_object.pkl", "rb") as f:
    final_model = pickle.load(f)

# X must contain the same 15 engineered features used during training
predictions = final_model.predict(X)
```

---

## 📊 Evaluation Metrics

Three complementary metrics used:

- **RMSE** — Root Mean Squared Error (penalises large errors)
- **MAE** — Mean Absolute Error (robust to outliers)
- **MAPE** — Mean Absolute Percentage Error (interpretable, scale-free)

**Validation strategy:** Strict temporal split — train on July 2024, test on August 2024. No data leakage. Random seed fixed at `42` for reproducibility.

---

## 🌏 Practical Applications

| Stakeholder | Application |
|---|---|
| **Drivers** | Weather-aware parking predictions integrated into navigation apps to reduce search time |
| **LTA / URA** | Proactive traffic management and infrastructure planning for weather-sensitive zones |
| **Parking Operators** | Dynamic pricing calibrated to weather-predicted demand shifts |
| **Smart Nation** | Evidence-based IoT analytics demonstrating real-time urban sensing at scale |

---

## ⚠️ Limitations

- Two-month temporal scope (July–August 2024) may not capture full seasonal variation
- Single central weather station — does not reflect Singapore's microclimate variability
- System-wide aggregation masks localised carpark-level weather sensitivity
- Dataset does not account for special events (concerts, National Day, etc.)

---

## 📄 Report & Slides

Full project report and methodology:  
`reports/EE4211-Data-Science-for-IoT-Group-Project-Report.pdf`

---

## 👥 Team

**Group 21 — EE4211 Data Science for IoT, NUS AY2025/26**

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🔗 References

1. Fabusuyi et al. (2014). Decision Analytics for Parking Availability in Downtown Pittsburgh. *Interfaces*, 44(3), 286–299.
2. Fokker et al. (2021). Short-Term Forecasting of Off-Street Parking Occupancy. *Transportation Research Record*, 2676(1), 637–654.
3. Zhang et al. (2024). Predicting Parking Availability in Singapore with Cross-Domain Data. [arXiv:2405.18910](https://arxiv.org/abs/2405.18910)
4. Singapore Government Open Data: [data.gov.sg](https://data.gov.sg)
