# Bike Sharing Demand Forecasting & Fleet Rebalancing Strategy

An end-to-end Machine Learning and Business Intelligence solution in Python predicting hourly bike rental demand (`cnt`) to optimize fleet allocation, prevent station stockouts, and streamline logistics.

---

## 👤 Author & Project Metadata

- **Full Name:** KADITHI SUSMITHA
- **Email:** [susmithakadithi93@gmail.com](mailto:susmithakadithi93@gmail.com)
- **GitHub Username:** [susmithakadithi93-art](https://github.com/susmithakadithi93-art)
- **Degree & Stream:** Master of Computer Applications (MCA)
- **Institution:** Kakinada Institute of Engineering and Technology
- **Primary Deliverable Files:**
  - Code Notebook: [`notebooks/KADITHI_SUSMITHA_BikeSharingDemand.ipynb`](file:///D:/Data%20analytics/Susi/bike-sharing-demand/notebooks/KADITHI_SUSMITHA_BikeSharingDemand.ipynb)
  - Project Report: [`docs/KADITHI_SUSMITHA_ProjectReport.docx`](file:///D:/Data%20analytics/Susi/bike-sharing-demand/docs/KADITHI_SUSMITHA_ProjectReport.docx)
  - Training Pipeline Script: [`src/train_model.py`](file:///D:/Data%20analytics/Susi/bike-sharing-demand/src/train_model.py)
  - Serialized Model: [`models/demand_model.pkl`](file:///D:/Data%20analytics/Susi/bike-sharing-demand/models/demand_model.pkl)
  - Hourly Forecast & Action Plan: [`results/hourly_forecast.csv`](file:///D:/Data%20analytics/Susi/bike-sharing-demand/results/hourly_forecast.csv)
  - Requirements File: [`requirements.txt`](file:///D:/Data%20analytics/Susi/bike-sharing-demand/requirements.txt)

---

## 📌 Project Overview

Urban bike-sharing systems rely on real-time vehicle availability. When docking stations run out of bikes during morning rush hours, commuters experience delays and operators suffer revenue loss. Conversely, leaving unallocated bikes at low-demand stations leads to high maintenance costs and idle capital.

This project formulates an applied machine learning pipeline predicting hourly rental count (`cnt`). Rather than treating prediction purely as an algorithmic benchmarking exercise, this project directly maps forecasts into an **hour-by-hour fleet rebalancing dispatch matrix** (`results/hourly_forecast.csv`).

---

## 🎯 Problem Statement

Bike-sharing demand is driven by extreme spatial-temporal asymmetry:
- **Commute Rushes:** Massive demand spikes occur during 08:00 AM and 17:00–18:00 PM workday commute windows.
- **Off-Peak Lulls:** Demand drops near zero during late-night hours.
- **Weather Vulnerability:** Rain, snow, and low temperatures drastically suppress rental activity.

Without machine learning forecasts, operators struggle to schedule rebalancing trucks efficiently, leading to stockouts at transit hubs and idle bikes across peripheral stations.

---

## 🚩 Objectives

1. **Predictive Modeling:** Build and train regression models to forecast total hourly bike rentals (`cnt`).
2. **Data Leakage Elimination:** Identify and drop target component features (`casual` and `registered`) that cause artificial overfitting.
3. **Chronological Validation:** Evaluate models using an 80/20 chronological split to simulate true out-of-time forecasting conditions.
4. **Model Benchmarking:** Compare Linear Regression, Ridge Regression, Random Forest, and Gradient Boosting across MAE, RMSE, and R² metrics.
5. **Operational Deliverable:** Generate a practical fleet rebalancing schedule mapping predicted demand to truck dispatch categories.

---

## 📊 Dataset Information

- **Dataset Name:** UCI Bike Sharing Dataset (Hourly)
- **Observations:** 17,379 hourly records (2011–2012)
- **Features:** 17 attributes detailing temporal indices, weather conditions, temperature, humidity, wind speed, user categories, and total rental count.
- **UCI Repository Link:** [https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)
- **Direct Official Dataset Zip Source:** [https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip](https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip)

> **Dataset Download Note:** The initial raw GitHub shortcut URL returned a 404 error during setup. To ensure full automated execution without manual steps, the project script automatically downloads the official UCI repository archive zip (`bike+sharing+dataset.zip`) directly from UCI servers and extracts `hour.csv` into `data/bike_sharing_hour.csv`.

---

## 🛠️ Technologies Used

- **Language:** Python 3.11 / 3.14
- **Data Manipulation & Analytics:** `pandas`, `numpy`
- **Machine Learning & Pipelines:** `scikit-learn`, `joblib`
- **Visualization:** `matplotlib`, `seaborn`
- **Document Generation:** `python-docx`, `jupyter`, `nbconvert`

---

## 📁 Project Structure

```
bike-sharing-demand/
├── data/
│   └── bike_sharing_hour.csv             # Raw UCI hourly dataset (17,379 rows)
├── notebooks/
│   ├── KADITHI_SUSMITHA_BikeSharingDemand.ipynb  # Fully executed notebook with embedded outputs
│   └── build_notebook.py                 # Notebook generation script
├── src/
│   └── train_model.py                    # Standalone model training & evaluation script
├── models/
│   └── demand_model.pkl                  # Serialized Scikit-learn Gradient Boosting pipeline
├── results/
│   ├── figures/                          # High-res chart figures (PNG)
│   │   ├── fig1_hourly_demand_patterns.png
│   │   ├── fig2_seasonal_weather_demand.png
│   │   ├── fig3_model_actual_vs_predicted.png
│   │   ├── fig5_residuals_distribution.png
│   │   └── fig6_rebalancing_schedule_heatmap.png
│   └── hourly_forecast.csv               # Managerial forecast & fleet rebalancing schedule
├── docs/
│   ├── KADITHI_SUSMITHA_ProjectReport.docx # 15-section DOCX project report with embedded charts
│   └── build_report_docx.py              # DOCX report generation script
├── requirements.txt                      # Verified dependencies file
├── .gitignore                            # Environment & temporary build ignores
└── README.md                             # Project documentation
```

---

## ⚙️ Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/susmithakadithi93-art/bike-sharing-demand.git
   cd bike-sharing-demand
   ```

2. **Create & Activate Virtual Environment:**
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 How to Run

1. **Execute Model Training Script:**
   ```bash
   python src/train_model.py
   ```
   This script ingests data, engineers features, trains all regressors, serializes `models/demand_model.pkl`, exports `results/hourly_forecast.csv`, and generates all chart visuals.

2. **Execute Jupyter Notebook End-to-End:**
   ```bash
   python -m nbconvert --to notebook --execute --inplace notebooks/KADITHI_SUSMITHA_BikeSharingDemand.ipynb
   ```

3. **Generate DOCX Project Report:**
   ```bash
   python docs/build_report_docx.py
   ```

---

## 📈 Key Performance Indicators (KPIs)

These three measured metrics frame the operational problem:

1. **Average Hourly Demand:** `189.46 bikes/hour` baseline demand across 17,379 dataset hours (total volume: `3,292,679 bikes`).
2. **Peak-to-Off-Peak Load Volatility Ratio:** `39.35x` ratio between peak commute demand (`461.45 bikes/hr` at 17:00 PM) and off-peak demand (`11.73 bikes/hr` at 03:00 AM).
3. **Adverse Weather Sensitivity Penalty:** `45.54%` rental volume drop during adverse weather (Light Rain/Snow: `111.58 bikes/hr`) compared to clear weather (`204.87 bikes/hr`).

---

## 🔬 Model Benchmarking & Evaluation Results

All models were evaluated on the independent chronological test set (**3,476 test hours** from Aug 7, 2012 to Dec 31, 2012):

| Regressor Model | MAE (bikes/hr) | RMSE (bikes/hr) | R² Score | Selection Status |
|---|---|---|---|---|
| **Linear Regression (Baseline)** | 122.243 | 164.542 | 0.4431 | Benchmark |
| **Ridge Regression (alpha=10.0)** | 122.198 | 164.560 | 0.4430 | Baseline |
| **Random Forest Regressor** | 53.784 | 81.021 | 0.8650 | Strong Candidate |
| **Gradient Boosting Regressor** | **49.062** | **74.044** | **0.8872** | **SELECTED BEST MODEL** |

> **Metric Rationale:** MAE provides clear physical units (bikes/hour) easily understood by operations teams. RMSE penalizes peak-hour miss errors heavily to prevent severe station stockouts. R² measures total variance explained in unseen out-of-time demand.

---

## 💡 Target Leakage Prevention Note

In the raw dataset, `casual` (unregistered users) and `registered` (subscribed users) sum **exactly** to `cnt` ($casual + registered = cnt$). 

Including `casual` and `registered` in the predictor matrix causes the model to learn a simple summation identity, producing a deceptive R² of ~1.0. In real-world deployment, casual and registered breakdowns are unknown before the hour completes. **Dropping `casual`, `registered`, and `instant` is essential to prevent severe target leakage.**

---

## 🎯 Risk, Opportunity & Operational Action Plan

- **Supply Deficit Risk:** Fleet exhaustion during 08:00 AM and 17:00–18:00 PM rush hours causes customer churn and lost revenue.
- **Unmet Surge Opportunity:** Warm summer weekend afternoons present high demand capacity in recreational zones.
- **Idle Asset Risk:** Rainy/cold weather leads to idle bikes; logistics dispatches should be scaled back during adverse weather.

### Operational Fleet Rebalancing Action Plan
Model predictions in `results/hourly_forecast.csv` are dynamically mapped to four action tiers across 3,476 test hours:

- **`CRITICAL_REBALANCE_HIGH` ($\ge 400$ bikes/hr):** 576 hours (16.6%). Dispatch rebalancing trucks 90 minutes pre-rush hour.
- **`MODERATE_REBALANCE` ($250–399$ bikes/hr):** 779 hours (22.4%). Monitor hub inventory.
- **`STANDARD_MONITOR` ($31–249$ bikes/hr):** 1,480 hours (42.6%). Standard routine monitoring.
- **`LOW_FLEET_IDLE` ($\le 30$ bikes/hr):** 641 hours (18.4%). High idle capacity; schedule bike maintenance windows.

---

## ⚠️ Limitations

1. **City-Wide Aggregation:** The dataset contains city-aggregate hourly totals rather than individual dock station GPS coordinates.
2. **Unobserved Events:** Special urban events (concerts, sports games, transit strikes) are not recorded in historical features.
3. **Fixed Fleet Size:** The dataset assumes a static overall bike fleet across 2011–2012.

---

## 🔮 Future Enhancements

- **Station-Level Micro-Routing:** Extend predictions to individual dock stations to generate optimal truck travel routes.
- **Real-Time Telemetry Integration:** Stream live weather forecasts via OpenWeather API.
- **Deep Learning Architectures:** Benchmark Temporal Fusion Transformers (TFT) and LSTM networks for multi-step horizon forecasting.

---

## 📜 License & Citation

Dataset provided by UCI Machine Learning Repository:
> Fanaee-T, H., & Gama, J. (2014). Event labeling combining ensemble detectors and background knowledge. *Progress in Artificial Intelligence*, 2(2), 113-127.
