# 📈 RV Retail Demand Forecast & OEM Sales Intelligence Platform

An enterprise-grade, multi-model time-series forecasting and sales analytics platform for the RV industry. It predicts retail market demand across North American RV manufacturers, segments, and product categories, evaluates models via rolling-origin backtesting, performs hierarchical WLS reconciliation, and projects OEM component demand using attach-rate analytics.

Built with **Python**, **LightGBM / XGBoost / Statsmodels / Prophet**, **Plotly**, **Streamlit**, and ready for **Power BI embedding**.

---

## 🌟 Key Features & Capabilities

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                SYSTEM ARCHITECTURE                                     │
├──────────────────────────┬─────────────────────────┬───────────────────────────────────┤
│ 1. RV Retail History     │ 2. FRED Macro Signals   │ 3. D365 OEM Sales & Master        │
│    (RV_Cust_Data.xlsx)   │    (Fed St. Louis API)  │    (Dynamics 365 SQL Server)      │
└─────────────┬────────────┴────────────┬────────────┴─────────────────┬─────────────────┘
              │                         │                              │
              ▼                         ▼                              ▼
┌──────────────────────────────────────────────┐       ┌─────────────────────────────────┐
│           ML FORECAST ENGINE                 │       │      ATTACH RATE ENGINE         │
│         (retail_forecast_engine.py)          │       │(compute_attach_rate_forecast.py)│
│  • Tiered model routing (Intermittent/ML)    │       │  • Division attach rates        │
│  • 5% Naive-beat baseline gating             │       │  • Product area splits          │
│  • Two-stage WLS hierarchical reconciliation │       │  • Upsell candidate discovery   │
│  • Empirical P10/P90 uncertainty intervals   │       │  • Peer segment benchmarking    │
└──────────────────────┬───────────────────────┘       └────────────────┬────────────────┘
                       │                                                │
                       ▼                                                ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        COMPRESSED PARQUET DATA LAYER (`parquet/`)                      │
│   • forecast.parquet   • history.parquet   • backtest.parquet   • attach_rates.parquet │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                   INTERACTIVE STREAMLIT DASHBOARD & POWER BI EMBED                     │
│                            (retail_forecast_dashboard.py)                              │
│   [📊 Executive Overview] [🔍 Series Explorer] [🏆 Rankings] [🎯 OEM Attach] [⚙️ Governance]│
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### 1. 📊 Interactive Dashboard Tabs
* **Executive Overview**: High-level market trajectory, top KPI cards (Total Demand, Tested Model Accuracy, Median Attach Rate), and automated narrative takeaways.
* **Series Explorer**: Interactive deep-dive by Manufacturer (`Grand Design`, `Jayco`, `Forest River`, `Keystone`, etc.) or RV Category (`Travel Trailer`, `Fifth Wheel`, `Class A/C`) with toggleable P10–P90 confidence ranges and historical model fit overlays.
* **Manufacturer Rankings**: Dynamic Top 5/10/25 horizontal bar charts with YoY growth calculations and sortable summaries.
* **Dometic OEM Projections**: Component demand forecasting based on manufacturer retail sales and trailing attach rates, broken down by category (Climate, Awnings, Sanitation, Refrigeration).
* **Model Governance & Diagnostics**: Full transparency into algorithm win counts, out-of-sample backtest wMAPE, and validation metrics.
* **Interactive Demo Mode**: Boots seamlessly with realistic sample data even on fresh deployments before production parquet files are generated.

---

## 🧠 Forecasting Methodology & Best Practices

The forecasting engine (`retail_forecast_engine.py`) adheres to modern time-series forecasting principles (M4/M5 competition standards):

1. **Volume & Sparsity-Aware Tiered Routing**:
   * **Intermittent Demand** ($>40\%$ zero months) $\rightarrow$ **TSB (Teunter-Syntetos-Babai)** & **Croston** algorithms.
   * **Thin Series** ($<12$ months / $<24$ units) $\rightarrow$ **Seasonal-Naive** baseline to prevent overfitting.
   * **Medium Series** ($12–24$ months) $\rightarrow$ **ETS**, **XGBoost**, **LightGBM**, and **SARIMAX**.
   * **Full Series** ($24+$ months, $200+$ units) $\rightarrow$ Full roster including **AutoARIMA**, **Prophet**, and optional foundation models.
2. **5% Naive Beat Gate (`NAIVE_BEAT_MARGIN = 0.05`)**:
   * A competing ML model must beat Seasonal-Naive by at least **5% relative wMAPE** to be deployed. Prevents deploying complex models that merely fit small-sample noise.
3. **WLS Hierarchical Reconciliation**:
   * Blends bottom-up $\text{Division} \times \text{Type}$ child sums with direct parent forecasts using **inverse-variance weighting**, then anchors to Total Market's forecast via trailing 12-month share weights.
4. **Empirical Prediction Intervals (P10 / P90)**:
   * Confidence intervals are constructed by pooling backtest percentage errors across all candidate models (conformal prediction spirit), accounting for both data variance and model selection risk.
5. **Real-Data Backtest Scoring (`min_real_date`)**:
   * Models train on full history (including RVIA-redistributed years) but are strictly **scored on genuine transactional months** to prevent models from gaming synthetic seasonal curves.

---

## 🛠️ Quickstart & Local Setup

### Prerequisites
* Python 3.9+ installed
* (macOS only) System libraries for FreeTDS and LightGBM:
  ```bash
  brew install freetds libomp
  ```

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/knmurug2/retail_forecast_dashboard.git
cd retail_forecast_dashboard

# Create and activate virtual environment
python -m venv .venv

# On macOS / Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy the template and set your local paths or credentials:
```bash
cp .env.example .env
```
Key configuration options inside `.env`:
```env
DATA_PATH=./data/RV_Cust_Data.xlsx
PARQUET_DIR=./parquet
FRED_API_KEY=your_fred_api_key_here
OWN_DIVISIONS=Grand Design, Jayco
```

---

## 🚀 Running the Pipeline

### 1. (Optional) Run Data Quality Audit
Scans the source Excel dataset for structure errors, unparseable dates, negative units, and duplicate transactions:
```bash
python data_quality_check.py
```

### 2. Run the ML Forecast Pipeline
Executes rolling backtests, model selection, hierarchical reconciliation, and outputs compressed parquet files:
```bash
python run_forecast.py
```
*Optional Flags*:
* `--full`: Runs all models including slower iterative solvers (AutoARIMA/Prophet).
* `--grains Total Division Type`: Restricts forecasting to specific grains.
* `--heavy`: Enables TimesFM / Chronos foundation models (if installed).

### 3. (Optional) Run OEM Attach Rate Projections
Pulls D365 sales data and calculates manufacturer attach rates:
```bash
# Pull fresh OEM sales from Dynamics 365 (requires SQL credentials in .env)
python pull_dometic_sales.py

# Project component demand onto the market forecast
python compute_attach_rate_forecast.py
```

### 4. Launch the Interactive Dashboard
```bash
python -m streamlit run retail_forecast_dashboard.py
```

---

## 🌐 Free Cloud Deployment (Streamlit Community Cloud)

1. Fork or push this repository to GitHub.
2. Go to **[share.streamlit.io](https://share.streamlit.io)** and log in with GitHub.
3. Click **New app** $\rightarrow$ Select `retail_forecast_dashboard` $\rightarrow$ Branch `main` $\rightarrow$ Main file `retail_forecast_dashboard.py`.
4. Click **Deploy!**

---

## 📊 Embedding in Power BI (100% Free)

You can embed this interactive Streamlit dashboard directly inside **Power BI Desktop** or **Power BI Service**:

1. In Power BI Desktop, go to **Visualizations** $\rightarrow$ **Get more visuals** $\rightarrow$ Search and add **"HTML Content"** (by Daniel Marsh-Patrick, free & certified).
2. Create a DAX measure referencing your hosted app with `?embed=true`:
   ```dax
   Streamlit_Embed = 
   "<iframe src='https://YOUR-APP-NAME.streamlit.app/?embed=true' width='100%' height='850px' frameborder='0' style='border-radius:8px;'></iframe>"
   ```
3. Drop `Streamlit_Embed` into the HTML Content visual on your canvas.

---

## 📁 Repository Structure

```
├── retail_forecast_dashboard.py    # Main interactive Streamlit application
├── retail_forecast_engine.py       # Core time-series forecasting & reconciliation engine
├── run_forecast.py                 # CLI runner for ML pipeline & parquet export
├── compute_attach_rate_forecast.py # Attach rate calculation & Dometic sales projection
├── pull_dometic_sales.py           # D365 SQL Server sales & customer master extraction
├── data_quality_check.py           # Pre-flight data validation & QA report generator
├── requirements.txt                # Python package dependencies
├── .env.example                    # Environment variable configuration template
├── .gitignore                      # Git exclusion rules (protects credentials & temp files)
└── README.md                       # Comprehensive documentation
```

---

## 📄 License
Internal proprietary analytics and forecasting tool.
