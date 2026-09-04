# RV Retail Forecast Dashboard

A Streamlit dashboard for forecasting RV retail sales by market, division, and RV type. It also supports backtesting, estimated Dometic-unit projections, customer ranking, and Excel exports.

## Setup

```bash
# 1. Create virtual environment
python -m venv .venv

# 2. Activate virtual environment
# On macOS / Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env with your local data paths and credentials
```

> [!NOTE]
> On macOS, `pymssql` and `lightgbm` may require system libraries (`brew install freetds libomp`).

## Pipeline Execution

1. **(Optional) Run data quality audit**:
   ```bash
   python data_quality_check.py
   ```

2. **Generate the retail market forecast**:
   ```bash
   python run_forecast.py
   ```

3. **(Optional) Generate Dometic sales attach-rate projections**:
   ```bash
   # Pull fresh OEM sales from D365 (requires SQL credentials in .env)
   python pull_dometic_sales.py
   
   # Project Dometic units onto retail market forecast
   python compute_attach_rate_forecast.py
   ```

4. **Launch the dashboard**:
   ```bash
   python -m streamlit run retail_forecast_dashboard.py
   ```