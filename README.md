# RV Retail Forecast Dashboard

A Streamlit dashboard for forecasting RV retail sales by market, division, and RV type. It also supports backtesting, estimated Dometic-unit projections, customer ranking, and Excel exports.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Configure the data-source settings in a local `.env` file. Do not commit credentials or generated forecast files.

## Run

Generate the forecast data first:

```bash
python run_forecast.py
```

Then start the dashboard:

```bash
python -m streamlit run retail_forecast_dashboard.py
```

Optional Dometic projections require running `compute_attach_rate_forecast.py` after the sales data is available.