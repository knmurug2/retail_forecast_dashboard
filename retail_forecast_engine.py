"""
retail_forecast_engine.py  (v2.1)
==================================
Multi-model demand forecasting engine for Dometic RV Retail Sales.

v2.1: RVIA-driven seasonal index (2024-2025, weighted) replaces hardcoded
      values for redistributing annual rollup rows into monthly values.

v2 features:
  1. Tiered model selection  -- thin/sparse series skip complex models
  2. Intermittent demand     -- TSB + Croston for >40% zero-month series
  3. Rolling backtest        -- 3 origins for robust wMAPE estimation
  4. FRED feature gating     -- exog only on series with enough history
  5. Hierarchical reconcil.  -- bottom-up: Division x Type sums to Division & Type
  6. Market share framing    -- own vs total market if OWN_DIVISIONS configured
  7. .env credential loading -- all paths/keys from environment
"""

import os
import sys
import json
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def _load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
    except ImportError:
        pass

_load_env()

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
def _resolve_path(env_key: str, default_win: str, default_rel: str) -> str:
    if env_key in os.environ and os.environ[env_key]:
        return os.environ[env_key]
    if os.path.exists(default_win):
        return default_win
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, default_rel)

DATA_PATH   = _resolve_path("DATA_PATH",
              r"C:\Users\Karmur\OneDrive - Dometic Group\RV_Cust_Data.xlsx",
              "data/RV_Cust_Data.xlsx")
PARQUET_DIR = _resolve_path("PARQUET_DIR",
              r"C:\Users\Karmur\OneDrive - Dometic Group\PY Project Output\RetailForecast\parquet",
              "parquet")
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")
OWN_DIVISIONS = [d.strip() for d in os.environ.get("OWN_DIVISIONS", "").split(",") if d.strip()]

FORECAST_MONTHS = 12
ROLLING_ORIGINS = int(os.environ.get("ROLLING_ORIGINS", 3))

TIER_FULL_MIN_MONTHS   = 24
TIER_FULL_MIN_VOLUME   = 200
TIER_MEDIUM_MIN_MONTHS = 12
TIER_MEDIUM_MIN_VOLUME = 24
INTERMITTENT_ZERO_FRAC = 0.40
MIN_SERIES_FOR_EXOG    = 30

# Best-practice additions:
# A competing model must beat seasonal-naive by at least this relative margin
# in backtest wMAPE to be used -- otherwise naive is selected. Protects against
# a "winning" model that's really just noise-fitting a small backtest sample.
NAIVE_BEAT_MARGIN = 0.05

# Months reserved as a genuine held-out validation window: never touched during
# model selection or backtest scoring, only used afterward to report an honest
# out-of-sample accuracy number. Requires enough history to spare 3 months
# and still have a meaningful backtest on what's left.
VALIDATION_MONTHS = 3
MIN_HISTORY_FOR_VALIDATION = TIER_MEDIUM_MIN_MONTHS + VALIDATION_MONTHS  # 15 months

# The redistributed annual-rollup years (see redistribute_annual_to_monthly) use
# the SAME fixed seasonal curve every year, so those years carry zero genuine
# month-to-month signal -- seasonal-naive is nearly unbeatable on them by
# construction, which biases normal backtesting toward naive regardless of
# whether a model would actually add value on real demand data. When enough
# real (non-redistributed) transactional months exist, backtest SCORING is
# restricted to just those months -- an honest test of real monthly variation
# -- while model FITTING still uses the full history including synthetic years.
MIN_REAL_TEST_POINTS = 3

GRAINS = ["Total", "Division", "Type", "Division_Type"]

FRED_SERIES = {
    "UMCSENT": "Consumer Sentiment", "GASREGW": "Gas Price",
    "FEDFUNDS": "Fed Funds Rate", "MORTGAGE30US": "30yr Mortgage Rate",
    "CSUSHPISA": "Home Price Index",
}

RVIA_PATH = os.environ.get("RVIA_PATH",
            r"C:\Users\Karmur\OneDrive - Dometic Group\Retail OEM RV shipment.xlsx")

_rvia_weights_env = os.environ.get("RVIA_YEAR_WEIGHTS", "")
if _rvia_weights_env:
    RVIA_YEAR_WEIGHTS = {}
    for pair in _rvia_weights_env.split(","):
        yr, wt = pair.split(":")
        RVIA_YEAR_WEIGHTS[int(yr.strip())] = float(wt.strip())
else:
    RVIA_YEAR_WEIGHTS = {2024: 0.35, 2025: 0.65}

RANDOM_STATE = 42


class Tier:
    INTERMITTENT = "INTERMITTENT"
    THIN         = "THIN"
    MEDIUM       = "MEDIUM"
    FULL         = "FULL"


# ---------------------------------------------------------------------------
# RVIA SEASONAL INDEX
# ---------------------------------------------------------------------------
RV_SEASONAL_INDEX_FALLBACK = {
    1: 0.050, 2: 0.058, 3: 0.092, 4: 0.112, 5: 0.110, 6: 0.098,
    7: 0.088, 8: 0.090, 9: 0.088, 10: 0.078, 11: 0.068, 12: 0.068,
}

_rvia_index_cache = None


def load_rvia_seasonal_index(rvia_path=None, year_weights=None, use_cache=True):
    global _rvia_index_cache
    if use_cache and _rvia_index_cache is not None:
        return _rvia_index_cache

    path    = rvia_path or RVIA_PATH
    weights = year_weights or RVIA_YEAR_WEIGHTS

    try:
        df = pd.read_excel(path, engine="openpyxl")
        df.columns = [str(c).strip() for c in df.columns]

        month_col  = next((c for c in df.columns if "month" in c.lower()), None)
        retail_col = next((c for c in df.columns
                           if "retail" in c.lower() and "sales" in c.lower()), None)
        if not month_col or not retail_col:
            raise ValueError(f"Columns not found: {df.columns.tolist()}")

        df = df[[month_col, retail_col]].copy()
        df.columns = ["Month", "Retail_Sales"]
        df["Month"] = pd.to_datetime(df["Month"], errors="coerce")
        df["Retail_Sales"] = pd.to_numeric(
            df["Retail_Sales"].astype(str).str.replace(",", "").str.strip(), errors="coerce")
        df = df.dropna(subset=["Month", "Retail_Sales"])
        df["Year"] = df["Month"].dt.year
        df["Month_Num"] = df["Month"].dt.month
        df = df[df["Year"].isin(weights.keys())]

        year_counts = df.groupby("Year")["Month_Num"].nunique()
        incomplete = year_counts[year_counts < 12].index.tolist()
        if incomplete:
            print(f"[RVIA] Skipping incomplete year(s): {incomplete}")
            df = df[~df["Year"].isin(incomplete)]
            weights = {y: w for y, w in weights.items() if y not in incomplete}

        if df.empty or not weights:
            raise ValueError("No complete years after filtering")

        year_totals = df.groupby("Year")["Retail_Sales"].transform("sum")
        df["Proportion"] = df["Retail_Sales"] / year_totals

        total_weight = sum(weights.values())
        blended = {}
        for m in range(1, 13):
            m_rows = df[df["Month_Num"] == m]
            if m_rows.empty:
                blended[m] = 0.0
                continue
            blended[m] = sum(row["Proportion"] * weights.get(int(row["Year"]), 0)
                             for _, row in m_rows.iterrows()) / total_weight

        total = sum(blended.values())
        if total <= 0:
            raise ValueError("Blended proportions summed to zero")
        blended = {m: round(v / total, 6) for m, v in blended.items()}

        names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        print(f"[RVIA] Loaded from: {path}")
        print(f"[RVIA] Years: {sorted(weights.keys())} | weights: {weights}")
        print("[RVIA] " + "  ".join(f"{names[m-1]}:{v:.3f}" for m, v in blended.items()))

        _rvia_index_cache = blended
        return blended
    except Exception as e:
        print(f"[RVIA] Could not load seasonal index ({e}) -- using hardcoded fallback")
        _rvia_index_cache = RV_SEASONAL_INDEX_FALLBACK
        return RV_SEASONAL_INDEX_FALLBACK


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------
def redistribute_annual_to_monthly(annual_df: pd.DataFrame) -> pd.DataFrame:
    seasonal_index = load_rvia_seasonal_index()
    parts = []
    for month, pct in seasonal_index.items():
        part = annual_df.copy()
        year_vals = part["Date"].dt.year
        part["Date"] = pd.to_datetime({"year": year_vals, "month": month, "day": 1})
        part["MonthStart"] = part["Date"].values.astype("datetime64[M]")
        part["Units"] = (part["Units"] * pct).round(4)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def load_raw_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=0, header=0, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]

    required = ["Manufacturer", "Division", "Model", "Type", "Units", "Date"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df["Units"] = pd.to_numeric(df["Units"], errors="coerce").fillna(0)
    df["MonthStart"] = df["Date"].values.astype("datetime64[M]")

    for c in ["Manufacturer", "Division", "Model", "Type"]:
        df[c] = df[c].astype(str).str.strip()

    is_total = (
        df["Manufacturer"].str.lower().eq("grand total") |
        df["Division"].str.lower().eq("grand total") |
        df["Type"].str.lower().eq("grand total")
    )
    dropped = is_total.sum()
    if dropped:
        print(f"[load] Dropping {dropped} Grand Total rollup row(s)")
    df = df[~is_total].copy()

    if "Dealer Name" in df.columns:
        is_annual = df["Dealer Name"].isna()
        annual_rows = df[is_annual].copy()
        monthly_rows = df[~is_annual].copy()
        if not annual_rows.empty:
            n_annual = len(annual_rows)
            annual_units = annual_rows["Units"].sum()
            redistributed = redistribute_annual_to_monthly(annual_rows)
            df = pd.concat([monthly_rows, redistributed], ignore_index=True)
            print(f"[load] Annual rollup rows: {n_annual:,} -> "
                  f"redistributed to {len(redistributed):,} monthly rows via RVIA index")
            print(f"[load] Monthly dealer rows: {len(monthly_rows):,}")
            print(f"[load] Combined total: {len(df):,} rows")

    return df


def build_monthly_series(df: pd.DataFrame, grain: str) -> pd.DataFrame:
    if grain == "Total":
        grouped = df.groupby(["MonthStart"], as_index=False)["Units"].sum()
        grouped["series_id"] = "Total Market"
        return grouped[["series_id", "MonthStart", "Units"]].sort_values("MonthStart")

    if grain == "Division":
        key_cols = ["Division"]
    elif grain == "Type":
        key_cols = ["Type"]
    elif grain == "Division_Type":
        key_cols = ["Division", "Type"]
    else:
        raise ValueError(f"Unknown grain: {grain}")

    grouped = df.groupby(key_cols + ["MonthStart"], as_index=False)["Units"].sum()
    grouped["series_id"] = (grouped[key_cols].agg(" | ".join, axis=1)
                            if len(key_cols) > 1 else grouped[key_cols[0]])
    return grouped[["series_id", "MonthStart", "Units"]].sort_values(["series_id", "MonthStart"])


def fill_month_gaps(sub: pd.DataFrame) -> pd.Series:
    full_range = pd.date_range(sub["MonthStart"].min(), sub["MonthStart"].max(), freq="MS")
    s = sub.set_index("MonthStart")["Units"].reindex(full_range, fill_value=0)
    s.index.name = "MonthStart"
    return s.asfreq("MS").fillna(0)


def fetch_fred_series(api_key: str, start_date: str = "2019-01-01") -> pd.DataFrame:
    from fredapi import Fred
    fred = Fred(api_key=api_key)
    frames = {}
    for code in FRED_SERIES:
        try:
            s = fred.get_series(code, observation_start=start_date)
            s.index = pd.to_datetime(s.index)
            frames[code] = s.resample("MS").mean()
        except Exception as e:
            print(f"[FRED] Skipping {code}: {e}")
    out = pd.DataFrame(frames).ffill().bfill()
    out.index.name = "MonthStart"
    return out.reset_index()


@dataclass
class SeriesProfile:
    series_id: str
    n_months: int
    total_volume: float
    zero_fraction: float
    tier: str
    backtest_months: int
    forecast_months: int = FORECAST_MONTHS
    note: str = ""


def classify_series(series_id: str, series: pd.Series) -> SeriesProfile:
    n = len(series)
    total = float(series.sum())
    zero_frac = float((series == 0).mean())
    bt = max(3, min(12, int(round(n * 0.20))))
    bt = min(bt, n // 2)
    bt = max(bt, 1)

    if zero_frac > INTERMITTENT_ZERO_FRAC:
        tier, note = Tier.INTERMITTENT, f"{zero_frac:.0%} zero months -> TSB + Croston"
    elif n < TIER_MEDIUM_MIN_MONTHS or total < TIER_MEDIUM_MIN_VOLUME:
        tier, note, bt = Tier.THIN, f"Only {n} months / {total:.0f} units -> seasonal-naive", 0
    elif n < TIER_FULL_MIN_MONTHS or total < TIER_FULL_MIN_VOLUME:
        tier, note = Tier.MEDIUM, f"{n} months / {total:.0f} units -> ETS+XGB+LGBM"
    else:
        tier, note = Tier.FULL, ""
    return SeriesProfile(series_id, n, total, zero_frac, tier, bt, FORECAST_MONTHS, note)


def wmape(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    denom = np.abs(actual).sum()
    return float(np.abs(actual - pred).sum() / denom) if denom else np.nan


def mape(actual, pred):
    """Mean Absolute Percentage Error -- unweighted, per-point average.
    Skips points where actual is zero (undefined percentage error there)."""
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    nz = actual != 0
    if not nz.any():
        return np.nan
    return float(np.mean(np.abs((actual[nz] - pred[nz]) / actual[nz])))


def r_squared(actual, pred):
    """Coefficient of determination. Undefined (NaN) when actual has no
    variance (e.g. a single point or a constant series) -- R2 requires a
    variance baseline to compare against."""
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    ss_res = np.sum((actual - pred) ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    if ss_tot == 0:
        return np.nan
    return float(1 - ss_res / ss_tot)


# --- Intermittent ---
def run_tsb(train, horizon, _exog=None):
    alpha, beta = 0.15, 0.15
    y = train.values.astype(float)
    nz = y[y > 0]
    z = nz.mean() if len(nz) else 1.0
    p = len(nz) / len(y) if len(y) else 0.5
    for v in y:
        if v > 0:
            z = beta * v + (1 - beta) * z
            p = alpha * 1 + (1 - alpha) * p
        else:
            p = alpha * 0 + (1 - alpha) * p
    return np.full(horizon, max(0.0, p * z))


def run_croston(train, horizon, _exog=None):
    alpha = 0.15
    y = train.values.astype(float)
    nz_idx = np.where(y > 0)[0]
    if len(nz_idx) == 0:
        return np.zeros(horizon)
    z = float(y[nz_idx[0]])
    q = float(nz_idx[0] + 1) if nz_idx[0] > 0 else 1.0
    prev = nz_idx[0]
    for idx in nz_idx[1:]:
        q = alpha * (idx - prev) + (1 - alpha) * q
        z = alpha * y[idx] + (1 - alpha) * z
        prev = idx
    forecast = z / q if q > 0 else 0.0
    return np.full(horizon, max(0.0, forecast))


def run_seasonal_naive(train, horizon, _exog=None):
    if len(train) >= 12:
        pattern = train.values[-12:]
        return np.tile(pattern, int(np.ceil(horizon / 12)))[:horizon]
    return np.repeat(float(train.values[-1]) if len(train) else 0.0, horizon)


def run_ets(train, horizon, _exog=None):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    sp = 12 if len(train) >= 24 else None
    m = ExponentialSmoothing(train.values.astype(float), trend="add",
                              seasonal="add" if sp else None, seasonal_periods=sp,
                              initialization_method="estimated").fit(optimized=True)
    return np.clip(m.forecast(horizon), 0, None)


def run_autoarima(train, horizon, _exog=None):
    from pmdarima import auto_arima
    m = 12 if len(train) >= 24 else 1
    model = auto_arima(train.values.astype(float), seasonal=(m > 1), m=m,
                        suppress_warnings=True, error_action="ignore", stepwise=True)
    return np.clip(model.predict(n_periods=horizon), 0, None)


def run_sarimax(train, horizon, exog=None):
    """
    ARIMA with exogenous regressors (FRED macro signals) -- the actual
    macro-aware competitor. ETS/AutoARIMA/Prophet/foundation models never
    touch exog even when it's supplied; XGBoost/LightGBM do, but as ML
    feature inputs rather than a proper time-series model with exogenous
    terms. This is the one classical model in the roster built to use FRED
    signals the way they're meant to be used.

    Requires exog -- raises (caught upstream, silently excluded from
    competition) if none is available, so it only ever competes on series
    where FRED data actually applies.
    """
    if exog is None or exog.empty:
        raise RuntimeError("SARIMAX requires exogenous data")

    from pmdarima import auto_arima

    future_dates = pd.date_range(train.index[-1] + pd.offsets.MonthBegin(1),
                                  periods=horizon, freq="MS")
    # Forward-fill across the combined train+future index so future exog
    # values default to the last known reading (matches how the tree models
    # handle exog for dates beyond what's actually known).
    combined = exog.reindex(train.index.union(future_dates)).sort_index().ffill().bfill()
    if combined.isna().any().any():
        raise RuntimeError("Exogenous data has unfillable gaps")

    X_train = combined.loc[train.index].values
    X_future = combined.loc[future_dates].values

    m = 12 if len(train) >= 24 else 1
    model = auto_arima(train.values.astype(float), X=X_train, seasonal=(m > 1), m=m,
                        suppress_warnings=True, error_action="ignore", stepwise=True)
    fc = model.predict(n_periods=horizon, X=X_future)
    return np.clip(np.asarray(fc), 0, None)


def run_prophet(train, horizon, _exog=None):
    from prophet import Prophet
    dfp = pd.DataFrame({"ds": train.index, "y": train.values})
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(dfp)
    future = model.make_future_dataframe(periods=horizon, freq="MS")
    return np.clip(model.predict(future).tail(horizon)["yhat"].values, 0, None)


def _tree_forecast(train, horizon, exog, kind):
    n_lags = min(6, max(2, len(train) // 4))
    df = train.to_frame("y").copy()
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = df["y"].shift(lag)
    df["month"] = df.index.month
    df["year"] = df.index.year
    df["roll3"] = df["y"].shift(1).rolling(3).mean()
    df["roll6"] = df["y"].shift(1).rolling(6).mean()
    if exog is not None and len(train) >= MIN_SERIES_FOR_EXOG:
        df = df.join(exog, how="left")
    df = df.dropna()
    if len(df) < 6:
        raise RuntimeError("Not enough rows")
    feat_cols = [c for c in df.columns if c != "y"]
    X, y = df[feat_cols], df["y"]

    if kind == "xgb":
        from xgboost import XGBRegressor
        model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1,
                              subsample=0.9, colsample_bytree=0.9,
                              random_state=RANDOM_STATE, n_jobs=1)
    else:
        from lightgbm import LGBMRegressor
        model = LGBMRegressor(n_estimators=100, max_depth=3, learning_rate=0.1,
                               subsample=0.9, colsample_bytree=0.9,
                               random_state=RANDOM_STATE, verbose=-1, n_jobs=1)
    model.fit(X, y)

    history = train.copy()
    preds = []
    cur_date = train.index[-1]
    for _ in range(horizon):
        cur_date = cur_date + pd.offsets.MonthBegin(1)
        row = {f"lag_{lag}": history.iloc[-lag] if len(history) >= lag else 0
               for lag in range(1, n_lags + 1)}
        row["month"] = cur_date.month
        row["year"] = cur_date.year
        row["roll3"] = history.iloc[-3:].mean() if len(history) >= 3 else history.mean()
        row["roll6"] = history.iloc[-6:].mean() if len(history) >= 6 else history.mean()
        if exog is not None and len(train) >= MIN_SERIES_FOR_EXOG:
            exog_row = exog.loc[cur_date] if cur_date in exog.index else exog.iloc[-1]
            for c in exog.columns:
                row[c] = exog_row[c]
        pred = float(model.predict(pd.DataFrame([row])[feat_cols])[0])
        preds.append(max(0.0, pred))
        history.loc[cur_date] = preds[-1]
    return np.asarray(preds)


def run_xgboost(train, horizon, exog=None):
    return _tree_forecast(train, horizon, exog, "xgb")

def run_lightgbm(train, horizon, exog=None):
    return _tree_forecast(train, horizon, exog, "lgbm")


# Foundation models (TimesFM, Chronos) are multi-hundred-MB to multi-GB and were
# previously reloaded from scratch on EVERY call -- every backtest fold, every
# series. Cached per-process here so each worker loads once and reuses it.
_timesfm_cache = None
_chronos_cache = None

TIMESFM_CONTEXT_LEN = 512  # fixed so the cached model works for any series length


def run_timesfm(train, horizon, _exog=None):
    global _timesfm_cache
    try:
        import timesfm
        if _timesfm_cache is None:
            model = timesfm.TimesFm(context_len=TIMESFM_CONTEXT_LEN, horizon_len=128,
                                     input_patch_len=32, output_patch_len=128,
                                     num_layers=20, model_dims=1280, backend="cpu")
            model.load_from_checkpoint(repo_id="google/timesfm-2.5-200m-pytorch")
            _timesfm_cache = model
        fc, _ = _timesfm_cache.forecast([train.values], freq=[0])
        return np.asarray(fc[0][:horizon])
    except Exception as e:
        raise RuntimeError(f"TimesFM: {e}")


def run_chronos(train, horizon, _exog=None):
    global _chronos_cache
    try:
        import torch
        from chronos import ChronosPipeline
        if _chronos_cache is None:
            _chronos_cache = ChronosPipeline.from_pretrained(
                "amazon/chronos-t5-large", device_map="cpu", torch_dtype=torch.float32)
        fc = _chronos_cache.predict(torch.tensor(train.values, dtype=torch.float32),
                                     prediction_length=horizon, num_samples=100)
        return np.quantile(fc[0].numpy(), 0.5, axis=0)[:horizon]
    except Exception as e:
        raise RuntimeError(f"Chronos: {e}")


ALL_MODEL_RUNNERS = {
    "TimesFM_2.5": run_timesfm, "Chronos_Large": run_chronos, "ETS": run_ets,
    "AutoARIMA": run_autoarima, "SARIMAX": run_sarimax, "Prophet": run_prophet,
    "XGBoost": run_xgboost, "LightGBM": run_lightgbm,
}
INTERMITTENT_RUNNERS = {"TSB": run_tsb, "Croston": run_croston}
TIER_MODEL_MAP = {
    Tier.FULL: ["TimesFM_2.5", "Chronos_Large", "ETS", "AutoARIMA", "SARIMAX", "Prophet", "XGBoost", "LightGBM"],
    Tier.MEDIUM: ["ETS", "XGBoost", "LightGBM", "SARIMAX"],
    Tier.THIN: [],
}


def rolling_backtest(runner, series, backtest_months, n_origins, exog=None, min_real_date=None):
    """
    Rolling-origin backtest. Returns (mean_wmape, pooled_relative_errors).
    pooled_relative_errors is a flat list of (actual-pred)/actual across every
    non-zero actual observed across all folds -- used afterward to build
    empirical prediction intervals (P10/P90) around the final point forecast.

    min_real_date: if given, each fold still trains on the full available
    history up to that origin, but SCORING only counts test months on or
    after this date. Folds with no real months in their test window are
    skipped entirely (they don't silently count as "no signal" -- they just
    don't contribute a score). This keeps model fitting using all available
    history while making the score itself an honest reflection of genuine
    transactional data rather than the synthetic redistributed years.
    """
    step = max(1, backtest_months // max(n_origins, 1))
    scores = []
    rel_errors = []
    for i in range(n_origins):
        offset = i * step
        end_train = len(series) - backtest_months - offset
        if end_train < 6:
            break
        train = series.iloc[:end_train]
        test = series.iloc[end_train: end_train + backtest_months]
        exog_t = (exog.loc[exog.index.isin(train.index)]
                  if exog is not None and len(train) >= MIN_SERIES_FOR_EXOG else None)
        try:
            pred = runner(train, len(test), exog_t)
            pred_arr = np.asarray(pred, dtype=float)
            actual = test.values.astype(float)

            if min_real_date is not None:
                real_mask = test.index >= min_real_date
                if not real_mask.any():
                    continue  # no genuine transactional months in this fold's test window
                actual = actual[real_mask]
                pred_arr = pred_arr[real_mask]

            score = wmape(actual, pred_arr)
            if not np.isnan(score):
                scores.append(score)
                # Normalize by PREDICTED value (not actual) -- this is the
                # correct basis for a multiplicative interval applied to a new
                # point forecast: actual = pred * (1 + pct_error), so quantiles
                # of pct_error translate directly into P10/P90 = point*(1+q).
                nz = pred_arr > 1e-9
                if nz.any():
                    rel_errors.extend(((actual[nz] - pred_arr[nz]) / pred_arr[nz]).tolist())
        except Exception:
            pass
    mean_score = float(np.mean(scores)) if scores else np.nan
    return mean_score, rel_errors


def count_available_real_test_points(series, min_real_date, backtest_months, n_origins):
    """
    Estimate how many genuine (non-synthetic) test-month observations would
    be available for scoring across all rolling-origin folds, without
    actually running any models. Used to decide upfront whether a series has
    enough real transactional history to trust a real-data-only backtest, or
    whether it needs to fall back to the full window (synthetic-inclusive).
    """
    if min_real_date is None:
        return 0
    step = max(1, backtest_months // max(n_origins, 1))
    count = 0
    for i in range(n_origins):
        offset = i * step
        end_train = len(series) - backtest_months - offset
        if end_train < 6:
            break
        test = series.iloc[end_train: end_train + backtest_months]
        count += int((test.index >= min_real_date).sum())
    return count


def _run_single_fold(runner, series, backtest_months, exog, min_real_date):
    """
    Runs just the MOST RECENT (offset=0) backtest fold and returns the full
    (dates, actual, predicted) arrays for that window, rather than just a
    score. This is what powers the "backtest overlay" on the chart -- how
    well the winning model's own predictions matched actuals over the most
    recent held-out window -- and the MAPE/R2 metrics, which need the raw
    values rather than an aggregated score. Only the most recent origin is
    captured (not all rolling folds) since that's the one directly relevant
    to "how is this model doing right now," and keeping it to one fold
    keeps the overlay a single clean line rather than overlapping segments.
    """
    end_train = len(series) - backtest_months
    if end_train < 6:
        return None
    train = series.iloc[:end_train]
    test = series.iloc[end_train: end_train + backtest_months]
    exog_t = (exog.loc[exog.index.isin(train.index)]
              if exog is not None and len(train) >= MIN_SERIES_FOR_EXOG else None)
    try:
        pred = runner(train, len(test), exog_t)
        pred_arr = np.clip(np.asarray(pred, dtype=float), 0, None)
        actual = test.values.astype(float)
        dates = test.index

        if min_real_date is not None:
            real_mask = dates >= min_real_date
            if not real_mask.any():
                return None
            dates = dates[real_mask]
            actual = actual[real_mask]
            pred_arr = pred_arr[real_mask]

        if len(dates) == 0:
            return None
        return dates, actual, pred_arr
    except Exception:
        return None



def backtest_and_select(series, profile, exog=None, active_models=None, n_origins=ROLLING_ORIGINS,
                         min_real_date=None):
    """
    Returns (final_forecasts dict, bt_scores dict, best_name, pred_interval dict, naive_info dict).
    pred_interval: {"P10_mult": lo, "P90_mult": hi} -- multiplicative offsets from the empirical
                   quantiles of the winning model's backtest relative errors, or None if not enough
                   backtest data to estimate a band.
    naive_info: {"Naive_wMAPE": x, "Beats_Naive": bool, "Real_Data_Backtest": bool,
                 "Real_Test_Points": int} -- always computed except THIN tier (THIN has no
                backtest at all, forecast IS naive already).

    min_real_date: if this series has enough genuine (non-redistributed) transactional
    months to score against (>= MIN_REAL_TEST_POINTS), backtest scoring for every
    candidate model -- including naive and the ensemble -- is restricted to just
    those months. Otherwise falls back to the full window (all history counts for
    scoring, same as before). Fitting always uses full history regardless.
    """
    empty_interval = None

    if profile.tier == Tier.THIN:
        fc = run_seasonal_naive(series, profile.forecast_months)
        return ({"Seasonal_Naive": fc}, {"Seasonal_Naive": np.nan}, "Seasonal_Naive",
                empty_interval, {"Naive_wMAPE": None, "Beats_Naive": None,
                                 "Real_Data_Backtest": False, "Real_Test_Points": 0,
                                 "MAPE": None, "R2": None, "Backtest_Fold_Dates": None,
                                 "Backtest_Fold_Actual": None, "Backtest_Fold_Predicted": None})

    if profile.tier == Tier.INTERMITTENT:
        runners = INTERMITTENT_RUNNERS
    else:
        tier_names = TIER_MODEL_MAP.get(profile.tier, TIER_MODEL_MAP[Tier.MEDIUM])
        base = active_models if active_models else ALL_MODEL_RUNNERS
        runners = {k: v for k, v in base.items() if k in tier_names}
        if not runners:
            runners = {"ETS": run_ets, "XGBoost": run_xgboost, "LightGBM": run_lightgbm}

    n_real_points = count_available_real_test_points(series, min_real_date, profile.backtest_months, n_origins)
    use_real_only = n_real_points >= MIN_REAL_TEST_POINTS
    score_cutoff = min_real_date if use_real_only else None

    # Always score seasonal-naive too, as the best-practice floor every
    # competing model must beat by a real margin, not just edge out by noise.
    naive_score, naive_rel_errors = rolling_backtest(
        run_seasonal_naive, series, profile.backtest_months, n_origins, exog=None,
        min_real_date=score_cutoff)

    bt_scores = {}
    rel_errors_by_model = {}
    for name, runner in runners.items():
        score, rel_errs = rolling_backtest(runner, series, profile.backtest_months, n_origins, exog,
                                            min_real_date=score_cutoff)
        if not np.isnan(score):
            bt_scores[name] = score
            rel_errors_by_model[name] = rel_errs

    if not bt_scores:
        fc = run_seasonal_naive(series, profile.forecast_months)
        naive_info = {"Naive_wMAPE": round(naive_score, 4) if not np.isnan(naive_score) else None,
                      "Beats_Naive": None, "Real_Data_Backtest": use_real_only,
                      "Real_Test_Points": n_real_points,
                      "MAPE": None, "R2": None, "Backtest_Fold_Dates": None,
                      "Backtest_Fold_Actual": None, "Backtest_Fold_Predicted": None}
        return ({"Seasonal_Naive": fc}, {"Seasonal_Naive": np.nan}, "Seasonal_Naive",
                empty_interval, naive_info)

    ranked = sorted(bt_scores.items(), key=lambda x: x[1])
    best_name = ranked[0][0]
    top2 = [n for n, _ in ranked[:2]]

    final = {}
    for name in set(top2):
        try:
            exog_full = exog if (exog is not None and len(series) >= MIN_SERIES_FOR_EXOG) else None
            final[name] = runners[name](series, profile.forecast_months, exog_full)
        except Exception as e:
            print(f"  Refit failed {name}: {e}")

    ensemble_rel_errors = []
    if len(top2) == 2 and all(n in final for n in top2):
        ensemble = np.mean([final[n] for n in top2], axis=0)
        final["Ensemble_Top2"] = ensemble
        ens_scores = []
        for i in range(n_origins):
            step = max(1, profile.backtest_months // max(n_origins, 1))
            offset = i * step
            end_tr = len(series) - profile.backtest_months - offset
            if end_tr < 6:
                break
            t_train = series.iloc[:end_tr]
            t_test = series.iloc[end_tr: end_tr + profile.backtest_months]
            preds = []
            for nm in top2:
                exog_t = (exog.loc[exog.index.isin(t_train.index)]
                          if exog is not None and len(t_train) >= MIN_SERIES_FOR_EXOG else None)
                try:
                    preds.append(runners[nm](t_train, len(t_test), exog_t))
                except Exception:
                    pass
            if len(preds) == 2:
                ens_pred_full = np.mean(preds, axis=0)
                actual_full = t_test.values.astype(float)
                if score_cutoff is not None:
                    real_mask = t_test.index >= score_cutoff
                    if not real_mask.any():
                        continue
                    actual_full = actual_full[real_mask]
                    ens_pred_full = ens_pred_full[real_mask]
                sc = wmape(actual_full, ens_pred_full)
                if not np.isnan(sc):
                    ens_scores.append(sc)
                    nz = ens_pred_full > 1e-9
                    if nz.any():
                        ensemble_rel_errors.extend(
                            ((actual_full[nz] - ens_pred_full[nz]) / ens_pred_full[nz]).tolist())
        if ens_scores:
            bt_scores["Ensemble_Top2"] = float(np.mean(ens_scores))
            rel_errors_by_model["Ensemble_Top2"] = ensemble_rel_errors
            if bt_scores["Ensemble_Top2"] < bt_scores[best_name]:
                best_name = "Ensemble_Top2"

    # Best-practice gate: the winning model must beat seasonal-naive by a real
    # relative margin, not just win a noisy small-sample comparison.
    beats_naive = None
    if not np.isnan(naive_score) and naive_score > 0:
        margin = (naive_score - bt_scores[best_name]) / naive_score
        beats_naive = margin >= NAIVE_BEAT_MARGIN
        if not beats_naive:
            best_name = "Seasonal_Naive"
            if "Seasonal_Naive" not in final:
                final["Seasonal_Naive"] = run_seasonal_naive(series, profile.forecast_months)
            bt_scores["Seasonal_Naive"] = naive_score
            rel_errors_by_model["Seasonal_Naive"] = naive_rel_errors

    # Always fold naive's error distribution into the pool too -- it's a
    # legitimate candidate whose plausible range matters for uncertainty
    # even when it didn't win the gate.
    if naive_rel_errors:
        rel_errors_by_model.setdefault("Seasonal_Naive", naive_rel_errors)

    naive_info = {"Naive_wMAPE": round(naive_score, 4) if not np.isnan(naive_score) else None,
                  "Beats_Naive": beats_naive, "Real_Data_Backtest": use_real_only,
                  "Real_Test_Points": n_real_points}

    # Prediction interval built from POOLED backtest relative errors across
    # every model that competed for this series (not just the winner). A
    # single model's own historical residual spread understates real
    # uncertainty -- it can't see that a different model would have called
    # a given month differently. Pooling candidate models' errors folds that
    # cross-model disagreement into the band, in the same spirit as
    # conformal prediction over a model set rather than a single fitted model.
    pooled_errors = []
    for name in rel_errors_by_model:
        pooled_errors.extend(rel_errors_by_model[name])

    pred_interval = empty_interval
    if len(pooled_errors) >= 4:
        lo_q = float(np.percentile(pooled_errors, 10))
        hi_q = float(np.percentile(pooled_errors, 90))
        pred_interval = {"P10_mult": 1 + lo_q, "P90_mult": 1 + hi_q}

    # Capture the winning model's most recent backtest fold in full detail --
    # powers the "backtest overlay" on the chart and the MAPE/R2 metrics.
    # Added into naive_info rather than a new return value, to avoid
    # changing this function's signature (and every call site) for what's
    # fundamentally more diagnostic metadata about the same selection.
    backtest_fold = None
    try:
        if best_name == "Ensemble_Top2":
            comp_names = top2
            comp_results = []
            for nm in comp_names:
                r = runners.get(nm)
                if r is not None:
                    comp_results.append(_run_single_fold(r, series, profile.backtest_months, exog, score_cutoff))
            comp_results = [r for r in comp_results if r is not None]
            if len(comp_results) == 2 and len(comp_results[0][0]) == len(comp_results[1][0]):
                dates = comp_results[0][0]
                actual = comp_results[0][1]
                pred = np.mean([r[2] for r in comp_results], axis=0)
                backtest_fold = (dates, actual, pred)
        elif best_name in runners:
            backtest_fold = _run_single_fold(runners[best_name], series, profile.backtest_months,
                                             exog, score_cutoff)
        elif best_name == "Seasonal_Naive":
            backtest_fold = _run_single_fold(run_seasonal_naive, series, profile.backtest_months,
                                             None, score_cutoff)
    except Exception as e:
        print(f"  Backtest fold capture failed: {e}")

    naive_info["MAPE"] = None
    naive_info["R2"] = None
    naive_info["Backtest_Fold_Dates"] = None
    naive_info["Backtest_Fold_Actual"] = None
    naive_info["Backtest_Fold_Predicted"] = None
    if backtest_fold is not None:
        dates, actual, pred = backtest_fold
        naive_info["MAPE"] = round(mape(actual, pred), 4)
        r2 = r_squared(actual, pred)
        naive_info["R2"] = round(r2, 4) if not np.isnan(r2) else None
        naive_info["Backtest_Fold_Dates"] = list(dates)
        naive_info["Backtest_Fold_Actual"] = actual.tolist()
        naive_info["Backtest_Fold_Predicted"] = pred.tolist()

    return final, bt_scores, best_name, pred_interval, naive_info


def _model_one_series(grain, sid, sub, exog, active_models, n_origins, min_real_date=None):
    series = fill_month_gaps(sub)
    series.name = sid

    hist_rows = [{"Grain": grain, "series_id": sid, "MonthStart": d, "Units": float(u)}
                 for d, u in zip(series.index, series.values)]

    def _make_exog(for_series, horizon):
        if exog is None or len(for_series) < MIN_SERIES_FOR_EXOG:
            return None
        fut_idx = pd.date_range(for_series.index[-1] + pd.offsets.MonthBegin(1),
                                 periods=horizon, freq="MS")
        return exog.reindex(for_series.index.union(fut_idx)).ffill().bfill()

    # ------------------------------------------------------------------
    # Best-practice held-out validation: reserve the last VALIDATION_MONTHS
    # as genuine unseen data. Model selection and backtest scoring happen
    # ONLY on data before this window -- it is never touched during selection.
    # The selected model is then refit on the pre-holdout data and scored
    # against the real holdout to get an honest out-of-sample accuracy
    # number, separate from the backtest wMAPE used to pick the model.
    # Finally, the selected model is refit on the FULL series (including
    # the holdout months) to produce the actual forward-looking forecast.
    #
    # Priority conflict: with only a handful of genuine transactional months
    # available right now, reserving the most recent VALIDATION_MONTHS for
    # holdout would remove most or all of the real-scoring data real-only
    # backtesting needs. Real-only backtest scoring matters more right now
    # (it's what tells you whether a model beats naive on genuine demand
    # variation, not synthetic redistribution) so validation is skipped for
    # a series when doing it wouldn't leave enough real months behind.
    # ------------------------------------------------------------------
    real_months_in_series = int((series.index >= min_real_date).sum()) if min_real_date is not None else 0
    would_starve_real_scoring = (
        min_real_date is not None
        and real_months_in_series > 0
        and (real_months_in_series - VALIDATION_MONTHS) < MIN_REAL_TEST_POINTS
    )
    do_validation = len(series) >= MIN_HISTORY_FOR_VALIDATION and not would_starve_real_scoring
    validation_wmape = None

    selection_series = series.iloc[:-VALIDATION_MONTHS] if do_validation else series
    profile = classify_series(sid, selection_series)

    try:
        selection_exog = _make_exog(selection_series, profile.forecast_months)
        forecasts, bt_scores, best, pred_interval, naive_info = backtest_and_select(
            selection_series, profile, selection_exog, active_models, n_origins,
            min_real_date=min_real_date)
    except Exception as e:
        print(f"  {sid} failed: {e}")
        forecasts = {"Seasonal_Naive": run_seasonal_naive(selection_series, profile.forecast_months)}
        bt_scores = {"Seasonal_Naive": np.nan}
        best = "Seasonal_Naive"
        pred_interval = None
        naive_info = {"Naive_wMAPE": None, "Beats_Naive": None,
                      "Real_Data_Backtest": False, "Real_Test_Points": 0,
                      "MAPE": None, "R2": None, "Backtest_Fold_Dates": None,
                      "Backtest_Fold_Actual": None, "Backtest_Fold_Predicted": None}

    if do_validation and profile.tier not in (Tier.THIN,):
        try:
            holdout = series.iloc[-VALIDATION_MONTHS:]
            runner = (ALL_MODEL_RUNNERS.get(best) or INTERMITTENT_RUNNERS.get(best)
                      or (run_seasonal_naive if best == "Seasonal_Naive" else None))
            if best == "Ensemble_Top2":
                # Approximate by re-deriving the two component models from bt_scores
                # ranking on the selection series (same models the ensemble used).
                candidates = sorted(bt_scores.items(), key=lambda x: x[1])
                comp_names = [n for n, _ in candidates if n != "Ensemble_Top2"][:2]
                comp_runners = [ALL_MODEL_RUNNERS.get(n) or INTERMITTENT_RUNNERS.get(n) for n in comp_names]
                if all(comp_runners):
                    val_exog = _make_exog(selection_series, VALIDATION_MONTHS)
                    preds = [r(selection_series, VALIDATION_MONTHS, val_exog) for r in comp_runners]
                    val_pred = np.mean(preds, axis=0)
                    validation_wmape = round(wmape(holdout.values, val_pred), 4)
            elif runner is not None:
                val_exog = _make_exog(selection_series, VALIDATION_MONTHS)
                val_pred = runner(selection_series, VALIDATION_MONTHS, val_exog)
                validation_wmape = round(wmape(holdout.values, val_pred), 4)
        except Exception as e:
            print(f"  Validation refit failed for {sid}: {e}")

    # Final refit on the FULL series (all history, including any holdout
    # months) to produce the actual forward-looking forecast that gets used.
    forecast_months = FORECAST_MONTHS
    full_exog = _make_exog(series, forecast_months)
    try:
        if best == "Ensemble_Top2":
            candidates = sorted(bt_scores.items(), key=lambda x: x[1])
            comp_names = [n for n, _ in candidates if n != "Ensemble_Top2"][:2]
            comp_runners = [ALL_MODEL_RUNNERS.get(n) or INTERMITTENT_RUNNERS.get(n) for n in comp_names]
            if all(comp_runners):
                preds = [r(series, forecast_months, full_exog) for r in comp_runners]
                final_fc = np.mean(preds, axis=0)
            else:
                final_fc = forecasts.get(best, run_seasonal_naive(series, forecast_months))
        else:
            runner = (ALL_MODEL_RUNNERS.get(best) or INTERMITTENT_RUNNERS.get(best)
                      or (run_seasonal_naive if best == "Seasonal_Naive" else None))
            final_fc = runner(series, forecast_months, full_exog) if runner else \
                       run_seasonal_naive(series, forecast_months)
    except Exception as e:
        print(f"  Final refit failed for {sid}: {e}")
        final_fc = run_seasonal_naive(series, forecast_months)

    final_fc = np.clip(np.asarray(final_fc, dtype=float), 0, None)

    p10 = p90 = None
    if pred_interval is not None:
        p10_raw = final_fc * pred_interval["P10_mult"]
        p90_raw = final_fc * pred_interval["P90_mult"]
        # Guarantee the point forecast always sits inside its own interval.
        # A model with systematic historical bias could otherwise produce a
        # band that's shifted entirely above or below the point estimate --
        # statistically real, but reads as "broken" to anyone looking at it.
        p10 = np.clip(np.minimum(p10_raw, final_fc), 0, None)
        p90 = np.clip(np.maximum(p90_raw, final_fc), 0, None)

    future_dates = pd.date_range(series.index[-1] + pd.offsets.MonthBegin(1),
                                  periods=forecast_months, freq="MS")

    fc_rows = []
    for i, d in enumerate(future_dates):
        fc_rows.append({
            "Grain": grain, "series_id": sid, "Model": best, "MonthStart": d,
            "Forecast_Units": max(0.0, round(float(final_fc[i]), 2)),
            "P10_Units": max(0.0, round(float(p10[i]), 2)) if p10 is not None else None,
            "P90_Units": max(0.0, round(float(p90[i]), 2)) if p90 is not None else None,
            "Is_Selected_Model": True, "Tier": profile.tier,
            "History_Months": profile.n_months, "Backtest_Months": profile.backtest_months,
            "Zero_Fraction": round(profile.zero_fraction, 3) if pd.notna(profile.zero_fraction) else 0.0, "Note": profile.note,
            "Naive_wMAPE": naive_info.get("Naive_wMAPE"), "Beats_Naive": naive_info.get("Beats_Naive"),
            "Validation_wMAPE": validation_wmape,
            "Real_Data_Backtest": naive_info.get("Real_Data_Backtest"),
            "Real_Test_Points": naive_info.get("Real_Test_Points"),
            "MAPE": naive_info.get("MAPE"), "R2": naive_info.get("R2"),
        })
    # Also record the non-selected competing models' backtest-window forecasts
    # (for the "show all models" comparison view) -- these use the selection
    # series' own forecast window, not the final full-history refit.
    for model_name, fc in forecasts.items():
        if model_name == best:
            continue
        sel_future = pd.date_range(selection_series.index[-1] + pd.offsets.MonthBegin(1),
                                    periods=len(fc), freq="MS")
        for d, v in zip(sel_future, fc):
            fc_rows.append({
                "Grain": grain, "series_id": sid, "Model": model_name, "MonthStart": d,
                "Forecast_Units": max(0.0, round(float(v), 2)),
                "P10_Units": None, "P90_Units": None,
                "Is_Selected_Model": False, "Tier": profile.tier,
                "History_Months": profile.n_months, "Backtest_Months": profile.backtest_months,
                "Zero_Fraction": round(profile.zero_fraction, 3) if pd.notna(profile.zero_fraction) else 0.0, "Note": profile.note,
                "Naive_wMAPE": naive_info.get("Naive_wMAPE"), "Beats_Naive": naive_info.get("Beats_Naive"),
                "Validation_wMAPE": validation_wmape,
                "Real_Data_Backtest": naive_info.get("Real_Data_Backtest"),
                "Real_Test_Points": naive_info.get("Real_Test_Points"),
                "MAPE": None, "R2": None,
            })

    bt_rows = [{"Grain": grain, "series_id": sid, "Model": m,
                "Backtest_wMAPE": round(s, 4) if not np.isnan(s) else None,
                "Selected": m == best, "Tier": profile.tier,
                "Naive_wMAPE": naive_info.get("Naive_wMAPE"),
                "Beats_Naive": naive_info.get("Beats_Naive"),
                "Validation_wMAPE": validation_wmape if m == best else None,
                "Real_Data_Backtest": naive_info.get("Real_Data_Backtest"),
                "Real_Test_Points": naive_info.get("Real_Test_Points"),
                "MAPE": naive_info.get("MAPE") if m == best else None,
                "R2": naive_info.get("R2") if m == best else None}
               for m, s in bt_scores.items()]

    # Backtest fold detail -- month-by-month actual vs predicted for the
    # winning model's most recent held-out window. Separate table (not
    # merged into fc_rows/hist_rows) since it's neither a future forecast
    # nor a historical actual -- it's what the model WOULD have predicted
    # for months we already have real answers for. Powers the "backtest
    # overlay" segment on the chart.
    backtest_detail_rows = []
    fold_dates = naive_info.get("Backtest_Fold_Dates")
    fold_actual = naive_info.get("Backtest_Fold_Actual")
    fold_pred = naive_info.get("Backtest_Fold_Predicted")
    if fold_dates:
        for d, a, p in zip(fold_dates, fold_actual, fold_pred):
            backtest_detail_rows.append({
                "Grain": grain, "series_id": sid, "Model": best, "MonthStart": d,
                "Actual": round(float(a), 2), "Predicted": round(float(p), 2),
            })

    return fc_rows, bt_rows, hist_rows, backtest_detail_rows


def _variance_from_band(p10, p90):
    """
    Approximate forecast error variance from a P10-P90 band, treating it as
    an 80% interval under a normal approximation (z ~= 1.2816 on each side).
    Used as the weighting basis for WLS-style reconciliation below -- a
    tighter historical band means a series' forecast gets more trust when
    combined with other information at the parent level.
    """
    if pd.isna(p10) or pd.isna(p90):
        return None
    half_width = (p90 - p10) / (2 * 1.2816)
    return max(half_width, 1e-6) ** 2


def _compute_trailing_shares(history_df: pd.DataFrame, grain_name: str, trailing_months: int = 12) -> dict:
    """
    Each series' share of Total, based on its most recent trailing_months of
    genuine actual history. Used to translate Total's forecast (highest
    volume, most statistically reliable series you have) into an implied
    forecast for each Division/Type, as a third information source for
    reconciliation -- exactly what pure parent-child WLS reconciliation
    was missing, since it never let Total have a say in the numbers below it.
    """
    sub = history_df[history_df["Grain"] == grain_name]
    total_hist = history_df[history_df["Grain"] == "Total"]
    if sub.empty or total_hist.empty:
        return {}

    last_date = total_hist["MonthStart"].max()
    cutoff = last_date - pd.DateOffset(months=trailing_months - 1)

    series_totals = (sub[sub["MonthStart"] >= cutoff]
                     .groupby("series_id")["Units"].sum())
    total_total = total_hist[total_hist["MonthStart"] >= cutoff]["Units"].sum()

    if total_total <= 0:
        return {}
    return (series_totals / total_total).to_dict()


def reconcile_bottom_up(forecast_df: pd.DataFrame, history_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    WLS-style hierarchical reconciliation (a practical, well-precedented
    simplification of full MinT/optimal reconciliation) in two stages:

    Stage 1 (parent-child): each Division/Type forecast is an
    inverse-variance-weighted combination of (a) the bottom-up sum of its
    live Division x Type children, and (b) that level's own directly-modeled
    forecast.

    Stage 2 (Total-anchored): the Stage 1 result is further blended with a
    THIRD source -- Total's own forecast (your highest-volume, most
    statistically reliable series) translated into an implied Division/Type
    number via that series' trailing 12-month share of Total. Without this,
    Total never had any influence on the Division/Type numbers at all, even
    though it's the most trustworthy single forecast you have.

    Both stages use the same principle: whichever source has the tighter
    (more trustworthy) historical error band gets more weight. This is a
    sequential approximation of full single-shot MinT reconciliation across
    the whole hierarchy at once -- directionally correct and standard
    practice for a lighter implementation, not the fully rigorous version.
    """
    sel = forecast_df[forecast_df["Is_Selected_Model"]].copy()
    dt = sel[sel["Grain"] == "Division_Type"].copy()
    if dt.empty:
        forecast_df["Reconciled"] = False
        return forecast_df

    dt[["_div", "_type"]] = dt["series_id"].str.split(" | ", n=1, expand=True, regex=False)

    # Different Division x Type combos can have different "last actual month" --
    # e.g. a discontinued model line whose data stops in 2022 vs a currently
    # active one with data through last month. Each combo's forecast starts
    # right after ITS OWN last actual month, so summing by calendar MonthStart
    # across combos with different anchors mixes forecasts from completely
    # different points in time under the same date label (a 2022 forecast for
    # a dead product line would get added into the "2022" bucket right next to
    # live products' 2026 forecasts, corrupting every calendar month's total).
    # Only combos anchored to the most common (current) starting point are
    # included in the bottom-up sum -- stale/discontinued combos are excluded
    # from reconciliation. They still get their own direct forecast if looked
    # up individually; they just don't distort the Division/Type roll-up.
    first_fc_month = dt.groupby("series_id")["MonthStart"].min()
    total_start = sel[sel["Grain"] == "Total"]["MonthStart"].min() if (sel["Grain"] == "Total").any() else None
    current_anchor = total_start if pd.notna(total_start) else first_fc_month.max()
    live_series = first_fc_month[first_fc_month == current_anchor].index
    n_total = dt["series_id"].nunique()
    n_live = len(live_series)
    if n_total > n_live:
        print(f"[reconcile] {n_total - n_live} of {n_total} Division x Type combo(s) have "
              f"stale/discontinued history and are excluded from bottom-up reconciliation "
              f"(forecast anchored earlier than {current_anchor:%Y-%m} instead of matching "
              f"the current period).")

    dt = dt[dt["series_id"].isin(live_series)]
    if dt.empty:
        forecast_df["Reconciled"] = False
        return forecast_df

    dt["_variance"] = dt.apply(lambda r: _variance_from_band(r.get("P10_Units"), r.get("P90_Units")), axis=1)

    total_fc = sel[sel["Grain"] == "Total"][["MonthStart", "Forecast_Units", "P10_Units", "P90_Units"]].copy()
    total_fc["Total_Variance"] = total_fc.apply(
        lambda r: _variance_from_band(r["P10_Units"], r["P90_Units"]), axis=1)
    total_fc = total_fc.rename(columns={"Forecast_Units": "Total_Forecast"})
    has_total_anchor = not total_fc.empty and history_df is not None and not history_df.empty

    recon_rows = []
    for level_col, grain_name in [("_div", "Division"), ("_type", "Type")]:
        # Bottom-up sum and its variance (sum of children's variances, assuming
        # independence across children -- the same simplifying assumption WLS
        # reconciliation makes by using a diagonal rather than full covariance).
        bu = (dt.groupby([level_col, "MonthStart"])
              .agg(BU_Forecast=("Forecast_Units", "sum"),
                   BU_Variance=("_variance", lambda s: s.sum(min_count=1)))
              .reset_index().rename(columns={level_col: "series_id"}))

        # The parent's own direct-model forecast + its own variance, from
        # BEFORE any superseding -- this is the second information source.
        direct = sel[sel["Grain"] == grain_name][
            ["series_id", "MonthStart", "Forecast_Units", "P10_Units", "P90_Units"]
        ].rename(columns={"Forecast_Units": "Direct_Forecast"})
        direct["Direct_Variance"] = direct.apply(
            lambda r: _variance_from_band(r["P10_Units"], r["P90_Units"]), axis=1)

        merged = bu.merge(direct[["series_id", "MonthStart", "Direct_Forecast", "Direct_Variance"]],
                          on=["series_id", "MonthStart"], how="left")

        def _combine2(bu_val, bu_var, dir_val, dir_var):
            w_bu = 1.0 / bu_var if pd.notna(bu_var) and bu_var > 0 else 0.0
            w_dir = 1.0 / dir_var if pd.notna(dir_var) and dir_var > 0 else 0.0
            if w_bu == 0 and w_dir == 0:
                return bu_val, None
            if w_dir == 0 or pd.isna(dir_val):
                return bu_val, bu_var
            return (bu_val * w_bu + dir_val * w_dir) / (w_bu + w_dir), 1.0 / (w_bu + w_dir)

        def _combine(row):
            # Missing variance = no usable weight from that source -> defer
            # entirely to whichever source does have one (or bottom-up if
            # neither does, since it's always available here by construction).
            val, var = _combine2(row["BU_Forecast"], row["BU_Variance"],
                                 row.get("Direct_Forecast"), row.get("Direct_Variance"))
            return pd.Series({"Forecast_Units": val, "_combined_var": var})

        combined = merged.apply(_combine, axis=1)
        merged = pd.concat([merged, combined], axis=1)

        # Stage 2: blend in Total's implied contribution via trailing share.
        if has_total_anchor:
            shares = _compute_trailing_shares(history_df, grain_name)
            merged["_share"] = merged["series_id"].map(shares)
            merged = merged.merge(total_fc[["MonthStart", "Total_Forecast", "Total_Variance"]],
                                  on="MonthStart", how="left")

            def _combine_stage2(row):
                share = row.get("_share")
                total_val, total_var = row.get("Total_Forecast"), row.get("Total_Variance")
                if pd.isna(share) or pd.isna(total_val) or pd.isna(total_var):
                    return row["Forecast_Units"], row["_combined_var"]
                td_val = total_val * share
                td_var = (share ** 2) * total_var
                return _combine2(row["Forecast_Units"], row["_combined_var"], td_val, td_var)

            stage2 = merged.apply(lambda r: pd.Series(_combine_stage2(r), index=["Forecast_Units", "_combined_var"]), axis=1)
            merged[["Forecast_Units", "_combined_var"]] = stage2

        merged["P10_Units"] = merged.apply(
            lambda r: max(0.0, r["Forecast_Units"] - 1.2816 * np.sqrt(r["_combined_var"]))
            if pd.notna(r["_combined_var"]) else None, axis=1)
        merged["P90_Units"] = merged.apply(
            lambda r: max(0.0, r["Forecast_Units"] + 1.2816 * np.sqrt(r["_combined_var"]))
            if pd.notna(r["_combined_var"]) else None, axis=1)

        agg = merged[["series_id", "MonthStart", "Forecast_Units", "P10_Units", "P90_Units"]].copy()
        agg["Grain"] = grain_name
        agg["Model"] = "WLS_Reconciled"
        agg["Is_Selected_Model"] = True
        agg["Reconciled"] = True
        recon_rows.append(agg)

    recon_df = pd.concat(recon_rows, ignore_index=True)
    for c in ["Tier", "History_Months", "Backtest_Months", "Zero_Fraction", "Note",
              "P10_Units", "P90_Units", "Naive_wMAPE", "Beats_Naive", "Validation_wMAPE",
              "Real_Data_Backtest", "Real_Test_Points"]:
        if c not in recon_df.columns:
            recon_df[c] = None

    forecast_df = forecast_df.copy()
    if "Reconciled" not in forecast_df.columns:
        forecast_df["Reconciled"] = False

    # A reconciled row now exists for these (Grain, series_id, MonthStart) combos.
    # Turn off Is_Selected_Model on the original direct-model row so exactly ONE
    # selected row survives per key -- otherwise summing Is_Selected_Model rows
    # double-counts every Division/Type series that Division_Type also covered.
    key_cols = ["Grain", "series_id", "MonthStart"]
    superseded = recon_df[key_cols].drop_duplicates().assign(_has_recon=True)
    merged = forecast_df.merge(superseded, on=key_cols, how="left")
    supersede_mask = merged["_has_recon"].fillna(False).values
    forecast_df.loc[supersede_mask, "Is_Selected_Model"] = False

    return pd.concat([forecast_df, recon_df], ignore_index=True)


def compute_market_share_from_raw(raw: pd.DataFrame) -> pd.DataFrame:
    if not OWN_DIVISIONS:
        return pd.DataFrame()
    own_mask = raw["Division"].isin(OWN_DIVISIONS)
    by_type_month = raw.groupby(["Type", "MonthStart"])["Units"].sum().reset_index()
    own_by_type_month = (raw[own_mask].groupby(["Type", "MonthStart"])["Units"].sum()
                         .reset_index().rename(columns={"Units": "Own_Units"}))
    merged = by_type_month.merge(own_by_type_month, on=["Type", "MonthStart"], how="left")
    merged["Own_Units"] = merged["Own_Units"].fillna(0)
    merged["Market_Share"] = merged["Own_Units"] / merged["Units"].replace(0, np.nan)
    return merged.rename(columns={"Units": "Total_Units"})


def run_pipeline(data_path: str = DATA_PATH, fred_api_key: str = None, grains=GRAINS,
                  top_n_per_grain: int = None, progress_callback=None, fast_mode: bool = True,
                  n_jobs: int = -1, n_origins: int = ROLLING_ORIGINS,
                  skip_heavy_models: bool = True):

    def _prog(stage, pct):
        if progress_callback:
            progress_callback(stage, min(pct, 1.0))
        print(f"\r[{pct*100:5.1f}%] {stage[:70]:<70}", end="", flush=True)

    active_models = {k: v for k, v in ALL_MODEL_RUNNERS.items()
                     if not (fast_mode and k in ("Prophet", "AutoARIMA"))}

    heavy_models_present = any(k in active_models for k in ("TimesFM_2.5", "Chronos_Large"))
    if skip_heavy_models:
        active_models = {k: v for k, v in active_models.items()
                         if k not in ("TimesFM_2.5", "Chronos_Large")}
        heavy_models_present = False

    # TimesFM/Chronos are multi-hundred-MB to multi-GB. Each worker PROCESS loads
    # its own copy (processes don't share memory), so running with many parallel
    # workers while these are active can load the model dozens of times
    # simultaneously and thrash system RAM -- often slower than fewer workers.
    # Cap workers when heavy models are in play unless the caller explicitly
    # requested a low number already.
    HEAVY_MODEL_MAX_JOBS = int(os.environ.get("HEAVY_MODEL_MAX_JOBS", 4))
    if heavy_models_present and (n_jobs == -1 or n_jobs > HEAVY_MODEL_MAX_JOBS):
        print(f"[perf] TimesFM/Chronos active -- capping parallel workers to "
              f"{HEAVY_MODEL_MAX_JOBS} to avoid loading multi-GB models into RAM "
              f"repeatedly across many processes (override with HEAVY_MODEL_MAX_JOBS "
              f"in .env, or pass --no-heavy to skip these models entirely for a much "
              f"faster run).")
        n_jobs = HEAVY_MODEL_MAX_JOBS

    _prog("Loading raw data", 0.02)
    raw = load_raw_data(data_path)
    print()

    # Earliest date where genuine (non-redistributed) transactional data
    # exists -- rows that came from the annual rollup redistribution keep
    # their original Dealer Name = null, so this survives the redistribution
    # untouched. Used to restrict backtest SCORING to real months only, on
    # any series with enough of them (see MIN_REAL_TEST_POINTS).
    min_real_date = None
    if "Dealer Name" in raw.columns:
        real_rows = raw[raw["Dealer Name"].notna()]
        if not real_rows.empty:
            min_real_date = real_rows["MonthStart"].min()
            print(f"[real-data] Genuine transactional data starts {min_real_date:%Y-%m} -- "
                  f"backtest scoring will restrict to real months where a series has "
                  f"enough of them (>= {MIN_REAL_TEST_POINTS} test points), falling back "
                  f"to the full window otherwise.")

    exog = None
    api_key = fred_api_key or FRED_API_KEY
    if api_key:
        _prog("Fetching FRED signals", 0.05)
        try:
            fred_df = fetch_fred_series(api_key, start_date=str(raw["MonthStart"].min().date()))
            exog = fred_df.set_index("MonthStart")
            print(f"\n  FRED: {list(exog.columns)}")
        except Exception as e:
            print(f"\n  FRED fetch failed: {e}")

    _prog("Computing market share", 0.07)
    market_share_df = compute_market_share_from_raw(raw)
    print()

    jobs = []
    for grain in grains:
        monthly = build_monthly_series(raw, grain)
        sids = monthly["series_id"].unique().tolist()
        if top_n_per_grain:
            totals = monthly.groupby("series_id")["Units"].sum().sort_values(ascending=False)
            sids = totals.head(top_n_per_grain).index.tolist()
        for sid in sids:
            jobs.append((grain, sid, monthly[monthly["series_id"] == sid]))

    n_jobs_str = "all cores" if n_jobs == -1 else str(n_jobs)
    _prog(f"Modeling {len(jobs)} series [{n_jobs_str} workers, fast={'Y' if fast_mode else 'N'}]", 0.10)
    print()

    results_list = []
    use_parallel = n_jobs != 1 and len(jobs) > 1
    if use_parallel:
        try:
            from joblib import Parallel, delayed
            results_list = Parallel(n_jobs=n_jobs, prefer="processes")(
                delayed(_model_one_series)(g, s, sub, exog, active_models, n_origins, min_real_date)
                for g, s, sub in jobs)
            _prog("Modeling complete", 0.90); print()
        except Exception as e:
            print(f"\n  Parallel failed ({e}), sequential fallback")
            use_parallel = False

    if not use_parallel:
        for i, (grain, sid, sub) in enumerate(jobs):
            pct = 0.10 + (i / max(len(jobs), 1)) * 0.80
            _prog(f"[{grain}] {sid[:50]}", pct)
            try:
                results_list.append(_model_one_series(grain, sid, sub, exog, active_models, n_origins, min_real_date))
            except Exception as e:
                print(f"\n  {sid} failed: {e}")
                results_list.append(([], [], [], []))

    all_fc, all_bt, all_hist, all_bt_detail = [], [], [], []
    for fc_rows, bt_rows, hist_rows, bt_detail_rows in results_list:
        all_fc.extend(fc_rows); all_bt.extend(bt_rows); all_hist.extend(hist_rows)
        all_bt_detail.extend(bt_detail_rows)

    _prog("Assembling tables", 0.92); print()
    forecast_df = pd.DataFrame(all_fc)
    backtest_df = pd.DataFrame(all_bt)
    history_df = pd.DataFrame(all_hist)
    backtest_detail_df = pd.DataFrame(all_bt_detail)

    if "Division_Type" in grains and len(forecast_df):
        _prog("Hierarchical reconciliation", 0.95); print()
        forecast_df = reconcile_bottom_up(forecast_df, history_df)
    elif len(forecast_df):
        forecast_df["Reconciled"] = False

    _prog("Done", 1.0); print()

    from datetime import datetime
    return {
        "forecast": forecast_df, "backtest": backtest_df, "history": history_df,
        "backtest_detail": backtest_detail_df,
        "exog": exog.reset_index() if exog is not None else pd.DataFrame(),
        "market_share": market_share_df, "run_timestamp": datetime.now(),
    }


if __name__ == "__main__":
    r = run_pipeline(fast_mode=True, n_jobs=1)
    print(r["forecast"].head())
