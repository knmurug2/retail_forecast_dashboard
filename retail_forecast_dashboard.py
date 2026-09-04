"""
retail_forecast_dashboard.py  (simple)
=======================================
One page. Pick a series, see the chart, export if you want. That's it.

Run: python -m streamlit run retail_forecast_dashboard.py
Requires: python run_forecast.py has been run first (produces parquet files)
"""
import os, sys, json, io
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import retail_forecast_engine as engine

st.set_page_config(page_title="RV Retail Forecast", layout="wide", page_icon="📈")

ACTUAL_COLOR = "#1f77b4"
FORECAST_COLOR = "#d62728"


@st.cache_data(show_spinner=False)
def load_data(pdir: str):
    needed = ["forecast.parquet", "backtest.parquet", "history.parquet"]
    missing = [f for f in needed if not os.path.exists(os.path.join(pdir, f))]
    if missing:
        return None, f"Missing: {missing}"

    fc = pd.read_parquet(os.path.join(pdir, "forecast.parquet"))
    bt = pd.read_parquet(os.path.join(pdir, "backtest.parquet"))
    hs = pd.read_parquet(os.path.join(pdir, "history.parquet"))
    fc["MonthStart"] = pd.to_datetime(fc["MonthStart"])
    hs["MonthStart"] = pd.to_datetime(hs["MonthStart"])

    meta = {}
    mp = os.path.join(pdir, "run_meta.json")
    if os.path.exists(mp):
        meta = json.load(open(mp))

    # Optional -- only present if compute_attach_rate_forecast.py has been run.
    # Dashboard works fine without it, just skips the Dometic-units view.
    attach_rates = None
    ar_path = os.path.join(pdir, "attach_rates.parquet")
    if os.path.exists(ar_path):
        try:
            attach_rates = pd.read_parquet(ar_path)
        except Exception:
            attach_rates = None

    # Optional -- Product Area breakdown, only if the sales pull included it.
    area_mix = None
    am_path = os.path.join(pdir, "area_mix.parquet")
    if os.path.exists(am_path):
        try:
            area_mix = pd.read_parquet(am_path)
        except Exception:
            area_mix = None

    # Optional -- month-by-month backtest actual vs predicted for each
    # series' winning model's most recent held-out window. Powers the
    # backtest overlay on the chart and the MAPE/R2 metric cards. Only
    # present if the engine that produced this parquet is new enough to
    # capture it -- older runs just won't have this file.
    backtest_detail = None
    btd_path = os.path.join(pdir, "backtest_detail.parquet")
    if os.path.exists(btd_path):
        try:
            backtest_detail = pd.read_parquet(btd_path)
            if not backtest_detail.empty:
                backtest_detail["MonthStart"] = pd.to_datetime(backtest_detail["MonthStart"])
        except Exception:
            backtest_detail = None

    return {"fc": fc, "bt": bt, "hs": hs, "meta": meta,
            "attach_rates": attach_rates, "area_mix": area_mix,
            "backtest_detail": backtest_detail}, None


# ---------------------------------------------------------------------------
# LOAD
# ---------------------------------------------------------------------------
pdir = engine.PARQUET_DIR
data, err = load_data(pdir)

if err:
    st.title("📈 RV Retail Sales Forecast")
    st.error(f"No forecast data found yet. {err}")
    st.code("python run_forecast.py", language="bash")
    st.stop()

fc_df, bt_df, hs_df, meta = data["fc"], data["bt"], data["hs"], data["meta"]
attach_rates_df = data.get("attach_rates")
area_mix_df = data.get("area_mix")
backtest_detail_df = data.get("backtest_detail")
sel_df = fc_df[fc_df["Is_Selected_Model"]].copy()

# ---------------------------------------------------------------------------
# ATTACH RATE LOOKUP -- lens for viewing Dometic units instead of market units
# Only meaningful at Division and Division_Type grain, since attach rate is
# computed per-manufacturer (Division). Type and Total mix many manufacturers
# together, so there's no single rate that applies.
# ---------------------------------------------------------------------------
HAS_ATTACH_RATES = attach_rates_df is not None and not attach_rates_df.empty
_attach_rate_map = {}
if HAS_ATTACH_RATES:
    _attach_rate_map = dict(zip(attach_rates_df["Division"], attach_rates_df["Attach_Rate"]))


def get_attach_rate(series_id: str, grain: str):
    """Returns the attach rate for a Division or Division_Type series, or
    None if not available (Type/Total grain, or no relationship data)."""
    if not HAS_ATTACH_RATES or grain not in ("Division", "Division_Type"):
        return None
    div = series_id.split(" | ", 1)[0] if grain == "Division_Type" else series_id
    rate = _attach_rate_map.get(div)
    return rate if pd.notna(rate) else None


def scale_series(values, rate):
    """Applies an attach rate to a pandas Series/array, passing through
    unchanged if no rate is available."""
    if rate is None:
        return values
    return values * rate


# ---------------------------------------------------------------------------
# TOP BAR — title + refresh
# ---------------------------------------------------------------------------
top_l, top_r = st.columns([5, 1])
with top_l:
    st.title("📈 RV Retail Sales Forecast")
    ts = meta.get("run_timestamp", "")
    if ts:
        st.caption(f"Updated {datetime.fromisoformat(ts):%b %d, %Y %I:%M %p}")
with top_r:
    if st.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

st.write("")

# ---------------------------------------------------------------------------
# MODE — Single series vs Compare (ranked)
# ---------------------------------------------------------------------------
mode_col, lens_col = st.columns([3, 2])
with mode_col:
    mode_options = ["Single series", "Compare", "Backtest"]
    if HAS_ATTACH_RATES:
        mode_options.append("Dometic Summary")
    mode = st.radio("Mode", mode_options, horizontal=True, label_visibility="collapsed")
with lens_col:
    if HAS_ATTACH_RATES and mode != "Dometic Summary":
        view_as = st.radio("View as", ["Market units", "Dometic units (est.)"],
                            horizontal=True, label_visibility="collapsed")
    else:
        view_as = "Market units"

show_dometic = view_as == "Dometic units (est.)"

st.write("")

GRAIN_LABELS = {"Total": "All (Total Market)", "Division": "Division",
                "Type": "RV Type", "Division_Type": "Division × Type"}
GRAIN_ORDER = ["Total", "Division", "Type", "Division_Type"]
available_grains = [g for g in GRAIN_ORDER if g in fc_df["Grain"].unique()]

# ============================================================
# MODE: SINGLE SERIES
# ============================================================
if mode == "Single series":
    c1, c2, c3 = st.columns([1, 2, 1])

    with c1:
        grain = st.selectbox("View by", available_grains,
                              format_func=lambda g: GRAIN_LABELS.get(g, g))

    with c2:
        if grain == "Total":
            series = "Total Market"
            st.selectbox("Series", ["Total Market"], disabled=True)
        else:
            options = sorted(fc_df[fc_df["Grain"] == grain]["series_id"].unique())
            if options:
                totals = (hs_df[hs_df["Grain"] == grain].groupby("series_id")["Units"]
                          .sum().sort_values(ascending=False))
                default_idx = options.index(totals.index[0]) if totals.index[0] in options else 0
            else:
                default_idx = 0
            series = st.selectbox("Series", options, index=default_idx if options else 0)

    with c3:
        horizon = st.radio("Forecast", ["12 months", "6 months", "3 months"], horizontal=False)
        h_months = {"12 months": 12, "6 months": 6, "3 months": 3}[horizon]

    st.write("")

    if not series:
        st.info("No series available for this view.")
        st.stop()

    hist = hs_df[(hs_df["Grain"] == grain) & (hs_df["series_id"] == series)].sort_values("MonthStart").copy()
    fc = sel_df[(sel_df["Grain"] == grain) & (sel_df["series_id"] == series)].sort_values("MonthStart").copy()
    fc_trim = fc.head(h_months).copy()

    rate = get_attach_rate(series, grain) if show_dometic else None
    if show_dometic and rate is None:
        st.warning(f"⚠️ Dometic-units view isn't available for {GRAIN_LABELS.get(grain, grain)} "
                   f"({'attach rate applies per-manufacturer, not per-type' if grain in ('Type', 'Total') else 'no Dometic sales relationship data for this series'}). "
                   f"Showing market units instead.")
    elif rate is not None:
        hist["Units"] = hist["Units"] * rate
        fc_trim["Forecast_Units"] = fc_trim["Forecast_Units"] * rate
        for col in ["P10_Units", "P90_Units"]:
            if col in fc_trim.columns:
                fc_trim[col] = fc_trim[col] * rate
        st.caption(f"📊 Showing estimated Dometic units — market units × {rate:.1%} attach rate for {series}")

    units_label = "Dometic units (est.)" if (show_dometic and rate is not None) else "Units"

    # Backtest overlay -- the winning model's own predictions over its most
    # recent held-out window, plotted alongside actuals so you can see how
    # well it actually fit recent history, not just a single score. Only
    # present if backtest_detail.parquet exists (engine run after this
    # feature was added).
    bt_detail = pd.DataFrame()
    if backtest_detail_df is not None and not backtest_detail_df.empty:
        bt_detail = backtest_detail_df[(backtest_detail_df["Grain"] == grain) &
                                       (backtest_detail_df["series_id"] == series)].sort_values("MonthStart").copy()
        if rate is not None and not bt_detail.empty:
            bt_detail["Actual"] = bt_detail["Actual"] * rate
            bt_detail["Predicted"] = bt_detail["Predicted"] * rate

    fig = go.Figure()

    # Shaded prediction interval band (P10-P90), drawn first so it sits behind the lines
    has_interval = fc_trim["P10_Units"].notna().any() if "P10_Units" in fc_trim.columns else False
    if has_interval:
        fig.add_trace(go.Scatter(
            x=pd.concat([fc_trim["MonthStart"], fc_trim["MonthStart"][::-1]]),
            y=pd.concat([fc_trim["P90_Units"], fc_trim["P10_Units"][::-1]]),
            fill="toself", fillcolor="rgba(214,39,40,0.12)",
            line=dict(width=0), hoverinfo="skip", showlegend=True, name="P10–P90 range",
        ))

    fig.add_trace(go.Scatter(
        x=hist["MonthStart"], y=hist["Units"], mode="lines", name="Actual",
        line=dict(color=ACTUAL_COLOR, width=2.5),
        hovertemplate="%{x|%b %Y}<br>%{y:,.0f} units<extra></extra>",
    ))
    if not bt_detail.empty:
        fig.add_trace(go.Scatter(
            x=bt_detail["MonthStart"], y=bt_detail["Predicted"], mode="lines+markers",
            name="Backtest fit", line=dict(color="#ff9f1c", width=2, dash="dot"),
            marker=dict(size=4),
            hovertemplate="%{x|%b %Y}<br>predicted %{y:,.0f} units<extra></extra>",
        ))
    if not hist.empty and not fc_trim.empty:
        bridge_x = [hist["MonthStart"].iloc[-1], fc_trim["MonthStart"].iloc[0]]
        bridge_y = [hist["Units"].iloc[-1], fc_trim["Forecast_Units"].iloc[0]]
        fig.add_trace(go.Scatter(x=bridge_x, y=bridge_y, mode="lines", showlegend=False,
                                  line=dict(color=FORECAST_COLOR, width=2, dash="dot")))
    fig.add_trace(go.Scatter(
        x=fc_trim["MonthStart"], y=fc_trim["Forecast_Units"], mode="lines+markers", name="Forecast",
        line=dict(color=FORECAST_COLOR, width=2.5),
        hovertemplate="%{x|%b %Y}<br>%{y:,.0f} units<extra></extra>",
    ))

    fig.update_layout(
        height=440, margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="", yaxis_title=units_label, hovermode="x unified",
        legend=dict(orientation="h", y=1.1, x=0), plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    k1, k2, k3, k4 = st.columns(4)
    last_12 = hist[hist["MonthStart"] >= hist["MonthStart"].max() - pd.DateOffset(months=11)]["Units"].sum() \
              if not hist.empty else 0
    next_period = fc_trim["Forecast_Units"].sum()
    model_used = fc_trim["Model"].iloc[0] if not fc_trim.empty else "–"

    k1.metric(f"Last 12 months (actual) — {units_label.lower()}", f"{last_12:,.0f}")
    k2.metric(f"Next {h_months} months (forecast) — {units_label.lower()}", f"{next_period:,.0f}")
    k3.metric("Model used", model_used)

    val_wmape = fc_trim["Validation_wMAPE"].iloc[0] if not fc_trim.empty and "Validation_wMAPE" in fc_trim.columns else None
    beats_naive = fc_trim["Beats_Naive"].iloc[0] if not fc_trim.empty and "Beats_Naive" in fc_trim.columns else None
    if pd.notna(val_wmape):
        k4.metric("Held-out validation accuracy", f"{1 - val_wmape:.0%}",
                  help="Accuracy on 3 months of data never used for model selection or backtest scoring -- "
                       "the most honest accuracy estimate available.")
    elif beats_naive is False:
        k4.metric("Model check", "Using naive")
    else:
        k4.metric("Held-out validation", "n/a (short history)")

    if beats_naive is False:
        st.caption("ℹ️ No competing model beat the seasonal-naive baseline by a meaningful margin here, "
                   "so seasonal-naive (repeat last year's pattern) was used instead of a more complex model.")

    # Model accuracy metrics -- MAPE and R2 come from the winning model's
    # most recent held-out backtest fold (same window as the overlay above);
    # Backtest wMAPE is the score actually used for model selection, pulled
    # from bt_df for reference alongside the other two.
    mape_val = fc_trim["MAPE"].iloc[0] if not fc_trim.empty and "MAPE" in fc_trim.columns else None
    r2_val = fc_trim["R2"].iloc[0] if not fc_trim.empty and "R2" in fc_trim.columns else None
    bt_wmape_val = None
    if not bt_df.empty:
        bt_row = bt_df[(bt_df["Grain"] == grain) & (bt_df["series_id"] == series) & (bt_df["Selected"] == True)]
        if not bt_row.empty:
            bt_wmape_val = bt_row["Backtest_wMAPE"].iloc[0]

    if pd.notna(mape_val) or pd.notna(r2_val) or pd.notna(bt_wmape_val):
        st.write("")
        m1, m2, m3 = st.columns(3)
        m1.metric("MAPE", f"{mape_val:.2%}" if pd.notna(mape_val) else "n/a",
                  help="Mean Absolute Percentage Error on the model's most recent held-out window "
                       "(the backtest overlay shown on the chart above)")
        m2.metric("Backtest wMAPE", f"{bt_wmape_val:.2%}" if pd.notna(bt_wmape_val) else "n/a",
                  help="Weighted MAPE — the score actually used to select the winning model")
        m3.metric("R² Score", f"{r2_val:.3f}" if pd.notna(r2_val) else "n/a",
                  help="How much of the actual variation the model's backtest predictions explain "
                       "(1.0 = perfect fit, 0 = no better than predicting the average)")

    if not bt_detail.empty:
        st.write("")
        with st.expander("📋 Backtest error by month (held-out test window)"):
            err_tbl = bt_detail[["MonthStart", "Actual", "Predicted"]].copy()
            err_tbl["Error"] = err_tbl["Predicted"] - err_tbl["Actual"]
            err_tbl["Error %"] = (err_tbl["Error"] / err_tbl["Actual"].replace(0, pd.NA) * 100)
            err_tbl["MonthStart"] = err_tbl["MonthStart"].dt.strftime("%b %Y")
            display_tbl = err_tbl.copy()
            for c in ["Actual", "Predicted", "Error"]:
                display_tbl[c] = display_tbl[c].map("{:,.0f}".format)
            display_tbl["Error %"] = display_tbl["Error %"].map(
                lambda v: f"{v:+.1f}%" if pd.notna(v) else "–")
            display_tbl.columns = ["Month", "Actual", "Predicted", "Error", "Error %"]
            st.dataframe(display_tbl, hide_index=True, use_container_width=True)
            st.caption("This is the winning model's own predictions for its most recent held-out "
                       "window — the same one shown as the orange 'Backtest fit' line on the chart "
                       "above, and the basis for the MAPE/R² metrics.")

    # Reconciliation check -- Total Market is independently modeled, not a
    # pure sum of Divisions (see caption below for why). This makes that gap
    # visible and checkable every time, instead of needing to compute it
    # by hand whenever someone asks "does this roll up correctly?"
    if grain == "Total" and not show_dometic:
        total_start = fc["MonthStart"].min() if not fc.empty else None
        div_sel = sel_df[sel_df["Grain"] == "Division"].copy()
        if pd.notna(total_start):
            div_sel = div_sel[div_sel.groupby("series_id")["MonthStart"].transform("min") == total_start]
        div_trim = div_sel[div_sel.groupby("series_id")["MonthStart"].rank(method="first") <= h_months]
        sum_of_divisions = div_trim["Forecast_Units"].sum()
        gap_pct = ((next_period / sum_of_divisions) - 1) * 100 if sum_of_divisions > 0 else None

        st.write("")
        with st.expander("🔍 Reconciliation check — Total vs. sum of Divisions"):
            r1, r2, r3 = st.columns(3)
            r1.metric(f"Total Market forecast ({h_months}mo)", f"{next_period:,.0f}")
            r2.metric(f"Sum of all Divisions ({h_months}mo)", f"{sum_of_divisions:,.0f}")
            r3.metric("Gap", f"{gap_pct:+.1f}%" if gap_pct is not None else "n/a")
            st.caption("Total Market is independently modeled (see 'Model used' above), not defined "
                       "as the sum of Divisions. Each Division's forecast is pulled toward consistency "
                       "with Total through weighted reconciliation, but not forced to exactly match it — "
                       "a small gap here is the expected, correct output of that method, not an error.")

    st.write("")
    with st.expander("⬇️ Export to Excel"):
        scope = st.radio("What to export", ["Just this series", "Everything"], horizontal=True)

        def build_excel_single(export_fc_df: pd.DataFrame, export_hs_df: pd.DataFrame):
            buf = io.BytesIO()
            export_fc = export_fc_df.copy()
            export_hs = export_hs_df.copy()
            export_fc["MonthStart"] = export_fc["MonthStart"].dt.strftime("%Y-%m")
            export_hs["MonthStart"] = export_hs["MonthStart"].dt.strftime("%Y-%m")
            with pd.ExcelWriter(buf, engine="openpyxl") as w:
                export_fc.to_excel(w, sheet_name="Forecast", index=False)
                export_hs.to_excel(w, sheet_name="History", index=False)
            buf.seek(0)
            return buf.read()

        scope_key = "series" if scope == "Just this series" else "all"
        if scope_key == "series":
            s_fc = fc[["MonthStart", "Model", "Forecast_Units"]].copy()
            s_hs = hist[["MonthStart", "Units"]].copy()
            xl_bytes = build_excel_single(s_fc, s_hs)
            fname = f"{series}_forecast.xlsx"
        else:
            xl_bytes = build_excel_single(sel_df, hs_df)
            fname = "RV_Retail_Forecast_All.xlsx"

        st.download_button("Download", data=xl_bytes, file_name=fname,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True, key="dl_single")

# ============================================================
# MODE: COMPARE (ranked Top N / All)
# ============================================================
elif mode == "Compare":
    compare_grains = [g for g in available_grains if g != "Total"]
    c1, c2, c3 = st.columns([1, 2, 1])

    with c1:
        grain = st.selectbox("Compare by", compare_grains,
                              format_func=lambda g: GRAIN_LABELS.get(g, g), key="cmp_grain")

    all_series = sorted(fc_df[fc_df["Grain"] == grain]["series_id"].unique())
    n_total = len(all_series)

    with c2:
        show_all = st.checkbox(f"Show all {n_total}", value=(n_total <= 10))
        if show_all:
            top_n = n_total
        else:
            top_n = st.slider("Top N", min_value=3, max_value=max(3, n_total), value=min(10, n_total))

    with c3:
        horizon = st.radio("Forecast", ["12 months", "6 months", "3 months"], horizontal=False, key="cmp_horizon")
        h_months = {"12 months": 12, "6 months": 6, "3 months": 3}[horizon]

    st.write("")

    compare_show_dometic = show_dometic and grain in ("Division", "Division_Type")
    if show_dometic and grain not in ("Division", "Division_Type"):
        st.warning(f"⚠️ Dometic-units view isn't available for {GRAIN_LABELS.get(grain, grain)} "
                   f"(attach rate applies per-manufacturer, not per-type). Showing market units instead.")

    g_fc = sel_df[sel_df["Grain"] == grain].copy()
    g_fc_trim = g_fc[g_fc.groupby("series_id")["MonthStart"].rank(method="first") <= h_months].copy()
    g_hist = hs_df[hs_df["Grain"] == grain].copy()

    compare_units_label = "Units"
    if compare_show_dometic:
        div_col = (g_fc_trim["series_id"].str.split(" | ", n=1, expand=False, regex=False).str[0]
                   if grain == "Division_Type" else g_fc_trim["series_id"])
        g_fc_trim["_Rate"] = div_col.map(_attach_rate_map)
        n_before = g_fc_trim["series_id"].nunique()
        g_fc_trim = g_fc_trim[g_fc_trim["_Rate"].notna()]
        n_after = g_fc_trim["series_id"].nunique()
        g_fc_trim["Forecast_Units"] = g_fc_trim["Forecast_Units"] * g_fc_trim["_Rate"]

        hist_div_col = (g_hist["series_id"].str.split(" | ", n=1, expand=False, regex=False).str[0]
                        if grain == "Division_Type" else g_hist["series_id"])
        g_hist = g_hist.copy()
        g_hist["_Rate"] = hist_div_col.map(_attach_rate_map)
        g_hist = g_hist[g_hist["_Rate"].notna()]
        g_hist["Units"] = g_hist["Units"] * g_hist["_Rate"]

        compare_units_label = "Dometic units (est.)"
        if n_after < n_before:
            st.caption(f"📊 Showing estimated Dometic units. {n_after} of {n_before} series have "
                       f"attach-rate data — the rest are excluded from this view (no Dometic sales "
                       f"relationship data for them).")
        else:
            st.caption("📊 Showing estimated Dometic units — market units × each Division's attach rate")

    ranked = (g_fc_trim.groupby("series_id")["Forecast_Units"].sum()
              .sort_values(ascending=False).reset_index()
              .rename(columns={"Forecast_Units": "Forecast"}))

    # Last 12 months actual, per series in this grain
    last_date = g_hist["MonthStart"].max() if not g_hist.empty else None
    g_hist_12 = g_hist[g_hist["MonthStart"] >= last_date - pd.DateOffset(months=11)] if last_date is not None else g_hist
    actual_12 = (g_hist_12.groupby("series_id")["Units"].sum()
                 .reset_index().rename(columns={"Units": "Actual_Last12"}))

    ranked = ranked.merge(actual_12, on="series_id", how="left")
    ranked["Actual_Last12"] = ranked["Actual_Last12"].fillna(0)
    ranked_top = ranked.head(top_n)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=ranked_top["series_id"], x=ranked_top["Actual_Last12"], orientation="h",
        name="Last 12 months (actual)", marker_color=ACTUAL_COLOR,
        hovertemplate="%{y}<br>%{x:,.0f} units<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=ranked_top["series_id"], x=ranked_top["Forecast"], orientation="h",
        name=f"Next {h_months} months (forecast)", marker_color=FORECAST_COLOR,
        hovertemplate="%{y}<br>%{x:,.0f} units<extra></extra>",
    ))
    fig.update_layout(
        height=max(320, 34 * len(ranked_top)),
        margin=dict(l=10, r=10, t=10, b=10),
        barmode="group",
        xaxis_title=compare_units_label, yaxis_title="",
        yaxis=dict(autorange="reversed"),
        legend=dict(orientation="h", y=1.08, x=0),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    total_shown = ranked_top["Forecast"].sum()
    total_actual_shown = ranked_top["Actual_Last12"].sum()
    total_all = ranked["Forecast"].sum()
    k1, k2, k3 = st.columns(3)
    k1.metric(f"Shown ({len(ranked_top)} of {n_total}) — last 12mo actual", f"{total_actual_shown:,.0f}")
    k2.metric(f"Shown — next {h_months}mo forecast", f"{total_shown:,.0f}")
    k3.metric("All " + GRAIN_LABELS.get(grain, grain).lower() + " — forecast", f"{total_all:,.0f}")

    st.write("")
    st.markdown(f"**Aggregate trend — sum of the {len(ranked_top)} series shown above**")

    shown_ids = ranked_top["series_id"].tolist()
    agg_hist = (hs_df[(hs_df["Grain"] == grain) & (hs_df["series_id"].isin(shown_ids))]
                .groupby("MonthStart")["Units"].sum().reset_index().sort_values("MonthStart"))

    # Different series can have different "last actual month" -- a Division
    # with stale/discontinued history has its own forecast anchored to
    # wherever ITS data ends (e.g. 2022), not the current period. Summing
    # Forecast_Units by calendar MonthStart across series with different
    # anchors mixes forecasts from completely different points in time under
    # the same date label (same bug already fixed for hierarchical
    # reconciliation -- applies here too since this chart also aggregates
    # across many series by calendar month). Only series anchored to the
    # most common (current) starting point are included in this trend line;
    # excluded series still count fully in the ranked bar chart above, which
    # sums each series' own total independent of calendar alignment.
    fc_for_trend = g_fc_trim[g_fc_trim["series_id"].isin(shown_ids)]
    if not fc_for_trend.empty:
        first_fc_month = fc_for_trend.groupby("series_id")["MonthStart"].min()
        current_anchor = first_fc_month.max()
        live_ids = first_fc_month[first_fc_month == current_anchor].index
        n_stale = fc_for_trend["series_id"].nunique() - len(live_ids)
        fc_for_trend = fc_for_trend[fc_for_trend["series_id"].isin(live_ids)]
        if n_stale:
            st.caption(f"ℹ️ {n_stale} of {len(shown_ids)} series have stale/discontinued history "
                       f"(forecast anchored to an earlier period than {current_anchor:%Y-%m}) and are "
                       f"excluded from this trend line specifically, to avoid mixing forecasts from "
                       f"different time periods under the same month. Still included in the ranked "
                       f"chart above.")

    agg_fc = (fc_for_trend.groupby("MonthStart")["Forecast_Units"].sum().reset_index().sort_values("MonthStart"))

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=agg_hist["MonthStart"], y=agg_hist["Units"], mode="lines", name="Actual",
        line=dict(color=ACTUAL_COLOR, width=2.5),
        hovertemplate="%{x|%b %Y}<br>%{y:,.0f} units<extra></extra>",
    ))
    if not agg_hist.empty and not agg_fc.empty:
        bridge_x = [agg_hist["MonthStart"].iloc[-1], agg_fc["MonthStart"].iloc[0]]
        bridge_y = [agg_hist["Units"].iloc[-1], agg_fc["Forecast_Units"].iloc[0]]
        fig_trend.add_trace(go.Scatter(x=bridge_x, y=bridge_y, mode="lines", showlegend=False,
                                        line=dict(color=FORECAST_COLOR, width=2, dash="dot")))
    fig_trend.add_trace(go.Scatter(
        x=agg_fc["MonthStart"], y=agg_fc["Forecast_Units"], mode="lines+markers", name="Forecast",
        line=dict(color=FORECAST_COLOR, width=2.5),
        hovertemplate="%{x|%b %Y}<br>%{y:,.0f} units<extra></extra>",
    ))
    fig_trend.update_layout(
        height=340, margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="", yaxis_title=compare_units_label, hovermode="x unified",
        legend=dict(orientation="h", y=1.1, x=0), plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    st.write("")
    with st.expander(f"📋 Table — {GRAIN_LABELS.get(grain, grain)} ranked"):
        tbl = ranked.copy()
        tbl.insert(0, "Rank", range(1, len(tbl) + 1))
        tbl = tbl.rename(columns={"Actual_Last12": "Last 12mo Actual", "Forecast": f"Next {h_months}mo Forecast"})
        tbl["Last 12mo Actual"] = tbl["Last 12mo Actual"].map("{:,.0f}".format)
        tbl[f"Next {h_months}mo Forecast"] = tbl[f"Next {h_months}mo Forecast"].map("{:,.0f}".format)
        st.dataframe(tbl, hide_index=True, use_container_width=True)

    with st.expander("⬇️ Export to Excel"):
        def build_excel_compare(export_fc_df: pd.DataFrame, export_rank_df: pd.DataFrame, h: int):
            buf = io.BytesIO()
            export_fc = export_fc_df[["series_id", "MonthStart", "Model", "Forecast_Units"]].copy()
            export_fc["MonthStart"] = export_fc["MonthStart"].dt.strftime("%Y-%m")
            export_rank = export_rank_df.rename(columns={
                "Actual_Last12": "Last_12mo_Actual", "Forecast": f"Next_{h}mo_Forecast"})
            with pd.ExcelWriter(buf, engine="openpyxl") as w:
                export_rank.to_excel(w, sheet_name="Ranked_Summary", index=False)
                export_fc.to_excel(w, sheet_name="Forecast_Detail", index=False)
            buf.seek(0)
            return buf.read()

        xl_bytes = build_excel_compare(g_fc_trim, ranked, h_months)
        st.download_button("Download", data=xl_bytes,
                            file_name=f"{grain}_comparison.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True, key="dl_compare")

# ============================================================
# MODE: BACKTEST
# ============================================================
elif mode == "Backtest":
    c1, c2 = st.columns([1, 3])

    with c1:
        grain = st.selectbox("Grain", available_grains,
                              format_func=lambda g: GRAIN_LABELS.get(g, g), key="bt_grain")

    with c2:
        view = st.radio("View", ["One series", "All series"], horizontal=True, key="bt_view")

    st.write("")

    grain_series = sorted(fc_df[fc_df["Grain"] == grain]["series_id"].unique())
    grain_bt = bt_df[bt_df["Grain"] == grain].copy()

    # ------------------------------------------------------------------
    # ONE SERIES — full model-by-model breakdown for a single series
    # ------------------------------------------------------------------
    if view == "One series":
        series = st.selectbox("Series", grain_series, key="bt_series")

        bt_series = grain_bt[grain_bt["series_id"] == series].dropna(subset=["Backtest_wMAPE"]).copy()

        if bt_series.empty:
            st.info("No backtest scores for this series — it used a fallback method "
                     "(not enough history to score against, e.g. seasonal-naive).")
        else:
            bt_series = bt_series.sort_values("Backtest_wMAPE")

            fig = go.Figure(go.Bar(
                x=bt_series["Backtest_wMAPE"], y=bt_series["Model"], orientation="h",
                marker_color=[FORECAST_COLOR if s else "#888" for s in bt_series["Selected"]],
                hovertemplate="%{y}<br>wMAPE: %{x:.1%}<extra></extra>",
            ))
            fig.update_layout(
                height=max(280, 40 * len(bt_series)),
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="wMAPE (lower = better)", yaxis_title="",
                xaxis_tickformat=".0%",
                yaxis=dict(autorange="reversed"),
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Red bar = model used for the forecast. wMAPE measured by holding out "
                       "recent months and checking how close each model's prediction was.")

            st.write("")
            real_flag = bt_series["Real_Data_Backtest"].iloc[0] if "Real_Data_Backtest" in bt_series.columns and not bt_series.empty else None
            n_real = bt_series["Real_Test_Points"].iloc[0] if "Real_Test_Points" in bt_series.columns and not bt_series.empty else None
            if real_flag is True:
                st.success(f"✓ Scored on {int(n_real)} genuine transactional months — not the redistributed synthetic years.")
            elif real_flag is False:
                st.caption("ℹ️ Not enough genuine transactional months yet for this series — "
                           "scored on the full window, which includes redistributed synthetic years.")

            bt_show = bt_series[["Model", "Backtest_wMAPE", "Selected", "Naive_wMAPE", "Validation_wMAPE"]].copy()
            bt_show["Backtest_wMAPE"] = bt_show["Backtest_wMAPE"].map("{:.1%}".format)
            bt_show["Naive_wMAPE"] = bt_show["Naive_wMAPE"].map(lambda v: f"{v:.1%}" if pd.notna(v) else "–")
            bt_show["Validation_wMAPE"] = bt_show["Validation_wMAPE"].map(lambda v: f"{v:.1%}" if pd.notna(v) else "–")
            bt_show["Selected"] = bt_show["Selected"].map({True: "✓", False: ""})
            bt_show.columns = ["Model", "Backtest wMAPE", "Used", "Naive baseline wMAPE", "Held-out validation wMAPE"]
            st.dataframe(bt_show, hide_index=True, use_container_width=True)
            st.caption("Naive baseline = seasonal-naive score every model had to beat by a real margin to be used. "
                       "Held-out validation = accuracy on months never touched during model selection — "
                       "the most honest number here.")

        fc_meta = fc_df[(fc_df["Grain"] == grain) & (fc_df["series_id"] == series)]
        # Reconciled (BU_Reconciled) rows don't carry per-series modeling metadata
        # (Tier/History_Months/Note come out as NaN/None on those) -- prefer a
        # direct-model row for this info if one exists, only fall back to
        # whatever's available if the series was fully superseded.
        if "Reconciled" in fc_meta.columns:
            meta_candidates = fc_meta[fc_meta["Reconciled"] != True]
            if meta_candidates.empty:
                meta_candidates = fc_meta
        else:
            meta_candidates = fc_meta

        if not meta_candidates.empty:
            row = meta_candidates.iloc[0]
            m1, m2, m3 = st.columns(3)
            tier_val = row.get("Tier")
            m1.metric("Tier", tier_val if pd.notna(tier_val) else "–")
            hist_val = row.get("History_Months")
            m2.metric("History months", int(hist_val) if pd.notna(hist_val) else "–")
            note_val = row.get("Note")
            if pd.notna(note_val) and str(note_val).strip():
                st.caption(f"ℹ️ {note_val}")

    # ------------------------------------------------------------------
    # ALL SERIES — one row per series, sortable table + leaderboard
    # ------------------------------------------------------------------
    else:
        selected_bt = grain_bt[grain_bt["Selected"]].dropna(subset=["Backtest_wMAPE"]).copy()

        if selected_bt.empty:
            st.info("No backtest scores available for this grain.")
        else:
            all_tbl = selected_bt[["series_id", "Model", "Backtest_wMAPE", "Validation_wMAPE", "Beats_Naive", "Tier"]].copy()
            all_tbl = all_tbl.sort_values("Backtest_wMAPE")
            all_tbl.columns = ["Series", "Model used", "Backtest wMAPE", "Validation wMAPE", "Beats naive?", "Tier"]
            all_tbl["Backtest wMAPE"] = all_tbl["Backtest wMAPE"].map("{:.1%}".format)
            all_tbl["Validation wMAPE"] = all_tbl["Validation wMAPE"].map(lambda v: f"{v:.1%}" if pd.notna(v) else "–")
            all_tbl["Beats naive?"] = all_tbl["Beats naive?"].map({True: "✓", False: "using naive", None: "–"})
            st.dataframe(all_tbl, hide_index=True, use_container_width=True, height=420)

            st.write("")
            st.markdown("**Leaderboard — model win count and typical accuracy**")
            win_counts = selected_bt["Model"].value_counts().reset_index()
            win_counts.columns = ["Model", "Times selected"]
            acc = selected_bt.groupby("Model")["Backtest_wMAPE"].median().reset_index()
            acc.columns = ["Model", "Median wMAPE"]
            leaderboard = win_counts.merge(acc, on="Model").sort_values("Times selected", ascending=False)
            leaderboard["Median wMAPE"] = leaderboard["Median wMAPE"].map("{:.1%}".format)
            st.dataframe(leaderboard, hide_index=True, use_container_width=True)

            n_no_score = grain_bt[grain_bt["Selected"]]["Backtest_wMAPE"].isna().sum()
            if n_no_score:
                st.caption(f"{n_no_score} series used a fallback method with no backtest score "
                           f"(not enough history) and aren't shown above.")

    with st.expander("⬇️ Export backtest scores"):
        def build_excel_backtest(grain_bt_df: pd.DataFrame):
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as w:
                grain_bt_df.to_excel(w, sheet_name="Backtest_Scores", index=False)
            buf.seek(0)
            return buf.read()

        xl_bytes = build_excel_backtest(grain_bt)
        st.download_button("Download", data=xl_bytes,
                            file_name=f"{grain}_backtest_scores.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True, key="dl_backtest")

# ============================================================
# MODE: DOMETIC SUMMARY -- exec-facing projection for existing customers
# ============================================================
elif mode == "Dometic Summary":
    st.markdown("**How much Dometic product will we sell, given the market forecast "
               "and our current order rates with existing customers.**")

    c1, c2 = st.columns([1, 1])
    with c1:
        exec_horizon = st.radio("Forecast horizon", ["12 months", "6 months", "3 months"],
                                horizontal=True, key="exec_horizon")
        exec_h_months = {"12 months": 12, "6 months": 6, "3 months": 3}[exec_horizon]
    with c2:
        exec_top_n = st.slider("Show top N customers", min_value=5, max_value=50, value=15, key="exec_topn")

    st.write("")

    # Only existing customers -- a real, known Dometic sales relationship.
    # Divisions with no relationship data are excluded entirely (projecting
    # a number for a company Dometic doesn't currently sell to isn't a
    # sales forecast, it's a market-entry guess -- different question).
    existing = attach_rates_df[attach_rates_df["Has_Dometic_Relationship"] == True].copy()

    if existing.empty:
        st.info("No existing-customer attach rate data available. Run compute_attach_rate_forecast.py first.")
    else:
        div_fc = sel_df[sel_df["Grain"] == "Division"].copy()
        div_fc_trim = div_fc[div_fc.groupby("series_id")["MonthStart"].rank(method="first") <= exec_h_months]
        market_totals = (div_fc_trim.groupby("series_id")["Forecast_Units"].sum()
                         .rename("Market_Forecast_Units"))

        existing = existing.merge(market_totals, left_on="Division", right_index=True, how="left")
        existing = existing.dropna(subset=["Market_Forecast_Units"])
        existing["Dometic_Forecast_Units"] = existing["Market_Forecast_Units"] * existing["Attach_Rate"]
        existing = existing.sort_values("Dometic_Forecast_Units", ascending=False)

        # Last-N-months actual Dometic units (est.), for growth context
        div_hist = hs_df[(hs_df["Grain"] == "Division") & (hs_df["series_id"].isin(existing["Division"]))].copy()
        last_date = div_hist["MonthStart"].max() if not div_hist.empty else None
        hist_window = (div_hist[div_hist["MonthStart"] >= last_date - pd.DateOffset(months=exec_h_months - 1)]
                       if last_date is not None else div_hist)
        actual_by_div = hist_window.groupby("series_id")["Units"].sum().rename("Market_Actual_Units")
        existing = existing.merge(actual_by_div, left_on="Division", right_index=True, how="left")
        existing["Market_Actual_Units"] = existing["Market_Actual_Units"].fillna(0)
        existing["Dometic_Actual_Units"] = existing["Market_Actual_Units"] * existing["Attach_Rate"]

        # Headline numbers, ranking, and charts use only TRUSTWORTHY rates --
        # a customer with a handful of market units and a wildly noisy rate
        # would otherwise dominate the ranking despite carrying no real
        # signal, and so would a customer with real volume but an
        # implausible rate (likely an ID-mapping issue). Everything still
        # shows up in the full table below with a flag, so nothing is
        # silently hidden -- it's just kept out of the headline number and
        # Top N chart. Trustworthy_Rate (reliable AND plausible) is used
        # when available; falls back to Reliable_Rate alone for older
        # attach_rates.parquet files that predate the plausibility check.
        if "Trustworthy_Rate" in existing.columns:
            existing_reliable = existing[existing["Trustworthy_Rate"] == True].copy()
        elif "Reliable_Rate" in existing.columns:
            existing_reliable = existing[existing["Reliable_Rate"] == True].copy()
        else:
            existing_reliable = existing
        n_excluded_unreliable = len(existing) - len(existing_reliable)

        # ---- Headline number ----
        total_dometic_forecast = existing_reliable["Dometic_Forecast_Units"].sum()
        total_dometic_actual = existing_reliable["Dometic_Actual_Units"].sum()
        yoy = ((total_dometic_forecast / total_dometic_actual - 1) * 100) if total_dometic_actual > 0 else None

        h1, h2, h3 = st.columns(3)
        h1.metric(f"Projected Dometic units — next {exec_h_months} months",
                  f"{total_dometic_forecast:,.0f}",
                  f"{yoy:+.1f}% vs trailing {exec_h_months}mo" if yoy is not None else None)
        h2.metric(f"Trailing {exec_h_months} months (est.)", f"{total_dometic_actual:,.0f}")
        h3.metric("Existing customers included", f"{len(existing_reliable):,}")

        st.caption(f"Based on {len(existing_reliable)} existing customer(s) with a known Dometic sales "
                   f"relationship and a trustworthy attach rate (enough trailing market volume, and a "
                   f"plausible rate), each projected using their own current attach rate "
                   f"(trailing Dometic units ÷ trailing RV market units) applied to the "
                   f"RV market forecast for that manufacturer.")
        if n_excluded_unreliable:
            st.caption(f"ℹ️ {n_excluded_unreliable} additional customer(s) have a rate that needs review "
                       f"(too little trailing market volume, or implausibly high despite enough volume) "
                       f"— excluded from the totals and ranking above, but visible in the full table below.")

        st.write("")

        # ---- Top N chart ----
        top = existing_reliable.head(exec_top_n)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=top["Division"], x=top["Dometic_Actual_Units"], orientation="h",
            name=f"Trailing {exec_h_months}mo (est.)", marker_color=ACTUAL_COLOR,
            hovertemplate="%{y}<br>%{x:,.0f} units<extra></extra>",
        ))
        fig.add_trace(go.Bar(
            y=top["Division"], x=top["Dometic_Forecast_Units"], orientation="h",
            name=f"Next {exec_h_months}mo (forecast)", marker_color=FORECAST_COLOR,
            hovertemplate="%{y}<br>%{x:,.0f} units<extra></extra>",
        ))
        fig.update_layout(
            height=max(320, 34 * len(top)), margin=dict(l=10, r=10, t=10, b=10),
            barmode="group", xaxis_title="Dometic units (est.)", yaxis_title="",
            yaxis=dict(autorange="reversed"),
            legend=dict(orientation="h", y=1.08, x=0), plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

        # ---- Product Area breakdown ----
        if area_mix_df is not None and not area_mix_df.empty:
            st.write("")
            st.markdown("**Projected Dometic units by Product Area**")

            by_area = existing_reliable[["Division", "ParentCustomerNumber", "Dometic_Forecast_Units",
                                "Dometic_Actual_Units"]].merge(
                area_mix_df, on="ParentCustomerNumber", how="inner")
            by_area["Forecast_By_Area"] = by_area["Dometic_Forecast_Units"] * by_area["Area_Share"]
            by_area["Actual_By_Area"] = by_area["Dometic_Actual_Units"] * by_area["Area_Share"]

            n_covered = by_area["Division"].nunique()
            if n_covered < len(existing_reliable):
                st.caption(f"Product Area split available for {n_covered} of {len(existing_reliable)} "
                           f"existing customers — the rest have no Product Area detail on file "
                           f"and are excluded from this breakdown (still included in the totals above).")

            area_totals = (by_area.groupby("ProductArea")
                          .agg(Forecast=("Forecast_By_Area", "sum"),
                               Trailing=("Actual_By_Area", "sum"))
                          .sort_values("Forecast", ascending=False).reset_index())

            fig_area = go.Figure()
            fig_area.add_trace(go.Bar(
                y=area_totals["ProductArea"], x=area_totals["Trailing"], orientation="h",
                name=f"Trailing {exec_h_months}mo (est.)", marker_color=ACTUAL_COLOR,
                hovertemplate="%{y}<br>%{x:,.0f} units<extra></extra>",
            ))
            fig_area.add_trace(go.Bar(
                y=area_totals["ProductArea"], x=area_totals["Forecast"], orientation="h",
                name=f"Next {exec_h_months}mo (forecast)", marker_color=FORECAST_COLOR,
                hovertemplate="%{y}<br>%{x:,.0f} units<extra></extra>",
            ))
            fig_area.update_layout(
                height=max(280, 34 * len(area_totals)), margin=dict(l=10, r=10, t=10, b=10),
                barmode="group", xaxis_title="Dometic units (est.)", yaxis_title="",
                yaxis=dict(autorange="reversed"),
                legend=dict(orientation="h", y=1.08, x=0), plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_area, use_container_width=True)
            st.caption("Split using each customer's trailing mix of Product Areas — the RV market "
                       "forecast has no Product Area dimension itself, so this assumes each "
                       "customer's category mix holds roughly steady going forward.")

            with st.expander("📋 By customer and Product Area"):
                customer_options = ["All customers"] + sorted(by_area["Division"].unique())
                selected_customer = st.selectbox("Filter to a customer", customer_options, key="area_customer_filter")

                detail_tbl = by_area[["Division", "ProductArea", "Area_Share", "Forecast_By_Area"]].copy()
                if selected_customer != "All customers":
                    detail_tbl = detail_tbl[detail_tbl["Division"] == selected_customer]
                    detail_tbl = detail_tbl.sort_values("Forecast_By_Area", ascending=False)
                else:
                    detail_tbl = detail_tbl.sort_values(["Division", "Forecast_By_Area"], ascending=[True, False])

                detail_tbl.columns = ["Customer", "Product Area", "Share of Customer's Total", "Forecast (units)"]
                detail_tbl["Share of Customer's Total"] = detail_tbl["Share of Customer's Total"].map("{:.1%}".format)
                detail_tbl["Forecast (units)"] = detail_tbl["Forecast (units)"].map("{:,.0f}".format)
                st.dataframe(detail_tbl, hide_index=True, use_container_width=True)

                if selected_customer == "All customers":
                    st.caption("Tip: use the filter above to jump straight to one customer's breakdown "
                               "instead of scrolling.")

        st.write("")
        with st.expander("📋 Full customer table"):
            tbl_cols = ["Division", "Attach_Rate", "Market_Forecast_Units",
                       "Dometic_Forecast_Units", "Dometic_Actual_Units"]
            col_names = ["Customer", "Attach Rate", "Market Forecast (units)",
                        "Dometic Forecast (units)", f"Trailing {exec_h_months}mo Dometic (est.)"]
            if "Siblings_Sharing_Parent" in existing.columns:
                tbl_cols.append("Siblings_Sharing_Parent")
                col_names.append("Sibling brands sharing this rate")
            if "Trustworthy_Rate" in existing.columns:
                tbl_cols.append("Trustworthy_Rate")
                col_names.append("Trustworthy rate?")
            elif "Reliable_Rate" in existing.columns:
                tbl_cols.append("Reliable_Rate")
                col_names.append("Reliable rate?")
            if "Vs_Peer_Ratio" in existing.columns:
                tbl_cols.append("Vs_Peer_Ratio")
                col_names.append("Vs. peer segment median")
            tbl = existing[tbl_cols].copy()
            tbl.columns = col_names
            tbl["Attach Rate"] = tbl["Attach Rate"].map("{:.1%}".format)
            for c in ["Market Forecast (units)", "Dometic Forecast (units)",
                     f"Trailing {exec_h_months}mo Dometic (est.)"]:
                tbl[c] = tbl[c].map("{:,.0f}".format)
            trust_col = "Trustworthy rate?" if "Trustworthy rate?" in tbl.columns else "Reliable rate?"
            if trust_col in tbl.columns:
                tbl[trust_col] = tbl[trust_col].map({True: "✓", False: "⚠️ needs review"})
                tbl = tbl.sort_values(trust_col, ascending=False)
            if "Vs. peer segment median" in tbl.columns:
                tbl["Vs. peer segment median"] = tbl["Vs. peer segment median"].map(
                    lambda v: f"{v:.0%}" if pd.notna(v) else "–")
            st.dataframe(tbl, hide_index=True, use_container_width=True)
            if "Siblings_Sharing_Parent" in existing.columns and (existing["Siblings_Sharing_Parent"] > 1).any():
                st.caption("ℹ️ Some customers share a consolidated parent account with other brands -- "
                           "Dometic's sales data only has one combined number per parent, so those "
                           "brands share the same attach rate rather than each having their own.")
            if "Plausible_Rate" in existing.columns and (existing["Plausible_Rate"] == False).any():
                st.caption("⚠️ 'Needs review' covers two different issues: too little trailing market "
                           "volume for the rate to be statistically trustworthy (a small denominator "
                           "can turn modest Dometic volume into an extreme percentage), OR enough "
                           "volume but a rate above the plausibility ceiling anyway -- usually a sign "
                           "of an ID-mapping issue rather than a real attach-rate story. Both are "
                           "excluded from the headline totals and charts above.")
            elif "Reliable_Rate" in existing.columns and (existing["Reliable_Rate"] == False).any():
                st.caption("⚠️ Customers flagged 'needs review' have too little trailing RV market volume "
                           "for their attach rate to be statistically trustworthy. Excluded from the "
                           "headline totals and charts above.")

        if "Vs_Peer_Ratio" in existing.columns:
            upsell = existing[(existing.get("Trustworthy_Rate", existing.get("Reliable_Rate", True)) == True) &
                              (existing["Vs_Peer_Ratio"] < 0.7)].sort_values("Vs_Peer_Ratio")
            if not upsell.empty:
                with st.expander(f"🎯 Upsell candidates ({len(upsell)}) — below peer segment median"):
                    up_tbl = upsell[["Division", "Attach_Rate", "Peer_Median_Rate", "Vs_Peer_Ratio"]].copy()
                    up_tbl.columns = ["Customer", "Their Attach Rate", "Peer Segment Median", "Vs. Peer"]
                    up_tbl["Their Attach Rate"] = up_tbl["Their Attach Rate"].map("{:.1%}".format)
                    up_tbl["Peer Segment Median"] = up_tbl["Peer Segment Median"].map("{:.1%}".format)
                    up_tbl["Vs. Peer"] = up_tbl["Vs. Peer"].map("{:.0%}".format)
                    st.dataframe(up_tbl, hide_index=True, use_container_width=True)
                    st.caption("Customers buying meaningfully less Dometic content per RV than peers "
                               "in the same sales segment — concrete candidates for account outreach.")

        with st.expander("⬇️ Export to Excel"):
            def build_excel_exec(existing_df: pd.DataFrame):
                buf = io.BytesIO()
                export_cols = ["Division", "ParentCustomerNumber", "Attach_Rate",
                              "Market_Forecast_Units", "Dometic_Forecast_Units", "Dometic_Actual_Units"]
                if "Siblings_Sharing_Parent" in existing_df.columns:
                    export_cols.append("Siblings_Sharing_Parent")
                export_cols = [c for c in export_cols if c in existing_df.columns]
                export_tbl = existing_df[export_cols].copy()
                with pd.ExcelWriter(buf, engine="openpyxl") as w:
                    export_tbl.to_excel(w, sheet_name="Dometic_Projection", index=False)
                buf.seek(0)
                return buf.read()

            xl_bytes = build_excel_exec(existing)
            st.download_button("Download", data=xl_bytes,
                              file_name=f"Dometic_Sales_Projection_{exec_h_months}mo.xlsx",
                              mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                              use_container_width=True, key="dl_exec_summary")
