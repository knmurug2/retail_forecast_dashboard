"""
compute_attach_rate_forecast.py
=================================
Combines Dometic's actual sales-into-OEMs with the RV retail market forecast
to project Dometic sales volume, using each manufacturer's attach rate:

    attach_rate(Division) = trailing Dometic units / trailing RV market units
    Dometic forecast = market forecast x attach_rate(Division)

Includes:
  - Shared-parent-aware denominator (multiple brands under one consolidated
    D365 account share one rate, correctly scaled to their combined volume)
  - Reliable_Rate flag: excludes rates computed on too little market volume
    to trust (a tiny denominator turns modest Dometic volume into an
    extreme percentage)
  - Plausible_Rate flag: NEW -- catches the case the volume floor can't:
    a customer with real, substantial volume whose rate is still
    implausibly high, most likely an ID-mapping issue rather than a real
    attach-rate story
  - Peer_Median_Rate / Vs_Peer_Ratio: NEW -- benchmarks each customer's
    rate against others in the same sales segment, surfacing upsell
    candidates (below-peer attach rate) and outliers worth a second look
  - Product Area mix, applied as a split against each customer's projection

Usage:
    python compute_attach_rate_forecast.py
    python compute_attach_rate_forecast.py --trailing-months 12
    python compute_attach_rate_forecast.py --max-plausible-rate 3.0
"""

import os
import sys
import argparse
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass

DATA_PATH = os.environ.get(
    "DATA_PATH",
    r"C:\Users\Karmur\OneDrive - Dometic Group\RV_Cust_Data.xlsx"
)
DOMETIC_SALES_PATH = os.environ.get(
    "DOMETIC_SALES_OUT",
    r"C:\Users\Karmur\OneDrive - Dometic Group\PY Project Output\DometicSales\dometic_sales_by_parent.xlsx"
)
PARQUET_DIR = os.environ.get(
    "PARQUET_DIR",
    r"C:\Users\Karmur\OneDrive - Dometic Group\PY Project Output\RetailForecast\parquet"
)
DEFAULT_OUT = os.environ.get(
    "ATTACH_RATE_OUT",
    r"C:\Users\Karmur\OneDrive - Dometic Group\PY Project Output\RetailForecast\Dometic_Sales_Projection.xlsx"
)

TRAILING_MONTHS_DEFAULT = 12

# Below this much trailing RV market volume, an attach rate is too
# statistically unstable to trust -- same principle as the THIN tier in the
# forecast engine.
MIN_MARKET_VOLUME_FOR_RATE = 100

# A SECOND, independent check the volume floor can't catch: a customer with
# plenty of market volume can still show an implausible rate (e.g. 2,000%+)
# -- that's not a small-sample noise problem, it's more likely a genuine
# ID-mapping error somewhere upstream, or a real business quirk worth a
# manual look either way. Flagged separately from Reliable_Rate so the two
# failure modes (too little data vs. too implausible regardless of data)
# stay distinguishable.
MAX_PLAUSIBLE_RATE = 3.0  # 300% -- generous enough to allow multi-component
                           # attach (several Dometic parts per RV) without
                           # flagging normal cases


def clean_id(series: pd.Series) -> pd.Series:
    """
    Normalizes an ID column for joining across sources. Guards against a
    numeric-looking ID column getting read from Excel as float64 (producing
    "12345.0" instead of "12345"), which would silently fail every match
    rather than raising an error.
    """
    s = series.astype(str).str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)
    return s


def build_division_parent_map(rv_cust_data_path: str) -> pd.DataFrame:
    """
    Direct ID mapping: Division -> parent account ID, read straight from
    RV_Cust_Data.xlsx. A Division should map to exactly one parent ID; if
    it shows more than one, the most frequent is used and the conflict is
    flagged.
    """
    df = pd.read_excel(rv_cust_data_path, sheet_name=0, header=0, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]

    PARENT_COL_ALIASES = ["ParentCustomerNumber", "ParentAccountNumber", "ParentAccountId"]
    col_lookup = {c.lower(): c for c in df.columns}
    parent_col = next((col_lookup[a.lower()] for a in PARENT_COL_ALIASES if a.lower() in col_lookup), None)

    if parent_col is None:
        raise ValueError(
            f"None of {PARENT_COL_ALIASES} found in RV_Cust_Data.xlsx. "
            f"Columns present: {df.columns.tolist()}"
        )
    if parent_col != "ParentCustomerNumber":
        print(f"  (Using '{parent_col}' as the parent account column)")

    if "Division" not in df.columns:
        raise ValueError("Division column not found in RV_Cust_Data.xlsx.")

    df["Division"] = df["Division"].astype(str).str.strip()
    df["ParentCustomerNumber"] = clean_id(df[parent_col])
    df = df[df["ParentCustomerNumber"].notna() & (df["ParentCustomerNumber"] != "") &
            (df["ParentCustomerNumber"].str.lower() != "nan")]

    counts = df.groupby(["Division", "ParentCustomerNumber"]).size().reset_index(name="RowCount")
    n_ids_per_division = counts.groupby("Division")["ParentCustomerNumber"].nunique()
    conflicted = n_ids_per_division[n_ids_per_division > 1].index.tolist()

    if conflicted:
        print(f"  WARNING: {len(conflicted)} Division(s) have more than one "
              f"ParentCustomerNumber in the source data -- using the most "
              f"frequent one for each. Worth checking: {conflicted[:10]}"
              f"{' ...' if len(conflicted) > 10 else ''}")

    best = (counts.sort_values("RowCount", ascending=False)
            .drop_duplicates("Division", keep="first")
            .rename(columns={"RowCount": "Match_Row_Count"}))
    best["Had_Conflict"] = best["Division"].isin(conflicted)

    return best[["Division", "ParentCustomerNumber", "Match_Row_Count", "Had_Conflict"]]


def compute_attach_rates(dometic_monthly: pd.DataFrame, market_history: pd.DataFrame,
                         division_map: pd.DataFrame, trailing_months: int,
                         min_market_volume: float = MIN_MARKET_VOLUME_FOR_RATE,
                         max_plausible_rate: float = MAX_PLAUSIBLE_RATE) -> pd.DataFrame:
    """
    attach_rate(ParentCustomerNumber) = trailing Dometic SalesQty / trailing
    RV market Units, where the market side is the SUM across every Division
    sharing this parent (see module docstring for why).
    """
    last_date = market_history["MonthStart"].max()
    cutoff = last_date - pd.DateOffset(months=trailing_months - 1)

    market_trailing = (market_history[(market_history["Grain"] == "Division") &
                                      (market_history["MonthStart"] >= cutoff)]
                       .groupby("series_id")["Units"].sum())

    dometic_trailing = (dometic_monthly[dometic_monthly["SalesMonth"] >= cutoff]
                        .groupby("ParentCustomerNumber")["SalesQty"].sum())

    known_parent_ids = set(dometic_monthly["ParentCustomerNumber"].unique())

    dm = division_map.copy()
    dm["Market_Units"] = dm["Division"].map(market_trailing).fillna(0)

    parent_market_totals = dm.groupby("ParentCustomerNumber")["Market_Units"].sum()
    siblings_count = dm.groupby("ParentCustomerNumber")["Division"].nunique()

    rows = []
    for _, r in dm.iterrows():
        div = r["Division"]
        pid = r["ParentCustomerNumber"]
        parent_market_units = parent_market_totals.get(pid, 0)
        n_siblings = int(siblings_count.get(pid, 1))

        if pid not in known_parent_ids:
            dometic_units = None
            rate = None
        else:
            dometic_units = dometic_trailing.get(pid, 0)
            rate = (dometic_units / parent_market_units) if parent_market_units > 0 else None

        reliable = bool(rate is not None and parent_market_units >= min_market_volume)
        plausible = bool(rate is not None and rate <= max_plausible_rate)

        rows.append({
            "Division": div, "ParentCustomerNumber": pid,
            "Trailing_Market_Units": r["Market_Units"],
            "Trailing_Parent_Market_Units": parent_market_units,
            "Siblings_Sharing_Parent": n_siblings,
            "Trailing_Dometic_Units": dometic_units,
            "Attach_Rate": rate, "Had_Conflict": r["Had_Conflict"],
            "Has_Dometic_Relationship": pid in known_parent_ids,
            "Reliable_Rate": reliable,
            "Plausible_Rate": plausible,
            "Trustworthy_Rate": reliable and plausible,
        })
    return pd.DataFrame(rows)


def add_peer_benchmark(attach_rates: pd.DataFrame, segment_map: pd.DataFrame) -> pd.DataFrame:
    """
    Benchmarks each customer's attach rate against the median rate of other
    customers in the same sales segment. Only Trustworthy_Rate rows
    (reliable AND plausible) count toward the peer median, so the benchmark
    itself isn't skewed by the same noisy outliers this whole system is
    trying to flag. A customer sitting well below their peer median is a
    concrete upsell candidate; well above might mean they're a genuinely
    stronger relationship, or worth a sanity check if the gap is extreme.
    """
    if segment_map.empty or "SalesSegmentID" not in segment_map.columns:
        attach_rates["SalesSegmentID"] = None
        attach_rates["Peer_Median_Rate"] = None
        attach_rates["Vs_Peer_Ratio"] = None
        return attach_rates

    merged = attach_rates.merge(
        segment_map[["ParentCustomerNumber", "SalesSegmentID"]],
        on="ParentCustomerNumber", how="left"
    )

    trustworthy = merged[merged["Trustworthy_Rate"] == True]
    peer_median = trustworthy.groupby("SalesSegmentID")["Attach_Rate"].median()

    merged["Peer_Median_Rate"] = merged["SalesSegmentID"].map(peer_median)
    merged["Vs_Peer_Ratio"] = merged.apply(
        lambda r: (r["Attach_Rate"] / r["Peer_Median_Rate"])
        if pd.notna(r["Attach_Rate"]) and pd.notna(r["Peer_Median_Rate"]) and r["Peer_Median_Rate"] > 0
        else None, axis=1)

    return merged


def compute_area_mix(dometic_monthly_by_area: pd.DataFrame, trailing_months: int) -> pd.DataFrame:
    """Each ParentCustomerNumber's trailing mix of Product Areas."""
    if dometic_monthly_by_area is None or dometic_monthly_by_area.empty:
        return pd.DataFrame()

    last_date = dometic_monthly_by_area["SalesMonth"].max()
    cutoff = last_date - pd.DateOffset(months=trailing_months - 1)
    trailing = dometic_monthly_by_area[dometic_monthly_by_area["SalesMonth"] >= cutoff]

    by_parent_area = trailing.groupby(["ParentCustomerNumber", "ProductArea"])["SalesQty"].sum()
    by_parent_total = trailing.groupby("ParentCustomerNumber")["SalesQty"].sum()

    mix = (by_parent_area / by_parent_total).reset_index().rename(columns={"SalesQty": "Area_Share"})
    return mix


def project_by_area(existing_customers: pd.DataFrame, area_mix: pd.DataFrame) -> pd.DataFrame:
    """Splits each customer's projected Dometic total across Product Areas."""
    if area_mix.empty:
        return pd.DataFrame()
    merged = existing_customers.merge(area_mix, on="ParentCustomerNumber", how="inner")
    merged["Dometic_Forecast_Units_By_Area"] = merged["Dometic_Forecast_Units"] * merged["Area_Share"]
    return merged


def project_dometic_sales(market_forecast: pd.DataFrame, attach_rates: pd.DataFrame) -> pd.DataFrame:
    """Applies each Division's attach rate to its market forecast."""
    rate_map = dict(zip(attach_rates["Division"], attach_rates["Attach_Rate"]))

    out = market_forecast.copy()
    out["Division_Match"] = out["series_id"]
    is_combo = out["Grain"] == "Division_Type"
    out.loc[is_combo, "Division_Match"] = out.loc[is_combo, "series_id"].str.split(
        " | ", n=1, expand=False, regex=False).str[0]

    out["Attach_Rate"] = out["Division_Match"].map(rate_map)
    out["Dometic_Forecast_Units"] = out["Forecast_Units"] * out["Attach_Rate"]

    for col in ["P10_Units", "P90_Units"]:
        if col in out.columns:
            out[f"Dometic_{col}"] = out[col] * out["Attach_Rate"]

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rv-cust-data", default=DATA_PATH)
    parser.add_argument("--dometic-sales", default=DOMETIC_SALES_PATH)
    parser.add_argument("--parquet-dir", default=PARQUET_DIR)
    parser.add_argument("--trailing-months", type=int, default=TRAILING_MONTHS_DEFAULT)
    parser.add_argument("--min-market-volume", type=float, default=MIN_MARKET_VOLUME_FOR_RATE,
                        help="Minimum trailing RV market units required before an attach rate "
                             "is flagged reliable")
    parser.add_argument("--max-plausible-rate", type=float, default=MAX_PLAUSIBLE_RATE,
                        help="Attach rates above this (as a decimal, e.g. 3.0 = 300%%) are "
                             "flagged implausible regardless of volume")
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    print("=" * 60)
    print("Dometic Attach-Rate Sales Projection")
    print("=" * 60)

    print(f"\nBuilding Division -> ParentCustomerNumber map from: {args.rv_cust_data}")
    division_map = build_division_parent_map(args.rv_cust_data)
    print(f"  {len(division_map)} Division(s) mapped, "
          f"{division_map['Had_Conflict'].sum()} with a conflicting ID (see warning above)")

    print(f"\nLoading Dometic sales: {args.dometic_sales}")
    dometic_monthly = pd.read_excel(args.dometic_sales, sheet_name="Monthly_By_Parent")
    dometic_monthly["SalesMonth"] = pd.to_datetime(dometic_monthly["SalesMonth"])
    dometic_monthly["ParentCustomerNumber"] = clean_id(dometic_monthly["ParentCustomerNumber"])

    print(f"Loading RV market forecast: {args.parquet_dir}")
    market_history = pd.read_parquet(os.path.join(args.parquet_dir, "history.parquet"))
    market_forecast = pd.read_parquet(os.path.join(args.parquet_dir, "forecast.parquet"))
    market_history["MonthStart"] = pd.to_datetime(market_history["MonthStart"])
    market_forecast["MonthStart"] = pd.to_datetime(market_forecast["MonthStart"])
    market_forecast = market_forecast[market_forecast["Is_Selected_Model"] == True]

    matched_ids = set(division_map["ParentCustomerNumber"]) & set(dometic_monthly["ParentCustomerNumber"])
    print(f"\n  {len(matched_ids)} of {len(division_map)} Division(s) have Dometic sales data "
          f"for their ParentCustomerNumber")

    print(f"\nComputing attach rates (trailing {args.trailing_months} months)...")
    attach_rates = compute_attach_rates(dometic_monthly, market_history, division_map,
                                        args.trailing_months, args.min_market_volume,
                                        args.max_plausible_rate)

    valid_rates = attach_rates.dropna(subset=["Attach_Rate"])
    trustworthy_rates = valid_rates[valid_rates["Trustworthy_Rate"] == True]
    n_unreliable = (valid_rates["Reliable_Rate"] == False).sum()
    n_implausible = (valid_rates["Plausible_Rate"] == False).sum()
    print(f"  {len(valid_rates)} Division(s) with a computable attach rate:")
    print(f"    {len(trustworthy_rates)} trustworthy (reliable AND plausible)")
    print(f"    {n_unreliable} below the {args.min_market_volume:.0f}-unit volume floor")
    print(f"    {n_implausible} above the {args.max_plausible_rate:.0%} plausibility ceiling")
    if len(trustworthy_rates):
        print(f"  Rate range (trustworthy only): {trustworthy_rates['Attach_Rate'].min():.2f} "
              f"to {trustworthy_rates['Attach_Rate'].max():.2f}")

    # Peer benchmarking -- needs the segment map from pull_dometic_sales.py
    segment_map = pd.DataFrame()
    try:
        segment_map = pd.read_excel(args.dometic_sales, sheet_name="Parent_Segment_Map")
        segment_map["ParentCustomerNumber"] = clean_id(segment_map["ParentCustomerNumber"])
        n_segs = segment_map['SalesSegmentID'].nunique() if 'SalesSegmentID' in segment_map.columns else 0
        print(f"\nComputing peer benchmarks ({n_segs} segments)...")
        attach_rates = add_peer_benchmark(attach_rates, segment_map)
        below_peer = attach_rates[(attach_rates["Trustworthy_Rate"] == True) &
                                  (attach_rates["Vs_Peer_Ratio"] < 0.7)]
        print(f"  {len(below_peer)} customer(s) sitting notably below their peer segment median "
              f"(Vs_Peer_Ratio < 0.7) -- potential upsell candidates")
    except Exception as e:
        print(f"\nNo segment data available for peer benchmarking ({e}) -- skipping.")
        attach_rates["SalesSegmentID"] = None
        attach_rates["Peer_Median_Rate"] = None
        attach_rates["Vs_Peer_Ratio"] = None

    # Product Area mix -- optional
    area_mix = pd.DataFrame()
    try:
        dometic_by_area = pd.read_excel(args.dometic_sales, sheet_name="Monthly_By_Parent_Area")
        dometic_by_area["SalesMonth"] = pd.to_datetime(dometic_by_area["SalesMonth"])
        dometic_by_area["ParentCustomerNumber"] = clean_id(dometic_by_area["ParentCustomerNumber"])
        print(f"\nComputing Product Area mix (trailing {args.trailing_months} months)...")
        area_mix = compute_area_mix(dometic_by_area, args.trailing_months)
        if not area_mix.empty:
            print(f"  {area_mix['ParentCustomerNumber'].nunique()} parent account(s), "
                  f"{area_mix['ProductArea'].nunique()} Product Area(s)")
    except Exception as e:
        print(f"\nNo Product Area breakdown available ({e}) -- skipping.")

    print("\nProjecting Dometic sales onto the market forecast...")
    projection = project_dometic_sales(market_forecast, attach_rates)

    n_projected = projection["Dometic_Forecast_Units"].notna().sum()
    print(f"  {n_projected:,} of {len(projection):,} forecast rows got a Dometic projection")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with pd.ExcelWriter(args.out, engine="openpyxl") as writer:
        summary = (projection.dropna(subset=["Dometic_Forecast_Units"])
                  .groupby(["Grain", "series_id"])["Dometic_Forecast_Units"].sum()
                  .reset_index().sort_values("Dometic_Forecast_Units", ascending=False))
        summary.to_excel(writer, sheet_name="Summary_By_Series", index=False)

        detail_cols = ["Grain", "series_id", "MonthStart", "Forecast_Units",
                       "Attach_Rate", "Dometic_Forecast_Units",
                       "Dometic_P10_Units", "Dometic_P90_Units"]
        detail_cols = [c for c in detail_cols if c in projection.columns]
        projection[detail_cols].to_excel(writer, sheet_name="Detail", index=False)

        attach_rates.to_excel(writer, sheet_name="Attach_Rates", index=False)
        division_map.to_excel(writer, sheet_name="Division_Parent_Map", index=False)
        if not area_mix.empty:
            area_mix.to_excel(writer, sheet_name="Product_Area_Mix", index=False)

        if "Vs_Peer_Ratio" in attach_rates.columns:
            upsell = (attach_rates[(attach_rates["Trustworthy_Rate"] == True) &
                                   (attach_rates["Vs_Peer_Ratio"] < 0.7)]
                     .sort_values("Vs_Peer_Ratio"))
            if not upsell.empty:
                upsell.to_excel(writer, sheet_name="Upsell_Candidates", index=False)

        unmatched = division_map[~division_map["ParentCustomerNumber"].isin(dometic_monthly["ParentCustomerNumber"])]
        unmatched.to_excel(writer, sheet_name="No_Dometic_Sales_Match", index=False)

    print(f"\nSaved: {args.out}")

    try:
        attach_rates_path = os.path.join(args.parquet_dir, "attach_rates.parquet")
        attach_rates.to_parquet(attach_rates_path, index=False, engine="pyarrow")
        print(f"Saved: {attach_rates_path}  (dashboard reads this for the Dometic-units view)")
    except Exception as e:
        print(f"Could not save attach_rates.parquet ({e})")

    if not area_mix.empty:
        try:
            area_mix_path = os.path.join(args.parquet_dir, "area_mix.parquet")
            area_mix.to_parquet(area_mix_path, index=False, engine="pyarrow")
            print(f"Saved: {area_mix_path}  (dashboard reads this for the Product Area breakdown)")
        except Exception as e:
            print(f"Could not save area_mix.parquet ({e})")


if __name__ == "__main__":
    main()
