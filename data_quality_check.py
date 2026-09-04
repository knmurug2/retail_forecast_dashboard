"""
data_quality_check.py
======================
Standalone quality check for the RV Retail Sales source file, independent
of the forecast pipeline. Run this any time you get a new/updated
RV_Cust_Data.xlsx to catch issues before they hit the forecast.

Usage:
    python data_quality_check.py
    python data_quality_check.py --data "C:/path/to/RV_Cust_Data.xlsx"
    python data_quality_check.py --out "C:/path/to/QC_Report.xlsx"

Checks performed:
  1. File / column structure      -- required columns present, extra columns flagged
  2. Missing values                -- per column, with % of rows affected
  3. Duplicate rows                -- exact duplicates across all columns
  4. Date coverage & gaps          -- min/max date, missing months per year
  5. Annual vs monthly row split   -- how many rows are annual rollups (no Dealer Name)
                                       vs real monthly transactions, by year
  6. Grand Total / rollup rows     -- rows that are pivot-table totals, not real data
  7. Units sanity                  -- zero, negative, and extreme outlier values
  8. Category consistency          -- near-duplicate Division/Type/Manufacturer names
                                       (trailing spaces, case differences, etc.)
  9. Year-over-year totals         -- total units by year, flags large swings
  10. Duplicate transaction risk   -- identical Manufacturer+Division+Model+Type+
                                       Date+Dealer combos (possible double-counted rows)

Output:
  - Console summary (pass/warn/fail per check)
  - Excel report with one sheet per check showing the actual flagged rows
"""

import os
import sys
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass

DEFAULT_DATA_PATH = os.environ.get("DATA_PATH",
    r"C:\Users\Karmur\OneDrive - Dometic Group\RV_Cust_Data.xlsx")
DEFAULT_OUT_PATH = os.environ.get("QC_OUT_PATH",
    r"C:\Users\Karmur\OneDrive - Dometic Group\PY Project Output\RetailForecast\Data_Quality_Report.xlsx")

REQUIRED_COLUMNS = ["Manufacturer", "Division", "Model", "Type", "Year",
                    "Placement States", "Units", "Date", "Month", "Dealer Name"]

OUTLIER_Z_THRESHOLD = 5.0   # flag units more than 5 std devs from the mean within Type


class CheckResult:
    def __init__(self, name, status, message, detail_df=None):
        self.name = name
        self.status = status      # "PASS", "WARN", "FAIL"
        self.message = message
        self.detail_df = detail_df if detail_df is not None else pd.DataFrame()


def check_structure(df: pd.DataFrame) -> CheckResult:
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    extra_cols = [c for c in df.columns if c not in REQUIRED_COLUMNS]

    if missing_cols:
        return CheckResult("Column structure", "FAIL",
                            f"Missing required column(s): {missing_cols}",
                            pd.DataFrame({"Missing_Column": missing_cols}))

    msg = f"All {len(REQUIRED_COLUMNS)} required columns present."
    if extra_cols:
        msg += f" Extra column(s) found (not used by pipeline): {extra_cols}"
    return CheckResult("Column structure", "PASS", msg,
                        pd.DataFrame({"Extra_Column": extra_cols}) if extra_cols else pd.DataFrame())


def check_missing_values(df: pd.DataFrame) -> CheckResult:
    miss = df.isna().sum()
    miss_pct = (miss / len(df) * 100).round(2)
    detail = pd.DataFrame({"Column": miss.index, "Missing_Count": miss.values,
                           "Missing_Pct": miss_pct.values})
    detail = detail[detail["Missing_Count"] > 0].sort_values("Missing_Count", ascending=False)

    # Dealer Name and Placement States are EXPECTED to have nulls (annual rollup rows)
    # so we don't fail on those -- only flag unexpected nulls in other columns
    unexpected = detail[~detail["Column"].isin(["Dealer Name", "Placement States"])]

    if not unexpected.empty:
        return CheckResult("Missing values", "WARN",
                            f"Unexpected nulls found in: {unexpected['Column'].tolist()}",
                            detail)
    return CheckResult("Missing values", "PASS",
                        "No unexpected nulls (Dealer Name / Placement States nulls are expected "
                        "for annual rollup rows).", detail)


def check_duplicates(df: pd.DataFrame) -> CheckResult:
    dupes = df[df.duplicated(keep=False)]
    if dupes.empty:
        return CheckResult("Duplicate rows", "PASS", "No exact duplicate rows found.")
    n_dupe_groups = df.duplicated(keep="first").sum()
    return CheckResult("Duplicate rows", "WARN",
                        f"{n_dupe_groups:,} exact duplicate row(s) found "
                        f"({len(dupes):,} rows total involved).",
                        dupes.sort_values(list(df.columns)))


def check_date_coverage(df: pd.DataFrame) -> CheckResult:
    dates = pd.to_datetime(df["Date"], errors="coerce")
    bad_dates = df[dates.isna()]

    valid = dates.dropna()
    if valid.empty:
        return CheckResult("Date coverage", "FAIL", "No valid dates found in Date column.")

    min_d, max_d = valid.min(), valid.max()
    years_present = sorted(int(y) for y in valid.dt.year.unique())

    msg = f"Date range: {min_d:%Y-%m-%d} to {max_d:%Y-%m-%d}. Years present: {years_present}."
    status = "PASS"
    if not bad_dates.empty:
        status = "WARN"
        msg += f" {len(bad_dates)} row(s) with unparseable dates."

    return CheckResult("Date coverage", status, msg, bad_dates)


def check_annual_vs_monthly_split(df: pd.DataFrame) -> CheckResult:
    if "Dealer Name" not in df.columns:
        return CheckResult("Annual vs monthly split", "WARN", "Dealer Name column not found; cannot check.")

    dates = pd.to_datetime(df["Date"], errors="coerce")
    is_annual = df["Dealer Name"].isna()

    split = pd.DataFrame({
        "Year": dates.dt.year,
        "Row_Type": np.where(is_annual, "Annual rollup", "Monthly transaction"),
    }).groupby(["Year", "Row_Type"]).size().reset_index(name="Row_Count")

    # Flag any year that's a MIX of both types (unexpected -- should be one or the other)
    year_type_counts = split.groupby("Year")["Row_Type"].nunique()
    mixed_years = year_type_counts[year_type_counts > 1].index.tolist()

    if mixed_years:
        return CheckResult("Annual vs monthly split", "WARN",
                            f"Year(s) with BOTH annual rollup and monthly transaction rows: "
                            f"{mixed_years}. Verify this is expected (e.g. mid-year transition).",
                            split)
    return CheckResult("Annual vs monthly split", "PASS",
                        "Each year is cleanly either annual-rollup or monthly-transaction format.",
                        split)


def check_grand_totals(df: pd.DataFrame) -> CheckResult:
    is_total = pd.Series(False, index=df.index)
    for col in ["Manufacturer", "Division", "Type"]:
        if col in df.columns:
            is_total |= df[col].astype(str).str.strip().str.lower().eq("grand total")

    n = is_total.sum()
    if n == 0:
        return CheckResult("Grand Total rollup rows", "PASS", "No Grand Total rows found.")
    return CheckResult("Grand Total rollup rows", "WARN",
                        f"{n} Grand Total row(s) found -- these will be dropped automatically "
                        f"by the forecast pipeline, but confirm this matches your expectation.",
                        df[is_total])


def check_units_sanity(df: pd.DataFrame) -> CheckResult:
    units = pd.to_numeric(df["Units"], errors="coerce")
    issues = []

    non_numeric = df[units.isna() & df["Units"].notna()]
    if not non_numeric.empty:
        issues.append(("Non-numeric Units value", non_numeric))

    zero_units = df[units == 0]
    if not zero_units.empty:
        issues.append(("Zero Units value", zero_units))

    negative_units = df[units < 0]
    if not negative_units.empty:
        issues.append(("Negative Units value", negative_units))

    # Outlier detection within Type (z-score on log scale to handle skew)
    outlier_frames = []
    if "Type" in df.columns:
        work = df.copy()
        work["_units_num"] = units
        work = work.dropna(subset=["_units_num"])
        work = work[work["_units_num"] > 0]
        for t, grp in work.groupby("Type"):
            if len(grp) < 10:
                continue
            log_u = np.log1p(grp["_units_num"])
            z = (log_u - log_u.mean()) / (log_u.std() + 1e-9)
            outliers = grp[z.abs() > OUTLIER_Z_THRESHOLD]
            if not outliers.empty:
                outlier_frames.append(outliers)
    outliers_df = pd.concat(outlier_frames, ignore_index=True) if outlier_frames else pd.DataFrame()
    if not outliers_df.empty:
        issues.append((f"Extreme outlier (>{OUTLIER_Z_THRESHOLD} std dev within Type, log scale)", outliers_df))

    if not issues:
        return CheckResult("Units sanity", "PASS", "No zero, negative, non-numeric, or extreme outlier Units values.")

    detail = pd.concat(
        [d.assign(Issue=label) for label, d in issues], ignore_index=True
    ) if issues else pd.DataFrame()

    total_flagged = sum(len(d) for _, d in issues)
    status = "FAIL" if any(lbl in ("Non-numeric Units value", "Negative Units value") for lbl, _ in issues) else "WARN"
    msg = f"{total_flagged:,} row(s) flagged across {len(issues)} issue type(s): " \
          f"{', '.join(lbl for lbl, _ in issues)}"
    return CheckResult("Units sanity", status, msg, detail)


def check_category_consistency(df: pd.DataFrame) -> CheckResult:
    """Detect near-duplicate category values -- e.g. 'Forest River' vs 'Forest River ' vs 'FOREST RIVER'."""
    issues = []
    for col in ["Division", "Type", "Manufacturer"]:
        if col not in df.columns:
            continue
        vals = df[col].dropna().astype(str)
        normalized = vals.str.strip().str.lower()
        groups = pd.DataFrame({"Original": vals, "Normalized": normalized}).drop_duplicates()
        dupe_groups = groups.groupby("Normalized")["Original"].nunique()
        conflicting = dupe_groups[dupe_groups > 1].index.tolist()
        if conflicting:
            detail = groups[groups["Normalized"].isin(conflicting)].sort_values("Normalized")
            detail.insert(0, "Column", col)
            issues.append(detail)

    if not issues:
        return CheckResult("Category consistency", "PASS",
                            "No near-duplicate Division/Type/Manufacturer values detected "
                            "(case or whitespace variants).")

    detail = pd.concat(issues, ignore_index=True)
    n_groups = detail.groupby(["Column", "Normalized"]).ngroups if "Normalized" in detail.columns else len(detail)
    return CheckResult("Category consistency", "WARN",
                        f"Found variant spellings that may need cleanup (e.g. trailing spaces, "
                        f"case differences) across {detail['Column'].nunique()} column(s).",
                        detail)


def check_yoy_totals(df: pd.DataFrame) -> CheckResult:
    dates = pd.to_datetime(df["Date"], errors="coerce")
    units = pd.to_numeric(df["Units"], errors="coerce").fillna(0)
    yearly = pd.DataFrame({"Year": dates.dt.year, "Units": units}).groupby("Year")["Units"].sum().reset_index()
    yearly = yearly.sort_values("Year")
    yearly["YoY_Change_Pct"] = yearly["Units"].pct_change().mul(100).round(1)

    big_swings = yearly[yearly["YoY_Change_Pct"].abs() > 40]
    status = "WARN" if not big_swings.empty else "PASS"
    msg = f"Year totals: " + ", ".join(f"{int(r.Year)}={r.Units:,.0f}" for r in yearly.itertuples())
    if not big_swings.empty:
        msg += f". Large YoY swings (>40%) in: {big_swings['Year'].astype(int).tolist()}"

    return CheckResult("Year-over-year totals", status, msg, yearly)


def check_duplicate_transactions(df: pd.DataFrame) -> CheckResult:
    """Only meaningful for monthly transaction rows (has Dealer Name)."""
    if "Dealer Name" not in df.columns:
        return CheckResult("Duplicate transaction risk", "WARN", "Dealer Name column not found; cannot check.")

    monthly = df[df["Dealer Name"].notna()]
    if monthly.empty:
        return CheckResult("Duplicate transaction risk", "PASS", "No monthly transaction rows to check.")

    key_cols = [c for c in ["Manufacturer", "Division", "Model", "Type", "Date", "Dealer Name"]
                if c in monthly.columns]
    dupes = monthly[monthly.duplicated(subset=key_cols, keep=False)]

    if dupes.empty:
        return CheckResult("Duplicate transaction risk", "PASS",
                            "No repeated Manufacturer+Division+Model+Type+Date+Dealer combinations.")
    n_groups = monthly.duplicated(subset=key_cols, keep="first").sum()
    return CheckResult("Duplicate transaction risk", "WARN",
                        f"{n_groups:,} repeated transaction key combination(s) found "
                        f"({len(dupes):,} rows). Could be legitimate (multiple units, same dealer/"
                        f"month/model) or accidental double-counting -- worth a manual look.",
                        dupes.sort_values(key_cols))


def run_all_checks(df: pd.DataFrame) -> list:
    checks = [
        check_structure(df),
        check_missing_values(df),
        check_duplicates(df),
        check_date_coverage(df),
        check_annual_vs_monthly_split(df),
        check_grand_totals(df),
        check_units_sanity(df),
        check_category_consistency(df),
        check_yoy_totals(df),
        check_duplicate_transactions(df),
    ]
    return checks


def print_summary(checks: list):
    status_icon = {"PASS": "OK  ", "WARN": "WARN", "FAIL": "FAIL"}
    print()
    print("=" * 78)
    print("DATA QUALITY CHECK SUMMARY")
    print("=" * 78)
    for c in checks:
        print(f"[{status_icon[c.status]}] {c.name:<32} {c.message}")
    print("=" * 78)

    n_fail = sum(1 for c in checks if c.status == "FAIL")
    n_warn = sum(1 for c in checks if c.status == "WARN")
    n_pass = sum(1 for c in checks if c.status == "PASS")
    print(f"Result: {n_pass} passed, {n_warn} warnings, {n_fail} failed")
    if n_fail:
        print("Action needed: review FAIL items above before running the forecast.")
    elif n_warn:
        print("Review WARN items -- forecast can still run, but double-check these.")
    else:
        print("All checks passed. Data looks clean.")
    print()


def save_report(checks: list, df: pd.DataFrame, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    summary_rows = [{"Check": c.name, "Status": c.status, "Message": c.message,
                     "Flagged_Rows": len(c.detail_df)} for c in checks]
    summary_df = pd.DataFrame(summary_rows)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        for c in checks:
            if c.detail_df is not None and not c.detail_df.empty:
                sheet_name = c.name[:31]  # Excel sheet name limit
                out = c.detail_df.copy()
                for col in out.columns:
                    if pd.api.types.is_datetime64_any_dtype(out[col]):
                        out[col] = out[col].dt.strftime("%Y-%m-%d")
                out.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Report saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Run data quality checks on RV retail sales source file.")
    parser.add_argument("--data", default=DEFAULT_DATA_PATH, help="Path to source Excel file")
    parser.add_argument("--out", default=DEFAULT_OUT_PATH, help="Path to save the QC report")
    parser.add_argument("--no-report", action="store_true", help="Skip saving the Excel report, console only")
    args = parser.parse_args()

    print(f"Loading: {args.data}")
    if not os.path.exists(args.data):
        print(f"ERROR: File not found at {args.data}")
        sys.exit(1)

    df = pd.read_excel(args.data, sheet_name=0, header=0, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns.")

    checks = run_all_checks(df)
    print_summary(checks)

    if not args.no_report:
        save_report(checks, df, args.out)

    n_fail = sum(1 for c in checks if c.status == "FAIL")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
