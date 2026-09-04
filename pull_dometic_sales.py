"""
pull_dometic_sales.py
======================
Pulls Dometic's actual sales-into-OEMs from D365, joins to customer master to
resolve each direct customer account up to its PARENT account (the actual
manufacturer), and saves monthly summaries at the parent level -- overall,
by Product Area, and by sales segment (for peer benchmarking downstream).

Run this before compute_attach_rate_forecast.py.

Usage:
    python pull_dometic_sales.py
    python pull_dometic_sales.py --start 2020-01-01
    python pull_dometic_sales.py --out "C:/path/to/output.xlsx"

Requires D365 SQL credentials in .env:
    D365_SQL_SERVER, D365_SQL_DATABASE, D365_SQL_USER, D365_SQL_PASSWORD
"""

import os
import sys
import argparse
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass

DEFAULT_OUT_PATH = os.environ.get(
    "DOMETIC_SALES_OUT",
    r"C:\Users\Karmur\OneDrive - Dometic Group\PY Project Output\DometicSales\dometic_sales_by_parent.xlsx"
)

SALES_QUERY_TEMPLATE = """
SELECT
    im.DOMITEMSTATUS AS Status,
    pm.PRODUCTNAME AS ItemName,
    a.FRIENDLYCATEGORYNAME AS ProductArea,
    b.FRIENDLYCATEGORYNAME AS ProductGroup,
    c.FRIENDLYCATEGORYNAME AS ModelFamily,
    d.FRIENDLYCATEGORYNAME AS Model,
    DATEADD(mm, DATEDIFF(mm, 0, dbo.DueDate(s.REQUESTEDSHIPPINGDATE,s.CONFIRMEDSHIPPINGDATE)), 0) AS SalesMonth,
    pl.ITEMNUMBER AS ItemNumber,
    cm.CUSTOMERACCOUNT AS CustomerAccount,
    cm.ORGANIZATIONNAME AS CustomerName,
    cm.salessegmentid AS SalesSegmentID,
    cm.salessubsegmentid AS SalesSubSegmentID,
    SUM(s.ORDEREDSALESQUANTITY) AS SalesQty,
    SUM(s.LINEAMOUNT) AS SalesRevenue
FROM (
    SELECT
        im.itemnumber,
        ic.domplannercode
    FROM EcoResReleasedProductV2Staging AS im
    LEFT JOIN ReqItemCoverageSettingsStaging AS ic
        ON im.ITEMNUMBER = ic.ITEMNUMBER
        AND im.DOMITEMDEFAULTWAREHOUSE = ic.COVERAGEWAREHOUSEID
) AS pl
LEFT JOIN DOMSalesOrderLineV3Staging AS s ON pl.ITEMNUMBER = s.ITEMNUMBER
LEFT JOIN DOMSalesOrderHeaderStaging AS sh ON s.SALESORDERNUMBER = sh.SALESORDERNUMBER
LEFT JOIN CustCustomerV3Staging AS cm ON sh.ORDERINGCUSTOMERACCOUNTNUMBER = cm.CUSTOMERACCOUNT
LEFT JOIN EcoResReleasedProductV2Staging AS im ON pl.ITEMNUMBER = im.ITEMNUMBER
LEFT JOIN EcoResProductV2Staging AS pm ON im.ITEMNUMBER = pm.PRODUCTNUMBER
LEFT JOIN EcoResProductCategoryStaging AS a ON pm.DOMPRODUCTAREAHIER = a.CATEGORYNAME
LEFT JOIN EcoResProductCategoryStaging AS b ON pm.DOMPRODUCTGROUPHIER = b.CATEGORYNAME
LEFT JOIN EcoResProductCategoryStaging AS c ON pm.DOMMODELFAMILYHIER = c.CATEGORYNAME
LEFT JOIN EcoResProductCategoryStaging AS d ON pm.DOMMODELHIER = d.CATEGORYNAME
WHERE
    pl.DOMPLANNERCODE IS NOT NULL AND pl.DOMPLANNERCODE <> ''
    AND dbo.DueDate(s.REQUESTEDSHIPPINGDATE,s.CONFIRMEDSHIPPINGDATE) >= '{start_date}'
GROUP BY
    DATEADD(mm, DATEDIFF(mm, 0, dbo.DueDate(s.REQUESTEDSHIPPINGDATE,s.CONFIRMEDSHIPPINGDATE)), 0),
    pl.ITEMNUMBER,
    cm.CUSTOMERACCOUNT,
    cm.ORGANIZATIONNAME,
    cm.salessegmentid,
    cm.salessubsegmentid,
    im.DOMITEMSTATUS,
    pm.PRODUCTNAME,
    a.FRIENDLYCATEGORYNAME,
    b.FRIENDLYCATEGORYNAME,
    c.FRIENDLYCATEGORYNAME,
    d.FRIENDLYCATEGORYNAME
ORDER BY
    cm.CUSTOMERACCOUNT, SalesMonth DESC
"""

CUSTOMER_MASTER_QUERY = """
SELECT CM.DOMPARENTACCOUNT
,CM.CUSTOMERACCOUNT
,CM.ORGANIZATIONNAME
,cmp.ORGANIZATIONNAME as DOMPARENTACCOUNTNAME
,CM.INVOICEACCOUNT
,CM.FULLPRIMARYADDRESS
,CM.ADDRESSZIPCODE
,CM.ADDRESSCITY
,CM.ADDRESSCOUNTY
,CM.DELIVERYADDRESSDISTRICTNAME
,CM.ADDRESSSTATE
,CM.COMPANYCHAIN
,CM.SALESSEGMENTID
,CM.SALESSUBSEGMENTID
,CM.COMMISSIONSALESGROUPID
,CM.PAYMENTTERMS
,CM.DOMHOMEWAREHOUSE
,CM.DISCOUNTPRICEGROUPID
From CustCustomerV3Staging as CM
left join CustCustomerV3Staging as cmp on cm.DOMPARENTACCOUNT = cmp.CUSTOMERACCOUNT
"""


def get_connection():
    import pymssql
    return pymssql.connect(
        server=os.environ["D365_SQL_SERVER"],
        database=os.environ["D365_SQL_DATABASE"],
        user=os.environ["D365_SQL_USER"],
        password=os.environ["D365_SQL_PASSWORD"],
        tds_version="7.3",
    )


def normalize_columns(df: pd.DataFrame, expected: list, source_name: str) -> pd.DataFrame:
    """
    SQL Server drivers (pymssql/FreeTDS in particular) can return column
    names in a different case than the query specifies. Resolve every
    expected column case-insensitively and fail immediately with a clear
    message (showing the actual columns found) if something's genuinely
    missing, rather than a cryptic KeyError deep in a later merge.
    """
    lookup = {c.lower(): c for c in df.columns}
    rename_map = {}
    missing = []
    for col in expected:
        actual = lookup.get(col.lower())
        if actual is None:
            missing.append(col)
        elif actual != col:
            rename_map[actual] = col

    if missing:
        raise KeyError(
            f"{source_name}: expected column(s) {missing} not found. "
            f"Actual columns returned: {df.columns.tolist()}"
        )
    if rename_map:
        print(f"  ({source_name}: {len(rename_map)} column name(s) came back in a different "
              f"case than expected, e.g. {list(rename_map.items())[0]} -- renamed automatically)")
        df = df.rename(columns=rename_map)
    return df


def filter_sentinel_dates(df: pd.DataFrame, start_date: str, max_months_future: int = 3) -> pd.DataFrame:
    """
    ERP systems commonly use a far-future placeholder date (e.g. 2099-01-01)
    to mean "no ship date set yet" on open orders. Left in, it makes
    max(SalesMonth) think "now" is 2099, silently breaking every trailing-
    window calculation downstream. Anything more than a few months beyond
    today, or before the requested start date, gets excluded and reported.
    """
    df = df.copy()
    df["SalesMonth"] = pd.to_datetime(df["SalesMonth"], errors="coerce")

    today = pd.Timestamp.now().normalize()
    future_cutoff = today + pd.DateOffset(months=max_months_future)
    start_cutoff = pd.to_datetime(start_date)

    bad_mask = (df["SalesMonth"].isna() | (df["SalesMonth"] > future_cutoff) |
                (df["SalesMonth"] < start_cutoff))
    n_bad = int(bad_mask.sum())

    if n_bad:
        bad_dates = sorted(df.loc[bad_mask, "SalesMonth"].dropna().unique())
        preview = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in (bad_dates[:3] + bad_dates[-3:])]
        print(f"  WARNING: {n_bad:,} row(s) had an out-of-range or missing SalesMonth "
              f"(likely placeholder/TBD ship dates, e.g. {preview}) -- excluded from the pull.")

    return df[~bad_mask].copy()


def _clean_id(series: pd.Series) -> pd.Series:
    """
    Guards against a numeric-looking ID column getting read as float64
    somewhere upstream (producing "12345.0" instead of "12345"), which
    would silently fail every join match rather than raising an error.
    """
    s = series.astype(str).str.strip()
    return s.str.replace(r"\.0$", "", regex=True)


def pull_sales(conn, start_date: str) -> pd.DataFrame:
    print(f"Pulling sales data from {start_date}...")
    query = SALES_QUERY_TEMPLATE.format(start_date=start_date)
    df = pd.read_sql(query, conn)
    print(f"  {len(df):,} rows pulled")
    df = normalize_columns(df, ["CustomerAccount", "SalesMonth", "SalesQty", "SalesRevenue"], "Sales query")
    df = filter_sentinel_dates(df, start_date)
    return df


def pull_customer_master(conn) -> pd.DataFrame:
    print("Pulling customer master...")
    df = pd.read_sql(CUSTOMER_MASTER_QUERY, conn)
    print(f"  {len(df):,} rows pulled")
    df = normalize_columns(df, ["CUSTOMERACCOUNT", "DOMPARENTACCOUNT",
                                "DOMPARENTACCOUNTNAME", "ORGANIZATIONNAME"], "Customer master query")
    return df


def build_parent_mapping(sales_df: pd.DataFrame, cust_df: pd.DataFrame) -> pd.DataFrame:
    """
    Sales data only has the DIRECT customer account (the dealer/plant that
    actually ordered), not the parent manufacturer. Join to customer master
    to roll each direct account up to its parent (DOMPARENTACCOUNT) --
    that's what matches "ParentCustomerNumber" in RV_Cust_Data.

    Where DOMPARENTACCOUNT is blank/null (the account IS its own parent),
    the account rolls up to itself.

    NOTE: the sales query aliases this column as "CustomerAccount" while the
    customer master query returns it unaliased as "CUSTOMERACCOUNT" -- these
    are genuinely different column names on the two sides, so the join uses
    left_on/right_on rather than a single on=.
    """
    sales_df = sales_df.copy()
    cust_df = cust_df.copy()
    sales_df["CustomerAccount"] = _clean_id(sales_df["CustomerAccount"])
    cust_df["CUSTOMERACCOUNT"] = _clean_id(cust_df["CUSTOMERACCOUNT"])
    cust_df["DOMPARENTACCOUNT"] = cust_df["DOMPARENTACCOUNT"].apply(
        lambda v: str(v).strip() if pd.notna(v) else v)

    cust_map = cust_df[["CUSTOMERACCOUNT", "DOMPARENTACCOUNT", "DOMPARENTACCOUNTNAME", "ORGANIZATIONNAME"]].copy()
    cust_map["DOMPARENTACCOUNT"] = cust_map["DOMPARENTACCOUNT"].fillna(cust_map["CUSTOMERACCOUNT"])
    cust_map["DOMPARENTACCOUNTNAME"] = cust_map["DOMPARENTACCOUNTNAME"].fillna(cust_map["ORGANIZATIONNAME"])

    merged = sales_df.merge(
        cust_map[["CUSTOMERACCOUNT", "DOMPARENTACCOUNT", "DOMPARENTACCOUNTNAME"]],
        left_on="CustomerAccount", right_on="CUSTOMERACCOUNT", how="left"
    )

    unmatched = merged["DOMPARENTACCOUNT"].isna().sum()
    if unmatched:
        print(f"  WARNING: {unmatched:,} sales rows had no customer master match "
              f"(CustomerAccount not found) -- these will be excluded from the "
              f"parent-level summary.")

    return merged


def summarize_monthly(merged: pd.DataFrame) -> pd.DataFrame:
    """Monthly Dometic sales by parent customer (manufacturer)."""
    df = merged.dropna(subset=["DOMPARENTACCOUNT"]).copy()
    summary = (df.groupby(["DOMPARENTACCOUNT", "DOMPARENTACCOUNTNAME", "SalesMonth"], as_index=False)
               .agg(SalesQty=("SalesQty", "sum"), SalesRevenue=("SalesRevenue", "sum")))
    summary = summary.rename(columns={
        "DOMPARENTACCOUNT": "ParentCustomerNumber",
        "DOMPARENTACCOUNTNAME": "ParentCustomerName",
    })
    return summary.sort_values(["ParentCustomerNumber", "SalesMonth"])


def summarize_monthly_by_area(merged: pd.DataFrame) -> pd.DataFrame:
    """Monthly Dometic sales by parent customer AND Product Area."""
    df = merged.dropna(subset=["DOMPARENTACCOUNT"]).copy()
    if "ProductArea" not in df.columns:
        print("  WARNING: ProductArea column not found -- skipping product-area breakdown.")
        return pd.DataFrame()

    df["ProductArea"] = df["ProductArea"].fillna("(Uncategorized)")
    summary = (df.groupby(["DOMPARENTACCOUNT", "DOMPARENTACCOUNTNAME", "ProductArea", "SalesMonth"],
                          as_index=False)
               .agg(SalesQty=("SalesQty", "sum"), SalesRevenue=("SalesRevenue", "sum")))
    summary = summary.rename(columns={
        "DOMPARENTACCOUNT": "ParentCustomerNumber",
        "DOMPARENTACCOUNTNAME": "ParentCustomerName",
    })
    return summary.sort_values(["ParentCustomerNumber", "ProductArea", "SalesMonth"])


def build_parent_segment_map(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Each parent account's sales segment -- used downstream for peer
    benchmarking (comparing a customer's attach rate against others in the
    same segment, rather than the whole population). Segment comes from the
    sales query's SalesSegmentID/SalesSubSegmentID, carried at the direct-
    customer level; rolled up to the parent using the most frequent value,
    same conflict-resolution approach used for the parent ID mapping itself
    -- a parent could theoretically show more than one segment across its
    direct accounts, so majority vote rather than assuming uniformity.
    """
    df = merged.dropna(subset=["DOMPARENTACCOUNT"]).copy()
    seg_cols = [c for c in ["SalesSegmentID", "SalesSubSegmentID"] if c in df.columns]
    if not seg_cols:
        print("  WARNING: no segment columns found -- skipping segment map.")
        return pd.DataFrame()

    rows = []
    for pid, grp in df.groupby("DOMPARENTACCOUNT"):
        row = {"ParentCustomerNumber": pid}
        for col in seg_cols:
            vals = grp[col].dropna()
            row[col] = vals.mode().iloc[0] if not vals.empty else None
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2020-01-01", help="Start date for sales pull")
    parser.add_argument("--out", default=DEFAULT_OUT_PATH)
    args = parser.parse_args()

    print("=" * 60)
    print("Dometic Sales Pull -- by Parent Customer")
    print("=" * 60)

    conn = get_connection()
    try:
        sales_df = pull_sales(conn, args.start)
        cust_df = pull_customer_master(conn)
    finally:
        conn.close()

    print("\nMapping direct customers to parent accounts...")
    merged = build_parent_mapping(sales_df, cust_df)

    print("Summarizing to monthly parent-customer totals...")
    monthly_summary = summarize_monthly(merged)

    print("Summarizing to monthly parent-customer x product-area totals...")
    monthly_by_area = summarize_monthly_by_area(merged)

    print("Building parent-level segment map (for peer benchmarking)...")
    segment_map = build_parent_segment_map(merged)

    print(f"\n  Parent customers found: {monthly_summary['ParentCustomerNumber'].nunique()}")
    print(f"  Months covered: {monthly_summary['SalesMonth'].min()} to {monthly_summary['SalesMonth'].max()}")
    if not monthly_by_area.empty:
        print(f"  Product areas found: {sorted(monthly_by_area['ProductArea'].unique())}")
    if not segment_map.empty:
        print(f"  Segments found: {segment_map['SalesSegmentID'].nunique() if 'SalesSegmentID' in segment_map.columns else 0}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Detail line-level data can easily exceed Excel's 1,048,576-row-per-sheet
    # limit on a multi-year pull. Saved as parquet instead.
    detail_path = os.path.join(os.path.dirname(args.out), "dometic_sales_detail.parquet")
    if len(merged) > 1_000_000:
        print(f"\nDetail line-level data has {len(merged):,} rows -- too large for an Excel "
              f"sheet (limit 1,048,576). Saving separately as parquet instead.")
    merged.to_parquet(detail_path, index=False, engine="pyarrow")
    print(f"Saved: {detail_path}")

    with pd.ExcelWriter(args.out, engine="openpyxl") as writer:
        monthly_summary.to_excel(writer, sheet_name="Monthly_By_Parent", index=False)
        if not monthly_by_area.empty:
            monthly_by_area.to_excel(writer, sheet_name="Monthly_By_Parent_Area", index=False)
        if not segment_map.empty:
            segment_map.to_excel(writer, sheet_name="Parent_Segment_Map", index=False)
        cust_df.to_excel(writer, sheet_name="Customer_Master", index=False)

    print(f"\nSaved: {args.out}")
    sheet_list = "Monthly_By_Parent, Monthly_By_Parent_Area, Parent_Segment_Map, Customer_Master"
    print(f"Sheets: {sheet_list}")
    print(f"Full line-level detail: {detail_path}")


if __name__ == "__main__":
    main()
