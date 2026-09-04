"""
run_forecast.py
Run the full ML forecast pipeline and save results as parquet files.
Credentials and paths are loaded from .env (see .env.example).

Usage:
    python run_forecast.py
    python run_forecast.py --grains Type
    python run_forecast.py --full
    python run_forecast.py --origins 5
    python run_forecast.py --topn 50
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import time
import argparse
import pandas as pd
import retail_forecast_engine as engine


def save_results(results: dict, out_dir: str, archive_dir: str = None):
    """
    Saves the "latest" parquet files (what the dashboard reads) to out_dir,
    and -- if archive_dir is given -- also saves a timestamped copy into
    archive_dir/<YYYY-MM-DD_HHMM>/, one snapshot per run. This builds a
    run history: what was forecast, when, so accuracy can be tracked over
    time later by comparing an old snapshot's forecast for a given month
    against what history.parquet later shows actually happened.
    """
    def _write(target_dir):
        os.makedirs(target_dir, exist_ok=True)
        saves = {
            "forecast": results["forecast"], "backtest": results["backtest"],
            "history": results["history"], "market_share": results.get("market_share", pd.DataFrame()),
            "backtest_detail": results.get("backtest_detail", pd.DataFrame()),
        }
        for name, df in saves.items():
            path = os.path.join(target_dir, f"{name}.parquet")
            df.to_parquet(path, index=False, engine="pyarrow")
            print(f"  Saved {name}.parquet  ({len(df):,} rows)  -> {target_dir}")

        exog_df = results.get("exog", pd.DataFrame())
        exog_df.to_parquet(os.path.join(target_dir, "exog.parquet"), index=False, engine="pyarrow")

        meta = {
            "run_timestamp": results["run_timestamp"].isoformat(),
            "forecast_rows": len(results["forecast"]),
            "series_count": int(results["forecast"]["series_id"].nunique()) if len(results["forecast"]) else 0,
            "grains": results["forecast"]["Grain"].unique().tolist() if len(results["forecast"]) else [],
            "has_fred": not exog_df.empty,
            "has_market_share": not results.get("market_share", pd.DataFrame()).empty,
            "own_divisions": engine.OWN_DIVISIONS,
            "parquet_dir": target_dir,
        }
        with open(os.path.join(target_dir, "run_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  Saved run_meta.json  -> {target_dir}")

    print("Saving latest (dashboard reads this):")
    _write(out_dir)

    if archive_dir:
        stamp = results["run_timestamp"].strftime("%Y-%m-%d_%H%M")
        snapshot_dir = os.path.join(archive_dir, stamp)
        print(f"\nSaving monthly archive snapshot:")
        _write(snapshot_dir)
        return snapshot_dir
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=engine.DATA_PATH)
    parser.add_argument("--out", default=engine.PARQUET_DIR)
    parser.add_argument("--grains", nargs="+", choices=["Total", "Division", "Type", "Division_Type"],
                        default=["Total", "Division", "Type", "Division_Type"])
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--origins", type=int, default=engine.ROLLING_ORIGINS)
    parser.add_argument("--topn", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=int(os.environ.get("N_JOBS", -1)))
    parser.add_argument("--heavy", action="store_true",
                        help="Include TimesFM/Chronos foundation models (if installed) -- "
                             "usually not worth it for series with 5+ years of history; "
                             "off by default per best-practice recommendation")
    parser.add_argument("--archive-dir", default=os.environ.get("ARCHIVE_DIR",
                        os.path.join(os.path.dirname(engine.PARQUET_DIR), "archive")),
                        help="Folder to save a timestamped snapshot of every run into, "
                             "building up a forecast history over time. Set ARCHIVE_DIR=none "
                             "in .env or pass --archive-dir none to disable.")
    args = parser.parse_args()

    fast_mode = not args.full
    fred_key = engine.FRED_API_KEY
    archive_dir = None if str(args.archive_dir).lower() == "none" else args.archive_dir

    print("=" * 60)
    print("RV Retail Sales - ML Forecast Pipeline")
    print("=" * 60)
    print(f"  Data      : {args.data}")
    print(f"  Output    : {args.out}")
    print(f"  Archive   : {archive_dir if archive_dir else 'disabled'}")
    print(f"  Grains    : {args.grains}")
    print(f"  Fast mode : {fast_mode}")
    print(f"  Heavy ML  : {'enabled (--heavy, TimesFM/Chronos if installed)' if args.heavy else 'off (default -- classical/ML models only)'}")
    print(f"  Workers   : {'all cores' if args.jobs == -1 else args.jobs}")
    print(f"  Origins   : {args.origins}")
    print(f"  FRED      : {'yes' if fred_key else 'no (set FRED_API_KEY in .env)'}")
    print(f"  Own divs  : {engine.OWN_DIVISIONS if engine.OWN_DIVISIONS else 'not set'}")
    print("=" * 60)

    t0 = time.time()
    results = engine.run_pipeline(
        data_path=args.data, fred_api_key=fred_key if fred_key else None,
        grains=args.grains, top_n_per_grain=args.topn, fast_mode=fast_mode,
        n_jobs=args.jobs, n_origins=args.origins, skip_heavy_models=not args.heavy,
    )
    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    print()
    snapshot_dir = save_results(results, args.out, archive_dir=archive_dir)

    n_series = results["forecast"]["series_id"].nunique() if len(results["forecast"]) else 0
    print(f"\nDone. {n_series} series saved to:\n  {args.out}")
    if snapshot_dir:
        print(f"Archived snapshot:\n  {snapshot_dir}")


if __name__ == "__main__":
    main()
