"""
deploy_data.py
==============
Uploads existing forecast parquet files from your OneDrive output folder
(or generates fresh ones if needed) and pushes to GitHub so Streamlit Cloud
and Power BI update automatically.

Usage:
    python deploy_data.py            # Uses existing parquet files if found (instant!)
    python deploy_data.py --recompute # Forces re-running the ML forecast
"""
import os
import sys
import shutil
import glob
import subprocess
import retail_forecast_engine as engine

# Configure safe UTF-8 output encoding for Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def run_cmd(cmd, desc):
    print(f"[*] {desc}")
    ret = subprocess.run(cmd, shell=True)
    if ret.returncode != 0:
        print(f"\n[ERROR] Command failed: {desc}")
        sys.exit(ret.returncode)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    target_parquet_dir = os.path.join(script_dir, "parquet")
    os.makedirs(target_parquet_dir, exist_ok=True)

    recompute = "--recompute" in sys.argv or "--fresh" in sys.argv

    # Source directory where your existing forecast files live
    source_pdir = os.environ.get(
        "PARQUET_DIR",
        r"C:\Users\Karmur\OneDrive - Dometic Group\PY Project Output\RetailForecast\parquet"
    )

    existing_fc = os.path.join(source_pdir, "forecast.parquet")
    has_existing = os.path.exists(existing_fc)

    print("=" * 60)
    print("RV Retail Forecast - Production Data Deployment")
    print("=" * 60)

    if has_existing and not recompute:
        print(f"[*] Found existing forecast files in:\n    {source_pdir}")
        print("[*] Copying existing parquet files into project repository...")
        
        copied_count = 0
        for f in glob.glob(os.path.join(source_pdir, "*")):
            if f.endswith(".parquet") or f.endswith(".json"):
                dest = os.path.join(target_parquet_dir, os.path.basename(f))
                shutil.copy2(f, dest)
                copied_count += 1
                print(f"    -> Copied {os.path.basename(f)}")
        
        print(f"[*] {copied_count} file(s) ready for deployment.")

    else:
        if recompute:
            print("[*] Flag --recompute detected. Running fresh forecast pipeline...")
        else:
            print(f"[*] No existing files found in {source_pdir}. Running forecast pipeline...")

        # 1. Run the ML forecast pipeline
        run_cmd(
            f'"{sys.executable}" run_forecast.py --out "{target_parquet_dir}"',
            "Running ML Forecast Pipeline"
        )

        # 2. Run Attach Rate Projections (if sales data exists)
        try:
            run_cmd(
                f'"{sys.executable}" compute_attach_rate_forecast.py --parquet-dir "{target_parquet_dir}"',
                "Computing OEM Attach Rates & Sales Projections"
            )
        except Exception as e:
            print(f"[NOTE] Attach rate step skipped or completed with notice: {e}")

    # 3. Stage parquet files in Git
    print("\n" + "=" * 60)
    run_cmd('git add parquet/', "Staging Parquet Data in Git")

    # 4. Commit data
    subprocess.run('git commit -m "Upload existing production forecast data"', shell=True)

    # 5. Push to GitHub
    run_cmd(
        'git push origin main',
        "Pushing Data to GitHub (Streamlit Cloud & Power BI will Auto-Update)"
    )

    print("\n" + "=" * 60)
    print("[SUCCESS] Production forecast data successfully uploaded!")
    print("Streamlit Cloud and Power BI will refresh automatically in ~15 seconds.")
    print("=" * 60)


if __name__ == "__main__":
    main()
