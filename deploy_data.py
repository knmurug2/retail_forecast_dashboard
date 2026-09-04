"""
deploy_data.py
==============
1-Click script to run the forecast pipeline on your work laptop
and upload the generated parquet files to GitHub so Streamlit Cloud
and Power BI update automatically.

Usage:
    python deploy_data.py
"""
import os
import sys
import subprocess

# Configure safe UTF-8 output encoding for Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def run_step(cmd, desc):
    print("\n" + "=" * 60)
    print(f"[*] {desc}")
    print("=" * 60)
    ret = subprocess.run(cmd, shell=True)
    if ret.returncode != 0:
        print(f"\n[ERROR] Step failed: {desc}")
        sys.exit(ret.returncode)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    parquet_dir = os.path.join(script_dir, "parquet")
    os.makedirs(parquet_dir, exist_ok=True)

    print("=" * 60)
    print("RV Retail Forecast - Production Data Deployment")
    print("=" * 60)

    # 1. Run the ML forecast pipeline
    run_step(
        f'"{sys.executable}" run_forecast.py --out "{parquet_dir}"',
        "Running ML Forecast Pipeline (Generating Parquet Files)"
    )

    # 2. Run Attach Rate Projections (if sales data exists)
    try:
        run_step(
            f'"{sys.executable}" compute_attach_rate_forecast.py --parquet-dir "{parquet_dir}"',
            "Computing Dometic OEM Attach Rates & Sales Projections"
        )
    except Exception as e:
        print(f"[NOTE] Attach rate step skipped or completed with notice: {e}")

    # 3. Stage parquet files in Git
    run_step(
        'git add parquet/',
        "Staging Parquet Data in Git"
    )

    # 4. Commit data
    subprocess.run('git commit -m "Upload production forecast data"', shell=True)

    # 5. Push to GitHub
    run_step(
        'git push origin main',
        "Pushing Data to GitHub (Streamlit Cloud & Power BI will Auto-Update)"
    )

    print("\n" + "=" * 60)
    print("[SUCCESS] Production forecast data successfully uploaded!")
    print("Streamlit Cloud and Power BI will refresh automatically in ~15 seconds.")
    print("=" * 60)


if __name__ == "__main__":
    main()
