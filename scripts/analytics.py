"""
Analytics CLI: run cohort summaries, correlations (via pandas), and export PNG charts to screenshots/.
Usage:
  python scripts/analytics.py --db data/health360.db
"""
import argparse, sqlite3, pandas as pd, numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def load(db_path):
    con = sqlite3.connect(db_path)
    users = pd.read_sql("SELECT * FROM users", con)
    daily = pd.read_sql("SELECT * FROM daily_logs", con, parse_dates=["date"])
    con.close()
    return users, daily

def save_plot(df, x, y, title, out_path):
    plt.figure(figsize=(8,4.5))
    if isinstance(y, (list, tuple)) and len(y) > 1:
        for col in y:
            plt.plot(df[x], df[col], label=col)
        plt.legend()
    else:
        plt.plot(df[x], df[y])
    plt.title(title)
    plt.xlabel(x); plt.ylabel(", ".join(y) if isinstance(y, (list, tuple)) else y)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=str, default="data/health360.db")
    args = ap.parse_args()

    users, daily = load(args.db)

    # KPI: daily average PHI
    kpi = daily.groupby("date")["phi"].mean().reset_index()
    save_plot(kpi, "date", "phi", "Average PHI Over Time", "screenshots/avg_phi.png")

    # Risk distribution
    risk = daily["risk"].value_counts().reset_index()
    risk.columns = ["risk", "count"]
    risk.to_csv("screenshots/risk_distribution.csv", index=False)

    # Cohort means
    df = daily.merge(users[["user_id","cohort"]], on="user_id", how="left")
    cohort = df.groupby("cohort")[["sleep_hours","water_liters","exercise_minutes","screen_time_hours","stress_level","productivity_score","phi"]].mean().reset_index()
    cohort.to_csv("screenshots/cohort_means.csv", index=False)

    # Correlation matrix (export)
    corr = df[["sleep_hours","water_liters","screen_time_hours","exercise_minutes","stress_level","steps","resting_hr","productivity_score","phi"]].corr(numeric_only=True)
    corr.to_csv("screenshots/correlation_matrix.csv")

    print("[Analytics] Exported PNG/CSV to screenshots/. Add them to README.")

if __name__ == "__main__":
    main()
