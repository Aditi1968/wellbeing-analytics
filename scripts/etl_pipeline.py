"""
ETL Pipeline: ingest CSV -> clean -> compute PHI/risk -> load into SQLite.
Usage:
  python scripts/etl_pipeline.py --input data/raw/health_logs.csv --db data/health360.db
If no CSV is provided, the script generates synthetic data (same logic as the app).
"""
import argparse, sqlite3, pandas as pd, numpy as np
from pathlib import Path
from datetime import datetime, timedelta

def phi_score(row):
    sleep_c = min(row.get("sleep_hours",0)/8, 1.2)
    water_c = min(row.get("water_liters",0)/3, 1.2)
    ex_c = min(row.get("exercise_minutes",0)/30, 1.5)
    stress_c = max(0, 1 - (row.get("stress_level",0)-3)/7)
    screen_c = max(0, 1 - (row.get("screen_time_hours",0)-4)/8)
    score = (0.30*sleep_c + 0.20*water_c + 0.25*ex_c + 0.15*stress_c + 0.10*screen_c) / (0.30+0.20+0.25+0.15+0.10)
    return float(max(0, min(score*100, 100)))

def classify_risk(phi, stress):
    if phi < 45 or stress >= 8: return "High"
    if phi < 65 or stress >= 6: return "Medium"
    return "Low"

def generate_synthetic(n_users=120, days=60, seed=123):
    rng = np.random.default_rng(seed)
    cohorts = ["Student", "Working Pro", "Night Owl", "Early Bird"]
    genders = ["Male", "Female", "Other"]
    ages = rng.integers(18, 45, size=n_users)
    cohort = rng.choice(cohorts, size=n_users, replace=True, p=[0.35, 0.35, 0.15, 0.15])
    gender = rng.choice(genders, size=n_users, replace=True, p=[0.48, 0.48, 0.04])
    users = pd.DataFrame({
        "user_id": range(1, n_users+1),
        "name": [f"User {i}" for i in range(1, n_users+1)],
        "age": ages, "gender": gender, "cohort": cohort,
    })
    start = datetime.now().date() - timedelta(days=days)
    recs = []
    for uid, c in zip(users.user_id, users.cohort):
        if c == "Student":
            base_sleep = 6.6; base_screen = 6.5; base_ex = 20; base_stress = 6.2
        elif c == "Working Pro":
            base_sleep = 6.8; base_screen = 7.5; base_ex = 25; base_stress = 6.8
        elif c == "Night Owl":
            base_sleep = 5.9; base_screen = 8.0; base_ex = 18; base_stress = 7.2
        else:
            base_sleep = 7.6; base_screen = 5.5; base_ex = 32; base_stress = 5.4
        base_water = 2.2 + (0.3 if c=="Early Bird" else 0)
        for d in range(days):
            date = start + timedelta(days=d+1)
            sleep = float(np.clip(np.random.normal(base_sleep, 0.9), 3.5, 9.5))
            water = float(np.clip(np.random.normal(base_water, 0.5), 0.8, 4.5))
            screen = float(np.clip(np.random.normal(base_screen, 1.8), 1.0, 14.0))
            ex = float(np.clip(np.random.normal(base_ex, 12), 0, 120))
            stress = float(np.clip(np.random.normal(base_stress, 1.5), 1, 10))
            steps = int(np.clip(np.random.normal(7000 + (ex*50), 2000), 500, 25000))
            heart = int(np.clip(np.random.normal(72 + (stress-5)*2 - (ex/60)*3, 6), 50, 110))
            productivity = float(np.clip(
                0.5*min(sleep/8,1) + 0.2*(ex/30) + 0.15*(water/3) - 0.25*(screen/8) - 0.2*((stress-5)/5) + np.random.normal(0,0.1),
                -0.3, 1.2
            ))
            productivity = round((productivity+0.3)/1.5*100,2)
            recs.append([uid, date.isoformat(), sleep, water, screen, ex, stress, steps, heart, productivity])
    daily = pd.DataFrame(recs, columns=["user_id","date","sleep_hours","water_liters","screen_time_hours","exercise_minutes","stress_level","steps","resting_hr","productivity_score"])
    return users, daily

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="", help="Optional CSV of daily logs to load instead of synthetic data.")
    ap.add_argument("--db", type=str, default="data/health360.db")
    args = ap.parse_args()

    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Load or generate daily logs
    if args.input and Path(args.input).exists():
        daily = pd.read_csv(args.input)
        # Ensure essential columns exist
        required = {"user_id","date","sleep_hours","water_liters","screen_time_hours","exercise_minutes","stress_level","steps","resting_hr","productivity_score"}
        missing = required - set(daily.columns)
        if missing: raise ValueError(f"Missing columns in input CSV: {missing}")
        # Minimal users table (derive from daily if not present)
        users = pd.DataFrame({"user_id": sorted(daily["user_id"].unique())})
        users["name"] = ["User {}".format(i) for i in users["user_id"]]
        users["age"] = 25
        users["gender"] = "Other"
        users["cohort"] = "Imported"
    else:
        users, daily = generate_synthetic()

    # Compute PHI & risk
    daily = daily.copy()
    daily["phi"] = daily.apply(phi_score, axis=1)
    daily["risk"] = [classify_risk(p, s) for p, s in zip(daily["phi"], daily["stress_level"])]

    # Create DB & apply schema
    con = sqlite3.connect(db_path)
    schema = Path("sql/schema.sql").read_text(encoding="utf-8")
    con.executescript(schema)

    users.to_sql("users", con, if_exists="append", index=False)
    daily.to_sql("daily_logs", con, if_exists="append", index=False)
    con.commit()
    con.close()
    print(f"[ETL] Loaded data into {db_path}")

if __name__ == "__main__":
    main()
