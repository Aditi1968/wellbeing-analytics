import sqlite3, pandas as pd, numpy as np
from datetime import datetime, timedelta
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "health360.db"

def generate_synthetic_data(n_users=120, days=60, seed=42):
    rng = np.random.default_rng(seed)
    cohorts = ["Student", "Working Pro", "Night Owl", "Early Bird"]
    genders = ["Male", "Female", "Other"]
    ages = rng.integers(18, 45, size=n_users)
    cohort = rng.choice(cohorts, size=n_users, replace=True, p=[0.35, 0.35, 0.15, 0.15])
    gender = rng.choice(genders, size=n_users, replace=True, p=[0.48, 0.48, 0.04])
    users = pd.DataFrame({
        "user_id": range(1, n_users+1),
        "name": [f"User {i}" for i in range(1, n_users+1)],
        "age": ages,
        "gender": gender,
        "cohort": cohort,
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
    daily = pd.DataFrame(recs, columns=[
        "user_id","date","sleep_hours","water_liters","screen_time_hours",
        "exercise_minutes","stress_level","steps","resting_hr","productivity_score"
    ])

    weeks = daily["date"].unique().tolist()[::7]
    sv = []
    mood_bank = [
        "Felt anxious and low energy today but managed a short walk.",
        "Calm and focused after a good night's sleep.",
        "Overwhelmed with tasks; screen time was very high.",
        "Motivated; workout was great and mood is positive.",
        "Stressed due to deadlines. Trouble sleeping.",
        "Feeling balanced and hydrated; focus improved."
    ]
    for uid in users.user_id:
        for w in weeks:
            mood = np.random.choice(mood_bank)
            mood_score = int(np.random.randint(1, 11))
            energy = int(np.random.randint(1, 11))
            sv.append([uid, w, mood, mood_score, energy])
    surveys = pd.DataFrame(sv, columns=["user_id","week_start","mood_text","mood_score","energy_level"])

    return users, daily, surveys

def phi_score(row):
    sleep_c = min(row["sleep_hours"]/8, 1.2)
    water_c = min(row["water_liters"]/3, 1.2)
    ex_c = min(row["exercise_minutes"]/30, 1.5)
    stress_c = max(0, 1 - (row["stress_level"]-3)/7)
    screen_c = max(0, 1 - (row["screen_time_hours"]-4)/8)
    score = (0.30*sleep_c + 0.20*water_c + 0.25*ex_c + 0.15*stress_c + 0.10*screen_c) / (0.30+0.20+0.25+0.15+0.10)
    return float(max(0, min(score*100, 100)))

def classify_risk(phi, stress):
    if phi < 45 or stress >= 8:
        return "High"
    elif phi < 65 or stress >= 6:
        return "Medium"
    return "Low"

def create_db(n_users=160, days=60, seed=42):
    users, daily, surveys = generate_synthetic_data(n_users=n_users, days=days, seed=seed)
    # enrich
    daily = daily.copy()
    daily["phi"] = daily.apply(phi_score, axis=1)
    daily["risk"] = [classify_risk(p, s) for p, s in zip(daily["phi"], daily["stress_level"])]

    # mood tags
    def mood_tags(text):
        text = (text or "").lower()
        tags = []
        bad_kw = ["anxious","overwhelmed","stressed","trouble","low energy","exhausted","tired","burnout","depressed"]
        good_kw = ["calm","focused","motivated","positive","balanced","great"]
        if any(k in text for k in bad_kw): tags.append("negative")
        if any(k in text for k in good_kw): tags.append("positive")
        if "sleep" in text: tags.append("sleep")
        if "screen" in text: tags.append("screen-time")
        if "workout" in text or "walk" in text: tags.append("activity")
        if "hydrated" in text: tags.append("hydration")
        return ",".join(sorted(set(tags))) or "neutral"

    surveys = surveys.copy()
    surveys["tags"] = surveys["mood_text"].apply(mood_tags)

    DB_PATH.parent.mkdir(exist_ok=True, parents=True)
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    # apply schema
    schema = (Path(__file__).resolve().parents[1] / "sql" / "schema.sql").read_text(encoding="utf-8")
    cur.executescript(schema)

    users.to_sql("users", con, if_exists="append", index=False)
    daily.to_sql("daily_logs", con, if_exists="append", index=False)
    surveys.to_sql("surveys", con, if_exists="append", index=False)
    con.commit()
    con.close()
    print(f"Created SQLite DB at: {DB_PATH}")

if __name__ == "__main__":
    create_db()
