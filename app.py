
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from io import StringIO
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Health360 – Preventive Health & Lifestyle Intelligence", layout="wide")

# -------------------- Utility & Data --------------------
@st.cache_data(show_spinner=False)
def generate_synthetic_data(n_users=120, days=60, seed=42):
    rng = np.random.default_rng(seed)
    # Users
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
    # Daily logs
    start = datetime.now().date() - timedelta(days=days)
    recs = []
    for uid, c in zip(users.user_id, users.cohort):
        # baseline patterns by cohort
        if c == "Student":
            base_sleep = 6.6; base_screen = 6.5; base_ex = 20; base_stress = 6.2
        elif c == "Working Pro":
            base_sleep = 6.8; base_screen = 7.5; base_ex = 25; base_stress = 6.8
        elif c == "Night Owl":
            base_sleep = 5.9; base_screen = 8.0; base_ex = 18; base_stress = 7.2
        else:  # Early Bird
            base_sleep = 7.6; base_screen = 5.5; base_ex = 32; base_stress = 5.4
        base_water = 2.2 + (0.3 if c=="Early Bird" else 0)
        for d in range(days):
            date = start + timedelta(days=d+1)
            sleep = np.clip(np.random.normal(base_sleep, 0.9), 3.5, 9.5)
            water = np.clip(np.random.normal(base_water, 0.5), 0.8, 4.5)
            screen = np.clip(np.random.normal(base_screen, 1.8), 1.0, 14.0)
            ex = np.clip(np.random.normal(base_ex, 12), 0, 120)
            stress = np.clip(np.random.normal(base_stress, 1.5), 1, 10)
            steps = int(np.clip(np.random.normal(7000 + (ex*50), 2000), 500, 25000))
            heart = int(np.clip(np.random.normal(72 + (stress-5)*2 - (ex/60)*3, 6), 50, 110))
            productivity = float(np.clip(
                0.5*min(sleep/8,1) + 0.2*(ex/30) + 0.15*(water/3) - 0.25*(screen/8) - 0.2*((stress-5)/5) + np.random.normal(0,0.1),
                -0.3, 1.2
            ))
            productivity = round((productivity+0.3)/1.5*100,2)  # 0-100 scale
            recs.append([uid, date.isoformat(), sleep, water, screen, ex, stress, steps, heart, productivity])
    daily = pd.DataFrame(recs, columns=[
        "user_id","date","sleep_hours","water_liters","screen_time_hours","exercise_minutes","stress_level","steps","resting_hr","productivity_score"
    ])
    # Surveys (weekly mood text + score)
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
            mood_score = np.random.randint(1, 11)
            energy = np.random.randint(1, 11)
            sv.append([uid, w, mood, mood_score, energy])
    surveys = pd.DataFrame(sv, columns=["user_id","week_start","mood_text","mood_score","energy_level"])
    return users, daily, surveys

def phi_score(row):
    # Weighted components -> 0..100
    # Ideal targets: sleep 8h, water 3L, exercise 30m, stress 3 (lower is better), screen 4h
    sleep_c = min(row["sleep_hours"]/8, 1.2)
    water_c = min(row["water_liters"]/3, 1.2)
    ex_c = min(row["exercise_minutes"]/30, 1.5)
    stress_c = max(0, 1 - (row["stress_level"]-3)/7)  # 1 when stress=3, lower when higher stress
    screen_c = max(0, 1 - (row["screen_time_hours"]-4)/8)
    # weights
    score = (0.30*sleep_c + 0.20*water_c + 0.25*ex_c + 0.15*stress_c + 0.10*screen_c) / (0.30+0.20+0.25+0.15+0.10)
    return float(np.clip(score*100, 0, 100))

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
    return list(set(tags)) or ["neutral"]

def classify_risk(phi, stress):
    if phi < 45 or stress >= 8:
        return "High"
    elif phi < 65 or stress >= 6:
        return "Medium"
    return "Low"

@st.cache_data(show_spinner=False)
def enrich(users, daily, surveys):
    d = daily.copy()
    d["phi"] = d.apply(phi_score, axis=1)
    d["risk"] = [classify_risk(p, s) for p, s in zip(d["phi"], d["stress_level"])]
    # Aggregate weekly and join surveys
    d["date"] = pd.to_datetime(d["date"])
    d["week_start"] = d["date"] - pd.to_timedelta(d["date"].dt.weekday, unit="d")
    mood = surveys.copy()
    mood["tags"] = mood["mood_text"].apply(mood_tags)
    return d, mood

def recommend_interventions(user_df):
    # Simple rule-based recommendations
    out = []
    if user_df["sleep_hours"].mean() < 6.5:
        out.append("Sleep Hygiene Reset (night routine, fixed schedule, no screens 1h before bed)")
    if user_df["water_liters"].mean() < 2.0:
        out.append("Hydration Habit Bootcamp (hourly reminders, 2.5–3L/day challenge)")
    if user_df["exercise_minutes"].mean() < 20:
        out.append("7-Day Movement Kickstart (20-min walks + light stretching)")
    if user_df["screen_time_hours"].mean() > 7.0:
        out.append("Digital Detox Sprint (app timers, 2h social cap, no-screens morning)")
    if user_df["stress_level"].mean() > 6.5:
        out.append("Stress Detox Week (guided breathing, journaling, 10-min mindfulness)")
    return out or ["Maintain current routine with weekly check-ins"]

def cohort_bootcamp_plan(cohort_df):
    # Identify top gaps across cohort
    gaps = []
    metrics = {
        "sleep_hours": 7.0,  # target minimum
        "water_liters": 2.5,
        "exercise_minutes": 30.0,
        "screen_time_hours": 5.0,
        "stress_level": 4.0
    }
    for k, v in metrics.items():
        m = cohort_df[k].mean()
        if k == "stress_level" or k == "screen_time_hours":
            if m > v: gaps.append(k)
        else:
            if m < v: gaps.append(k)
    themes = {
        "sleep_hours": "Sleep Hygiene Reset",
        "water_liters": "Hydration Habit",
        "exercise_minutes": "Movement Kickstart",
        "screen_time_hours": "Digital Detox",
        "stress_level": "Stress Detox"
    }
    selected = [themes[g] for g in gaps][:3]
    # Create a 7-day schedule
    days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    plan = []
    for i, day in enumerate(days):
        block = []
        for t in selected:
            if t == "Sleep Hygiene Reset":
                block.append("Lights-out target + no-screens 60m")
            if t == "Hydration Habit":
                block.append("5x 500ml targets; log at lunch & 5pm")
            if t == "Movement Kickstart":
                block.append("20–30 min brisk walk + 5 min stretch")
            if t == "Digital Detox":
                block.append("App timers + morning no-screen rule")
            if t == "Stress Detox":
                block.append("10-min breathing + 5-min gratitude")
        plan.append({"day": day, "focus": ", ".join(selected), "actions": " | ".join(block)})
    return pd.DataFrame(plan)

@st.cache_data(show_spinner=False)
def train_predictive_model(daily_enriched):
    # Train a simple model to predict High/Med vs Low risk using last 14 days aggregates per user
    d = daily_enriched.copy()
    d["target"] = (d["risk"].isin(["High","Medium"])).astype(int)
    # features
    feats = ["sleep_hours","water_liters","screen_time_hours","exercise_minutes","stress_level","steps","resting_hr","productivity_score","phi"]
    # aggregate to user-week level
    d["week_start"] = d["date"] - pd.to_timedelta(d["date"].dt.weekday, unit="d")
    agg = d.groupby(["user_id","week_start"])[feats+["target"]].mean().reset_index()
    X = agg[feats]
    y = agg["target"]
    pipe = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=1000))])
    pipe.fit(X, y)
    prob = pipe.predict_proba(X)[:,1]
    auc = roc_auc_score(y, prob) if len(np.unique(y)) > 1 else np.nan
    # Feature importance via absolute coefficients
    coefs = pipe.named_steps["lr"].coef_[0]
    imp = pd.DataFrame({"feature": feats, "importance": np.abs(coefs)}).sort_values("importance", ascending=False)
    return pipe, auc, imp

# -------------------- App UI --------------------
st.title("🩺 Health360 – Preventive Health & Lifestyle Intelligence")
st.caption("Complex health analytics demo showcasing SQL-like organization, Python analytics, empathy-driven interventions, and proactive bootcamps.")

with st.expander("📥 Data Input / Generation", expanded=True):
    colA, colB, colC = st.columns([1,1,1])
    with colA:
        n_users = st.slider("Users", 40, 400, 160, step=20)
    with colB:
        days = st.slider("Days of history", 30, 120, 60, step=5)
    with colC:
        seed = st.number_input("Random seed", value=42, step=1)
    users, daily, surveys = generate_synthetic_data(n_users=n_users, days=days, seed=int(seed))
    st.success(f"Generated dataset: {len(users)} users, {len(daily)} daily logs, {len(surveys)} surveys")

daily_enriched, surveys_enriched = enrich(users, daily, surveys)

# Global filters
st.sidebar.header("Filters")
cohort_opts = ["All"] + sorted(users["cohort"].unique().tolist())
cohort_sel = st.sidebar.selectbox("Cohort", cohort_opts, index=0)
date_min, date_max = pd.to_datetime(daily_enriched["date"]).min(), pd.to_datetime(daily_enriched["date"]).max()
date_range = st.sidebar.date_input("Date range", [date_min, date_max], min_value=date_min, max_value=date_max)

mask = (pd.to_datetime(daily_enriched["date"])>=pd.to_datetime(date_range[0])) & (pd.to_datetime(daily_enriched["date"])<=pd.to_datetime(date_range[1]))
if cohort_sel != "All":
    uids = users.loc[users["cohort"]==cohort_sel, "user_id"]
    mask = mask & (daily_enriched["user_id"].isin(uids))

view = daily_enriched.loc[mask].copy()

# KPIs
kcol1, kcol2, kcol3, kcol4 = st.columns(4)
kcol1.metric("Avg PHI (Preventive Health Index)", f"{view['phi'].mean():.1f}")
kcol2.metric("Avg Stress", f"{view['stress_level'].mean():.2f}")
kcol3.metric("High-Risk Days", int((view['risk']=='High').sum()))
kcol4.metric("Avg Productivity", f"{view['productivity_score'].mean():.1f}")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Cohort Analytics","🧍 Individual Coach","🧪 Predictive Model","🗓️ Bootcamp Planner","🗂️ Data & Export"])

with tab1:
    c1, c2 = st.columns([1.2, 1])
    with c1:
        fig = px.line(view.groupby("date")["phi"].mean().reset_index(), x="date", y="phi", title="Average PHI over time")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        risk_counts = view["risk"].value_counts().rename_axis("risk").reset_index(name="count")
        fig2 = px.pie(risk_counts, names="risk", values="count", title="Risk distribution")
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns([1,1])
    with c3:
        corr = view[["sleep_hours","water_liters","screen_time_hours","exercise_minutes","stress_level","steps","resting_hr","productivity_score","phi"]].corr(numeric_only=True)
        fig3 = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
        st.plotly_chart(fig3, use_container_width=True)
    with c4:
        # Success Genome: feature importance from a quick model on filtered set
        try:
            model, auc, imp = train_predictive_model(view)
            fig4 = px.bar(imp, x="feature", y="importance", title=f"Success Genome – Drivers of Risk (AUC={auc:.2f})")
            st.plotly_chart(fig4, use_container_width=True)
        except Exception as e:
            st.info("Not enough label variety for modeling in current slice. Try broader filters.")

with tab2:
    user_id = st.selectbox("Select user", sorted(view["user_id"].unique().tolist()))
    u_df = view[view["user_id"]==user_id].sort_values("date")
    st.subheader(f"User #{user_id}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg PHI", f"{u_df['phi'].mean():.1f}")
    c2.metric("Avg Stress", f"{u_df['stress_level'].mean():.2f}")
    c3.metric("High-Risk Days", int((u_df['risk']=='High').sum()))

    figu = px.line(u_df, x="date", y=["phi","productivity_score"], title="PHI & Productivity trend")
    st.plotly_chart(figu, use_container_width=True)

    st.markdown("**Recommendations**")
    recs = recommend_interventions(u_df)
    for r in recs:
        st.write("• ", r)

    # Recent survey & mood tags
    sv = surveys_enriched[surveys_enriched["user_id"]==user_id].tail(3)
    if len(sv):
        st.markdown("**Recent Mood Notes**")
        for _, row in sv.iterrows():
            st.write(f"_{row['week_start']}_ — {row['mood_text']}  \nTags: `{', '.join(row['tags'])}` (mood={row['mood_score']}, energy={row['energy_level']})")

    # Predictive alert (simple PHI slope)
    if len(u_df) >= 5:
        recent = u_df.tail(5)["phi"].values
        slope = np.polyfit(np.arange(len(recent)), recent, 1)[0]
        if slope < -1.5:
            st.error("⚠️ Predictive Alert: PHI trending down sharply in last 5 days. Recommend immediate intervention.")
        else:
            st.success("PHI trend is stable in the last 5 days.")

with tab3:
    st.subheader("Predictive Model – Weekly Risk Classifier")
    try:
        model, auc, imp = train_predictive_model(view)
        st.write(f"**AUC:** {auc:.3f} (higher is better)")
        st.dataframe(imp.reset_index(drop=True))
    except Exception as e:
        st.info("Model training requires sufficient label variety; broaden filters if needed.")

with tab4:
    st.subheader("Cohort Bootcamp Planner")
    plan = cohort_bootcamp_plan(view)
    st.dataframe(plan, use_container_width=True)
    st.caption("Plan auto-generated from cohort gaps (sleep, hydration, exercise, screen-time, stress).")

with tab5:
    st.subheader("Preview Data")
    st.dataframe(view.head(200), use_container_width=True)
    st.download_button("⬇️ Download current slice as CSV", data=view.to_csv(index=False), file_name="health360_slice.csv", mime="text/csv")

st.markdown("---")
with st.expander("ℹ️ About this Demo & How to Deploy", expanded=False):
    st.markdown("""
**Health360** demonstrates: structured data design, Python analytics, empathy-first recommendations, proactive bootcamps, and predictive insights.

### 🚀 One-click Deploy (Hugging Face Spaces)
1. Create a new Space -> **Streamlit**.
2. Upload these files: `app.py`, `requirements.txt` (and optionally `/data`).
3. Click **Commit** -> your live URL is ready (`https://huggingface.co/spaces/<you>/Health360`).

### 📦 Local Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
""")
