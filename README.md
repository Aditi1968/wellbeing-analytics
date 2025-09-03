
# Health360 – Preventive Health & Lifestyle Intelligence

A complex, wow-factor portfolio project that demonstrates:
- Strong communication & organization (structured repository & clear UX)
- Empathy & learner-first mindset (wellbeing focus + recommendations)
- Analytical thinking (correlations, trends, predictive model)
- Proactive ownership (predictive alerts + bootcamp planner)
- Python + SQL aptitude (pandas, scikit-learn, data modeling mindset)
- Google Sheets/basic tracking (exportable CSV; easy to plug into Sheets via IMPORTDATA)

## ✨ Features
- Preventive Health Index (PHI) per day & cohort
- Risk classification (Low/Medium/High)
- Success Genome (feature importance) via logistic regression
- NLP-style mood tagging from journal text (rule-based for zero-dependency)
- Predictive alerts when PHI trends down
- Auto-generated cohort bootcamp plan (sleep, hydration, movement, detox, stress)
- Individual coach view with targeted recommendations
- Downloadable CSV for easy Sheets integration

## 🧱 Tech
- Streamlit, pandas, numpy, scikit-learn, plotly
- Synthetic dataset generator (no PII needed)

## 🚀 Deploy (Hugging Face Spaces – Streamlit)
1. Create a new Space: https://huggingface.co/new-space -> Select **Streamlit**.
2. Name it `Health360` (or anything).
3. Upload these files: `app.py`, `requirements.txt`.
4. Hit **Commit**. Your live app will be served at `https://huggingface.co/spaces/<username>/<space-name>`.

> Tip: You can also deploy on **Streamlit Community Cloud** (share.streamlit.io). Repo only needs these two files.

## 🧪 Local Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📝 License
MIT
