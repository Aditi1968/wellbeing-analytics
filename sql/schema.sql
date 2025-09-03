-- Health360 SQL Schema (SQLite compatible)
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS users (
  user_id       INTEGER PRIMARY KEY,
  name          TEXT NOT NULL,
  age           INTEGER CHECK(age >= 0),
  gender        TEXT CHECK(gender IN ('Male','Female','Other')),
  cohort        TEXT
);

CREATE TABLE IF NOT EXISTS daily_logs (
  log_id            INTEGER PRIMARY KEY,
  user_id           INTEGER NOT NULL REFERENCES users(user_id),
  date              TEXT NOT NULL,                -- ISO date
  sleep_hours       REAL,
  water_liters      REAL,
  screen_time_hours REAL,
  exercise_minutes  REAL,
  stress_level      REAL,
  steps             INTEGER,
  resting_hr        INTEGER,
  productivity_score REAL,
  phi               REAL,
  risk              TEXT
);

CREATE INDEX IF NOT EXISTS idx_daily_user_date ON daily_logs(user_id, date);
CREATE INDEX IF NOT EXISTS idx_daily_date ON daily_logs(date);
CREATE INDEX IF NOT EXISTS idx_daily_risk ON daily_logs(risk);

CREATE TABLE IF NOT EXISTS surveys (
  survey_id    INTEGER PRIMARY KEY,
  user_id      INTEGER NOT NULL REFERENCES users(user_id),
  week_start   TEXT NOT NULL,       -- ISO date (Monday start)
  mood_text    TEXT,
  mood_score   INTEGER,
  energy_level INTEGER,
  tags         TEXT                 -- comma-separated tags extracted from mood_text
);

CREATE INDEX IF NOT EXISTS idx_surveys_user_week ON surveys(user_id, week_start);
