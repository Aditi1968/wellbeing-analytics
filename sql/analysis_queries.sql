-- Top-level KPIs
-- 1) Average PHI and stress in a date range
--   Replace :start and :end with actual ISO dates, e.g., '2025-07-01'
SELECT AVG(phi) AS avg_phi, AVG(stress_level) AS avg_stress
FROM daily_logs
WHERE date BETWEEN :start AND :end;

-- 2) Risk distribution
SELECT risk, COUNT(*) AS cnt
FROM daily_logs
WHERE date BETWEEN :start AND :end
GROUP BY risk
ORDER BY cnt DESC;

-- 3) Correlation-like exploration (example using covariance approximations is limited in SQLite;
--    you can export to pandas for Pearson correlation). Here are cohort means by metric:
SELECT cohort,
       AVG(sleep_hours) AS avg_sleep,
       AVG(water_liters) AS avg_water,
       AVG(exercise_minutes) AS avg_exercise,
       AVG(screen_time_hours) AS avg_screen,
       AVG(stress_level) AS avg_stress,
       AVG(productivity_score) AS avg_productivity,
       AVG(phi) AS avg_phi
FROM daily_logs d
JOIN users u ON u.user_id = d.user_id
WHERE date BETWEEN :start AND :end
GROUP BY cohort
ORDER BY avg_phi DESC;

-- 4) Weekly aggregation (per user)
WITH weekified AS (
  SELECT user_id,
         -- week_start = date - weekday offset (SQLite lacks weekday(); precompute in ETL or use pandas)
         substr(date,1,10) AS day,
         sleep_hours, water_liters, screen_time_hours, exercise_minutes,
         stress_level, steps, resting_hr, productivity_score, phi,
         risk
  FROM daily_logs
)
SELECT user_id,
       substr(day,1,7) AS year_month, -- rough bucketing
       AVG(sleep_hours) AS sleep_avg,
       AVG(water_liters) AS water_avg,
       AVG(screen_time_hours) AS screen_avg,
       AVG(exercise_minutes) AS exercise_avg,
       AVG(stress_level) AS stress_avg,
       AVG(phi) AS phi_avg
FROM weekified
GROUP BY user_id, year_month
ORDER BY user_id, year_month;

-- 5) Simple driver exploration: compare low vs high risk averages
WITH flagged AS (
  SELECT CASE WHEN risk = 'Low' THEN 0 ELSE 1 END AS risky,
         sleep_hours, water_liters, screen_time_hours, exercise_minutes,
         stress_level, steps, resting_hr, productivity_score, phi
  FROM daily_logs
  WHERE date BETWEEN :start AND :end
)
SELECT 'low' AS bucket, AVG(sleep_hours), AVG(water_liters), AVG(exercise_minutes),
       AVG(screen_time_hours), AVG(stress_level), AVG(productivity_score), AVG(phi)
FROM flagged WHERE risky=0
UNION ALL
SELECT 'high_or_med' AS bucket, AVG(sleep_hours), AVG(water_liters), AVG(exercise_minutes),
       AVG(screen_time_hours), AVG(stress_level), AVG(productivity_score), AVG(phi)
FROM flagged WHERE risky=1;
