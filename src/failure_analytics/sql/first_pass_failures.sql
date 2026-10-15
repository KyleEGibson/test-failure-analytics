-- One row per (first-pass run, failing measurement).
WITH first_runs AS (
    SELECT run_id FROM (
        SELECT run_id, ROW_NUMBER() OVER (PARTITION BY dut_serial ORDER BY started_at) AS attempt
        FROM runs
    ) WHERE attempt = 1
)
SELECT m.run_id, m.name AS failure_mode, m.value, m.low, m.high, m.units
FROM measurements m
JOIN first_runs f ON f.run_id = m.run_id
WHERE m.passed = 0;
