-- Failing measurements on first-pass runs, most frequent first, with the
-- cumulative share used to draw a Pareto chart. A unit failing two
-- measurements counts once under each.
WITH first_runs AS (
    SELECT run_id FROM (
        SELECT run_id, ROW_NUMBER() OVER (PARTITION BY dut_serial ORDER BY started_at) AS attempt
        FROM runs
    ) WHERE attempt = 1
),
counts AS (
    SELECT m.name AS failure_mode, COUNT(*) AS failures
    FROM measurements m
    JOIN first_runs f ON f.run_id = m.run_id
    WHERE m.passed = 0
    GROUP BY m.name
)
SELECT failure_mode,
       failures,
       ROUND(100.0 * failures / SUM(failures) OVER (), 1)                         AS share_pct,
       ROUND(100.0 * SUM(failures) OVER (ORDER BY failures DESC, failure_mode
                                         ROWS UNBOUNDED PRECEDING)
                   / SUM(failures) OVER (), 1)                                     AS cumulative_pct
FROM counts
ORDER BY failures DESC, failure_mode;
