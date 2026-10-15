-- For every unit that failed first pass: what failed, and did it pass on
-- retest? A high retest pass rate (no fault found) for one failure mode
-- points at the test, not the unit.
WITH ordered AS (
    SELECT run_id, dut_serial, station, outcome,
           ROW_NUMBER() OVER (PARTITION BY dut_serial ORDER BY started_at) AS attempt
    FROM runs
),
first_fail AS (
    SELECT o.dut_serial, o.station AS first_station, m.name AS failure_mode
    FROM ordered o
    JOIN measurements m ON m.run_id = o.run_id AND m.passed = 0
    WHERE o.attempt = 1
),
retest AS (
    SELECT dut_serial, station AS retest_station, outcome AS retest_outcome
    FROM ordered WHERE attempt = 2
)
SELECT f.failure_mode,
       f.first_station,
       COUNT(*)                                          AS failed_units,
       SUM(r.retest_outcome = 'PASS')                    AS passed_retest,
       ROUND(100.0 * SUM(r.retest_outcome = 'PASS') / COUNT(r.dut_serial), 1) AS no_fault_found_pct
FROM first_fail f
LEFT JOIN retest r ON r.dut_serial = f.dut_serial
GROUP BY f.failure_mode, f.first_station
ORDER BY f.failure_mode, f.first_station;
