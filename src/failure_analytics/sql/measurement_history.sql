-- Every first-pass reading of one measurement, in test order, for SPC.
WITH ordered AS (
    SELECT run_id, station, started_at,
           ROW_NUMBER() OVER (PARTITION BY dut_serial ORDER BY started_at) AS attempt
    FROM runs
)
SELECT o.started_at, o.station, m.value, m.low, m.high, m.units, m.passed
FROM measurements m
JOIN ordered o ON o.run_id = m.run_id
WHERE o.attempt = 1 AND m.name = :measurement
ORDER BY o.started_at;
