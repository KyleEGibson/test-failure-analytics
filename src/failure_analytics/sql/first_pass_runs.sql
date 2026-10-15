-- First test run of every unit (first-pass yield excludes retests),
-- joined to manufacturing records when a units table exists. Production week is counted from the
-- first day in the database (week 1).
WITH ordered AS (
    SELECT r.*,
           ROW_NUMBER() OVER (PARTITION BY r.dut_serial ORDER BY r.started_at) AS attempt
    FROM runs r
),
origin AS (SELECT MIN(julianday(date(started_at))) AS day0 FROM runs)
SELECT o.run_id,
       o.dut_serial                                              AS serial,
       o.started_at,
       CAST((julianday(date(o.started_at)) - origin.day0) / 7 AS INTEGER) + 1 AS week,
       CASE WHEN CAST(strftime('%H', o.started_at) AS INTEGER) < 14 THEN 'day' ELSE 'swing' END AS shift,
       o.station,
       o.operator,
       o.dut_firmware                                            AS firmware,
       COALESCE(u.gimbal_lot, 'unknown')                         AS gimbal_lot,
       COALESCE(u.cooler_lot, 'unknown')                         AS cooler_lot,
       COALESCE(u.detector_lot, 'unknown')                       AS detector_lot,
       o.outcome = 'FAIL'                                        AS failed
FROM ordered o
LEFT JOIN units u ON u.serial = o.dut_serial
CROSS JOIN origin
WHERE o.attempt = 1
ORDER BY o.started_at;
