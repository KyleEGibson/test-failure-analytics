# Root Cause Analysis: EOS-100 First-Pass Yield Decline

Report 8D-EOS-2026-01 | Product: EOS-100 electro-optical sensor | Period: production weeks 1-12
Data: 1,247 units, 1,409 test runs (`python -m failure_analytics analyze`)

> The production data is synthetic, with planted causes, so the method can be checked against a known answer. The analysis code never reads the answer key; the known-answer tests in `tests/test_eos.py` check that it recovers it.


## D1: Team
Test engineering (data analysis), production test operators, manufacturing engineering, supplier quality, facilities.

## D2: Problem description
First-pass yield (FPY) fell from 99% in weeks 1-3 to 45% in weeks 11-12. Three failure modes account for 94% of first-pass failures:

| Failure mode | Requirement | First-pass failures | Share |
|---|---|---:|---:|
| Bus voltage above 29.0 V | SYS-005 | 100 | 55% |
| 90° slew time above 2.0 s | SYS-022 | 45 | 25% |
| Cooldown time above 15.0 s | SYS-010 | 26 | 14% |
| Other (dead pixels, linearity, temperature) | | 11 | 6% |

Is / is not:

| | Is | Is not |
|---|---|---|
| Voltage | BENCH-02, weeks 11-12 | BENCH-01; any week before 11 |
| Slew | Units with gimbal lot G-2603 | Lot G-2601 or G-2602, on either firmware |
| Cooldown | Swing shift (14:00-22:00), weeks 6-12 | Day shift; weeks 1-5 |

## D3: Containment
- Quarantine all units that failed bus voltage on BENCH-02 and retest them on BENCH-01.
- Hold remaining gimbal lot G-2603 stock.
- Move cooldown testing to the day shift until facilities confirm the lab temperature.

## D4: Root cause

### Method
1. When did it start? Each mode's weekly failure rate was compared with a p-chart from weeks 1-3; the first week to break the upper limit is the onset.
2. What changed then? Seven factors (station, shift, operator, firmware, gimbal lot, cooler lot, detector lot) were screened with chi-square tests, restricted to weeks after onset, Bonferroni-corrected.
3. Which factor is real? Each significant factor was re-tested with every other one held fixed (stratified chi-square). A factor that stops separating failures is explained away.
4. Unit or test? Retest outcomes: real defects fail again, while test-equipment problems pass on another bench.

The naive alternative, ranking factors by p-value over the whole history, blamed the wrong factor for two of the three causes (details below). The onset window and the stratified test are what fixed that.

### Fishbone (6M)
```
  Measurement   BENCH-02 voltage calibration drift      ROOT CAUSE 1
  Material      Gimbal motor lot G-2603 (slow slew)     ROOT CAUSE 2
  Environment   Lab HVAC, swing shift                   ROOT CAUSE 3
  Method        Firmware 2.4.2 (week 5)                 ruled out
  Manpower      Operator kgibson                        explained by shift
  Machine       Station software, comm retries          no association
```

### Root cause 1: BENCH-02 voltage measurement drift (test equipment)
- Evidence: from week 11, 98% of BENCH-02 units fail versus 0% on BENCH-01. 86% of retested BENCH-02 failures (82 of 95) passed on retest at BENCH-01 (no fault found).
- Early warning: an X-bar chart of BENCH-02's raw voltage readings left its control limits in week 9, two weeks before the first failure, which pass/fail alone wouldn't show.
- 5 Whys: units fail voltage -> BENCH-02 reads high -> its measurement has drifted upward ~0.35 V/week -> there is no periodic check against a reference -> the calibration interval was the only control, and it is longer than the drift takes to matter.
- Naive screen got this wrong: over the whole history, the strongest association is gimbal lot G-2603, which was on the line during the same weeks. Within the onset window every unit has lot G-2603, so the lot cannot explain the failures and drops out.

### Root cause 2: gimbal motor lot G-2603 (supplier)
- Evidence: 8.9% of G-2603 units fail slew time; 0 of 743 units from other lots fail. Onset in week 7, when G-2603 entered the line.
- Firmware cleared: firmware 2.4.2 shipped in week 5 and also correlates with the failures. With firmware held at 2.4.2, lot G-2602 had 0 failures in 321 units and G-2603 had 45 in 504.
- Retest: 32% (14 of 44 retested) pass on retest. These units sit near the 2.0 s limit, so measurement repeatability decides the outcome. They are real marginal units, not false failures.
- 5 Whys: slow slews -> G-2603 motors are slower (mean 1.90 s vs 1.56 s) -> lot-to-lot torque variation -> no incoming inspection of slew performance -> the supplier's certificate of conformance was accepted without a sample check.

### Root cause 3: swing-shift lab temperature (environment)
- Evidence: from week 6, cooldown on the swing shift averages ~13.8 s versus ~9.9 s on day shift (limit 15.0 s); 7.3% of swing-shift units fail and 0% of day-shift units.
- Operator explained away: operator kgibson has the highest failure rate after onset (10.0% vs 1.5% for everyone else). But every operator works one shift, and with shift held fixed the operator difference is not statistically significant (swing shift: 18 of 180 vs 8 of 174). Operator stands in for shift.
- Retest: 92% (24 of 26) pass on retest, because retests run in cooler conditions. The cause is intermittent and environmental, not in the unit.
- 5 Whys: slow cooldown -> the cooler starts from a warmer ambient -> the lab is warmer in the afternoon -> the HVAC cannot hold setpoint in summer -> ambient temperature is not recorded or limited during testing.

### Background (no assignable cause)
Dead pixels (6), linearity (3) and maximum temperature (2) show no association with any factor and no onset week. That is consistent with random detector defects at a stable rate, so these modes are not worth an investigation.

## D5: Corrective actions
| # | Action | Owner |
|---|---|---|
| 1 | Recalibrate BENCH-02; re-disposition its falsely failed units | Test engineering |
| 1 | Daily reference-unit check on every bench, with SPC on the reading | Test engineering |
| 2 | Return lot G-2603 with this data; request supplier 8D | Supplier quality |
| 2 | Incoming sample slew test for every gimbal motor lot | Manufacturing engineering |
| 3 | Service lab HVAC; log ambient temperature with every test run | Facilities / test engineering |
| 3 | Station refuses cooldown test when ambient is outside 20-24 °C | Test engineering |

## D6: Verification
- Re-run `failure_analytics analyze` on the following four weeks of data. Each mode's failure rate must return inside its baseline p-chart limits.
- BENCH-01 vs BENCH-02 mean voltage difference must stay inside the X-bar limits.

## D7: Prevention
- SPC on the measurements, not just pass/fail: the voltage drift was visible two weeks before it caused a single failure.
- Record ambient conditions and component lots in the results database, so the next investigation can be run directly with queries.
- Validate analyses in time order: see the SECOM section of the report, where shuffled cross-validation overstated a model's AUC (0.76 vs 0.47).

## D8: Closure
Close when D6 verification passes for four consecutive weeks.
