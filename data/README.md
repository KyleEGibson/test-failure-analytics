# Data

## `raw/secom.zip` — real production data
**SECOM**: McCann, M. & Johnston, A. (2008). *SECOM* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C54305

Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The archive is included unmodified so the analysis is reproducible offline. `secom.load_raw()` checks it against a pinned SHA-256 and refuses to run if the file has changed.

- 1,567 production units from a semiconductor manufacturing line, July–October 2008
- 590 anonymized process sensor signals per unit (with missing values)
- An in-house line test result (pass/fail, 104 fails) and a timestamp

## `generated/eos_production.db` — synthetic production test history
Created by `python -m failure_analytics generate` (deterministic, seed 2026). It is not committed; the report pipeline regenerates it.

It models 12 weeks of EOS-100 sensor acceptance testing using the **same database schema as the [EOS-100 test station](https://github.com/KyleEGibson/eo-sensor-test-station)** (`runs`, `test_results`, `measurements`), plus a `units` table standing in for manufacturing records (component lot numbers). The station's real `results.db` files can be analyzed with the same queries.

Three problems are deliberately planted in the generated data, so the analysis can be checked against a known answer. They are documented in `src/failure_analytics/eos_generator.py`, and the analysis does not read that file.
