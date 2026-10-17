# test-failure-analytics

[![CI](https://github.com/KyleEGibson/test-failure-analytics/actions/workflows/ci.yml/badge.svg)](https://github.com/KyleEGibson/test-failure-analytics/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)

Failure, yield and trend analysis on manufacturing test data: SPC control charts, Pareto, and root cause isolation that can tell a real defect from a test-equipment or process problem.

[Live report](https://kyleegibson.github.io/test-failure-analytics/) (CI rebuilds it on every push) | [8D root cause analysis](docs/root_cause_analysis.md) | dashboard: `streamlit run app/dashboard.py`

![Report overview](docs/images/report_overview.png)

## The two datasets

| | Data | What it's for |
|---|---|---|
| EOS-100 production | 12 weeks, 1,247 units, in the same DB schema as my [EOS-100 test station](https://github.com/KyleEGibson/eo-sensor-test-station) | root cause isolation with a known answer: three causes are planted in synthetic data and the analysis has to find them without looking |
| UCI SECOM | 1,567 real units from a semiconductor line, 590 process sensors, pass/fail | seeing what messy, drifting real data can and can't tell you |

### EOS-100: the Pareto points the wrong way
The biggest failure category (55% of failures) isn't a sensor defect at all. 86% of those units passed when retested on the other bench. The analysis gets each cause right:

![Root cause findings](docs/images/report_findings.png)

An X-bar chart on the raw measurement flagged the drifting bench in week 9, two weeks before the first actual failure:

![Station drift](docs/images/report_drift.png)

The naive approach gets 2 of the 3 causes wrong. Ranking factors by p-value over the whole history blames a component lot that just happened to be on the line those same weeks, and an operator who's really a stand-in for their shift. What fixed it:
1. When did it start? Onset is the first week that breaks a baseline p-chart.
2. What changed then? Screen factors only inside the onset window, Bonferroni-corrected.
3. Which factor is real? Stratified chi-square holding each other factor fixed. Factors that stop mattering get explained away.
4. Unit or test? No-fault-found rate on retest.

There's a test (`test_naive_screen_would_have_been_wrong`) that keeps the naive result around as a check.

### SECOM: how you validate changes the answer
- Data quality: 144 of 590 signals dropped (28 over 50% missing, 116 constant) and 10,868 missing values imputed. Every decision gets logged.
- Yield trend: the p-chart says the process wasn't stable. There's a high-failure stretch in July-August and a lower level from late August on.
- The top signal is really a clock: 25 signals pass the FDR test, but the strongest one (S059) is tracking which era a unit came from, and it stops separating on later units. Only S103 holds up in both periods.
- Model validation:

  | Validation method | AUC |
  |---|---|
  | shuffled 5-fold CV | 0.76 (looks useful) |
  | train on earlier units, test on later | 0.47 (coin flip) |

  Shuffling lets the model learn the era instead of the failure.

![SECOM signal stability](docs/images/report_secom.png)

## Dashboard
Filters, a root-cause drill-down per failure mode, and you can upload any test station `results.db`. It works on databases straight from the EOS-100 station, which don't have lot data.

![Dashboard](docs/images/dashboard.png)

## What it covers
| Job requirement | Where |
|---|---|
| Failure and trend analysis | `eos.py` (onset, factor isolation), `spc.py`, p-charts, X-bar early warning |
| Root cause analysis and corrective action | [`docs/root_cause_analysis.md`](docs/root_cause_analysis.md) (8D, fishbone, 5 Whys) |
| SQL | [`src/failure_analytics/sql/`](src/failure_analytics/sql): window functions, CTEs, joins to manufacturing records |
| Statistics | Western Electric rules, Cpk, chi-square and stratified tests, Mann-Whitney U, Benjamini-Hochberg FDR |
| Data cleaning | logged SECOM cleaning, checksum-pinned source data, chronological validation |
| Reporting | static Plotly report (light/dark, table view per chart), Streamlit app |
| Testing | 49 pytest tests: SPC math vs hand calcs, known-answer tests, SQL checked against pandas, headless dashboard tests |

## Running it
```bash
python -m venv .venv
.venv\Scripts\activate                       # Windows  (source .venv/bin/activate elsewhere)
pip install -e ".[dev,dashboard]"
pytest                                       # 49 tests, ~15 s
python -m failure_analytics report --out site   # site/index.html
python -m failure_analytics analyze             # findings in the terminal
python -m failure_analytics analyze --db path/to/results.db   # any test station database
streamlit run app/dashboard.py
```

## Layout
```
data/raw/secom.zip              UCI SECOM (CC BY 4.0), SHA-256 pinned
src/failure_analytics/
  secom.py                      load, clean, time split
  spc.py                        I-MR, p-chart, Western Electric rules, Cpk
  signals.py                    signal ranking, FDR, stability, CV comparison
  eos_generator.py              synthetic production history (answer key in the docstring)
  eos.py                        onset, factor screen, stratified isolation, retest, X-bar
  sql/                          the queries behind the EOS analysis
  charts.py, report.py          Plotly figures and the static report
app/dashboard.py                Streamlit dashboard
docs/root_cause_analysis.md     8D report
tests/                          SPC, signals, EOS known answers, report, dashboard
```

## Bugs found while building it
Each one is also in its commit message.

| Symptom | Cause | Fix |
|---|---|---|
| every FDR q-value came out NaN | one untestable signal's NaN p-value spread through Benjamini-Hochberg | leave NaN p-values out of the test count, plus a regression test |
| root cause pinned on the wrong factor (2 of 3) | whole-history p-value ranking gets fooled by time-correlated and nested factors | onset window + stratified chi-square |
| report charts were blank | Plotly loaded after the inline chart scripts | load it in `<head>`, and a test checks the order |
| trend paragraph didn't match the chart | I wrote it by hand from the daily analysis, but the chart is weekly | generate the text from the rule violations |
| retest pass rate was 82% in the report and 86% in SQL | different denominators (failed vs actually retested units) | use retested units everywhere, plus a regression test |
| dashboard crashed comparing readings by station | duplicate `station` column after a merge | the headless `AppTest` caught it (a screenshot wouldn't have) |

Data: McCann, M. & Johnston, A. (2008). SECOM. UCI Machine Learning Repository. https://doi.org/10.24432/C54305 (CC BY 4.0).
