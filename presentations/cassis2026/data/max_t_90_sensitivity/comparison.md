# Max-t sensitivity: 95% versus 90%

Same completed fits, paired differences, and 10,000 bootstrap draws. All 60 model-pair by training-size comparisons remain in each problem's family. Only primary-loss comparison confidence changes. Prediction coverage, interval-width inference, rerun ribbons, and the presentation are unchanged.

| Problem | Clear winners at 95% | Clear winners at 90% | Newly clear training sizes |
|---|---:|---:|---|
| 2020: Rushing yards | 3/6 | 3/6 | None |
| 2021: Pass completion | 5/6 | 5/6 | None |
| 2022: Punt returns | 0/6 | 0/6 | None |
| 2023: Sack probability | 0/6 | 0/6 | None |
| 2024: Tackle candidate | 6/6 | 6/6 | None |
| 2025: Man/Zone | 3/6 | 4/6 | 50 games: Boosted trees |

Total: 17/36 at 95%; 18/36 at 90%.

## Newly clear: 2025 Man/Zone, 50 training games

Boosted trees now beats all four alternatives. The limiting comparison is against Sumer Transformer. Differences below are Boosted trees minus Sumer Transformer, so negative favors Boosted trees.

| Confidence | Lower bound | Upper bound |
|---|---:|---:|
| 95% | -0.005405 | 0.000214 |
| 90% | -0.005112 | -0.000079 |

A clear winner must have lower mean loss than all four alternatives under the simultaneous intervals. No clear winner does not imply equivalence. These comparisons describe repeated procedures on the available cohorts. This is a sensitivity analysis to a less conservative confidence level, not new model fitting.

The original 95% intervals and winner declarations were reproduced before evaluating 90%. Source checksums and bootstrap registry hashes match. Tables retain full precision.
