# Drift report: wwc2023 vs. training data (model v1.0.0)

- Passes scored: 16022
- Passes against opponents without power index: 66.0%
- Prediction PSI: 0.0327 (ok)
- Brier score: reference 0.1111 -> new 0.1341

## Feature drift (PSI)

| feature           |    psi | status   |
|:------------------|-------:|:---------|
| ground_pass       | 0.0204 | ok       |
| low_pass          | 0.0058 | ok       |
| high_pass         | 0.011  | ok       |
| defensive_third   | 0.0062 | ok       |
| midfield          | 0.0062 | ok       |
| attacking_third   | 0.0001 | ok       |
| backward_pass     | 0.0001 | ok       |
| square_pass       | 0.007  | ok       |
| forward_pass      | 0.009  | ok       |
| pass_length       | 0.0113 | ok       |
| outplayed_players | 0.0112 | ok       |
| goal_diff         | 0.0877 | ok       |
| power_index       | 1.2744 | ALERT    |

## Decision: RETRAINING RECOMMENDED
- feature drift (PSI > 0.25)
- performance drop (Brier 0.1111 -> 0.1341)
- 66% of passes against opponents without power index