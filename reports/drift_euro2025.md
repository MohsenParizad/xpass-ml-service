# Drift report: euro2025 vs. training data (model v1.0.0)

- Passes scored: 19667
- Passes against opponents without power index: 12.9%
- Prediction PSI: 0.0057 (ok)
- Brier score: reference 0.1111 -> new 0.1089

## Feature drift (PSI)

| feature           |    psi | status   |
|:------------------|-------:|:---------|
| ground_pass       | 0      | ok       |
| low_pass          | 0      | ok       |
| high_pass         | 0.0001 | ok       |
| defensive_third   | 0.0001 | ok       |
| midfield          | 0.0006 | ok       |
| attacking_third   | 0.0017 | ok       |
| backward_pass     | 0      | ok       |
| square_pass       | 0      | ok       |
| forward_pass      | 0      | ok       |
| pass_length       | 0.0032 | ok       |
| outplayed_players | 0.0005 | ok       |
| goal_diff         | 0.0098 | ok       |
| power_index       | 0.2923 | ALERT    |

## Decision: RETRAINING RECOMMENDED
- feature drift (PSI > 0.25)
- 13% of passes against opponents without power index