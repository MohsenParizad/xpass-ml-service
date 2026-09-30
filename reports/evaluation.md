# xPass model evaluation (v1.0.0)

## Experiments (test set, split by match)

| model    | resampling   |   brier |   log_loss |   roc_auc |   accuracy |    mcc |   mean_predicted |   observed_rate |
|:---------|:-------------|--------:|-----------:|----------:|-----------:|-------:|-----------------:|----------------:|
| logreg   | smote        |  0.1338 |     0.4124 |    0.8403 |     0.8195 | 0.4806 |           0.7033 |          0.7716 |
| logreg   | none         |  0.1251 |     0.3901 |    0.8444 |     0.819  | 0.4656 |           0.7665 |          0.7716 |
| catboost | smote        |  0.113  |     0.3542 |    0.8792 |     0.8366 | 0.5097 |           0.7648 |          0.7716 |
| catboost | none         |  0.1111 |     0.347  |    0.8821 |     0.837  | 0.5182 |           0.7763 |          0.7716 |

**Selected:** catboost with resampling = `none` (lowest Brier score)

## Calibration (quantile bins)

|   mean_predicted |   observed |
|-----------------:|-----------:|
|            0.304 |      0.269 |
|            0.432 |      0.405 |
|            0.609 |      0.551 |
|            0.78  |      0.799 |
|            0.846 |      0.882 |
|            0.89  |      0.9   |
|            0.949 |      0.952 |
|            0.977 |      0.967 |
|            0.988 |      1     |
|            0.997 |      1     |

## Evaluation gate

- brier: PASS
- log_loss: PASS
- roc_auc: PASS

**Gate result: PASSED - model may be deployed**

## Top 10 players (xPass score, >= 51 passes)

| player                    | team    | position                |   passes |   completed |   expected |   completion_rate |   score |
|:--------------------------|:--------|:------------------------|---------:|------------:|-----------:|------------------:|--------:|
| Griedge Mbock Bathy Nka   | France  | Right Center Back       |      147 |         139 |   123.514  |            0.9456 |  0.1053 |
| Nicky Evrard              | Belgium | Goalkeeper              |       98 |          83 |    73.0893 |            0.8469 |  0.1011 |
| Laura De Neve             | Belgium | Left Center Back        |      139 |         118 |   104.992  |            0.8489 |  0.0936 |
| Elena Linari              | Italy   | Left Center Back        |      148 |         137 |   123.517  |            0.9257 |  0.0911 |
| Martina Rosucci           | Italy   | Left Defensive Midfield |      100 |          85 |    76.3791 |            0.85   |  0.0862 |
| María Pilar León Cebrián  | Spain   | Left Center Back        |      352 |         319 |   292.684  |            0.9062 |  0.0748 |
| Mary Alexandra Earps      | England | Goalkeeper              |       77 |          67 |    61.4742 |            0.8701 |  0.0718 |
| Linda Brigitta Sembrant   | Sweden  | Right Center Back       |       79 |          69 |    63.7788 |            0.8734 |  0.0661 |
| Laura Giuliani            | Italy   | Goalkeeper              |       59 |          54 |    50.6456 |            0.9153 |  0.0569 |
| Glódís Perla Viggósdóttir | Iceland | Right Center Back       |       91 |          75 |    69.8329 |            0.8242 |  0.0568 |