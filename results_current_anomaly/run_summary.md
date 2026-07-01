# Kempower Anomaly Ensemble Run Summary

Data: `..\private_data\charger_telemetry_full_private.csv`

Target: `label_current_anomaly`

Alert mode: `balanced`

Warning threshold: `0.491482`

Critical threshold: `0.963807`

Rows loaded: 100,000

## Split Summary

| split      |   rows |   sessions |   positive_rate |
|:-----------|-------:|-----------:|----------------:|
| train      |  70397 |        436 |       0.0883418 |
| validation |  15534 |         93 |       0.0942449 |
| test       |  14069 |         94 |       0.103063  |

## Best Test Model by F1

|   threshold |   accuracy |   precision |   recall |       f1 |   false_alarm_rate |   missed_anomaly_rate |   alert_rate |   positive_rate |   roc_auc |   pr_auc | dataset   | model                        | target                |   fit_seconds |
|------------:|-----------:|------------:|---------:|---------:|-------------------:|----------------------:|-------------:|----------------:|----------:|---------:|:----------|:-----------------------------|:----------------------|--------------:|
|    0.479755 |   0.979672 |    0.924198 | 0.874483 | 0.898653 |         0.00824154 |              0.125517 |    0.0975194 |        0.103063 |  0.959615 | 0.928872 | test      | supervised_weighted_ensemble | label_current_anomaly |           nan |

## Threshold Mode Comparison

|   threshold |   accuracy |   precision |   recall |       f1 |   false_alarm_rate |   missed_anomaly_rate |   alert_rate |   positive_rate |   roc_auc |   pr_auc | dataset   | alert_mode   | target                |
|------------:|-----------:|------------:|---------:|---------:|-------------------:|----------------------:|-------------:|----------------:|----------:|---------:|:----------|:-------------|:----------------------|
|    0.491482 |   0.978819 |    0.9273   | 0.862069 | 0.893495 |         0.00776607 |              0.137931 |    0.0958135 |        0.103063 |  0.950116 | 0.921983 | test      | balanced     | label_current_anomaly |
|    0.498047 |   0.978677 |    0.928465 | 0.85931  | 0.89255  |         0.00760758 |              0.14069  |    0.095387  |        0.103063 |  0.950116 | 0.921983 | test      | conservative | label_current_anomaly |

## Output Files

- `tables/overall_anomaly_rates.csv`
- `tables/rates_by_scenario.csv`
- `tables/model_comparison.csv`
- `tables/anomaly_predictions_test.csv`
- `tables/anomaly_reason_ranking.csv`
- `tables/threshold_modes.csv`
- `figures/*.png`
- `models/*.joblib`
