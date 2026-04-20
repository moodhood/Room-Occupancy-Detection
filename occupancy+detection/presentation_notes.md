# Presentation Notes

## One-Minute Summary

This project predicts whether a room is occupied using environmental sensor data from the UCI Occupancy Detection dataset. I compared Logistic Regression and Random Forest using temperature, humidity, light, CO2, and humidity ratio as input features. Logistic Regression with feature scaling produced the best overall result, with a holdout F1 score of 0.9824. The most important predictors were light and CO2, which makes sense because both are strongly affected by human presence in a room.

## Suggested Slide Order

1. Problem motivation: why occupancy detection matters in smart buildings
2. Dataset overview: source, size, features, and class balance
3. Methods: Logistic Regression vs Random Forest
4. Evaluation metrics: accuracy, precision, recall, F1, confusion matrix
5. Validation comparison table
6. Holdout test results table
7. Random Forest feature importance and interpretation
8. Final conclusion and future work

## Results To Say Out Loud

- Logistic Regression with scaling was the best model on both validation and holdout data.
- Validation F1 for scaled Logistic Regression was 0.9694.
- Holdout F1 for scaled Logistic Regression was 0.9824.
- Random Forest still performed well, but it had lower precision and lower F1 overall.
- Light and CO2 were the strongest predictors of occupancy.

## If Asked About Limitations

- I only compared two algorithms.
- I did not engineer extra time-based features from the timestamp.
- The project used a single dataset, so generalization to other buildings was not tested.
