# Room Occupancy Detection Using Environmental Sensor Data

## Abstract

This project studies room occupancy detection as a binary classification problem using environmental sensor data. The dataset comes from the UCI Machine Learning Repository and contains 20,560 observations with measurements for temperature, humidity, light, CO2, and humidity ratio, along with a binary occupancy label. Two supervised learning methods were compared: Logistic Regression and Random Forest. The goal was to test whether a simple linear model could perform competitively against a more complex ensemble method for occupancy detection. Results showed that Logistic Regression with feature scaling produced the best overall performance, reaching an F1 score of 0.9694 on the validation split and 0.9824 on the final holdout test split. Random Forest also performed well, but it was slightly weaker overall. Feature importance analysis showed that light and CO2 were the strongest predictors of occupancy. These findings suggest that occupancy can be detected accurately using common building sensor data without relying on invasive sensing methods such as cameras.

## Introduction

Occupancy detection is an important problem in smart buildings and Internet of Things systems. Reliable occupancy estimates can improve heating, cooling, lighting, and ventilation control, which can reduce energy use and improve occupant comfort. In practice, building systems often rely on environmental sensor measurements instead of cameras because those sensors are less invasive and easier to deploy.

This project uses the UCI Occupancy Detection dataset to predict whether a room is occupied. The problem is a supervised binary classification task where the target label is `Occupancy`, and the input features are environmental sensor readings. The dataset is suitable for a class project because it is clean, has no missing values, and has already been used in published research on occupancy detection.

The main objective of this project is to compare a simple model and a more complex model on the same dataset. Logistic Regression was selected as the baseline because it is fast, interpretable, and commonly used for binary classification. Random Forest was selected because it can capture more complex nonlinear relationships between features. The main research question is whether the added complexity of Random Forest leads to better performance than Logistic Regression on this occupancy task.

## Dataset

The project uses the three occupancy files included in the provided dataset folder:

- `datatraining.txt`
- `datatest.txt`
- `datatest2.txt`

The files were used as predefined dataset splits rather than creating new random splits:

- Training split: 8,143 rows
- Validation split: 2,665 rows
- Final holdout test split: 9,752 rows

Each record contains the following fields:

- `Temperature`
- `Humidity`
- `Light`
- `CO2`
- `HumidityRatio`
- `Occupancy`

The `date` column was parsed as a timestamp but was not used directly as a predictive feature in the main experiments. This kept the comparison focused on the environmental sensor values listed in the proposal.

### Class Balance

The dataset is moderately imbalanced toward the `not occupied` class, which means accuracy alone is not enough to evaluate performance.

| Split | Not Occupied | Occupied |
| --- | ---: | ---: |
| Training | 6,414 | 1,729 |
| Validation | 1,693 | 972 |
| Test | 7,703 | 2,049 |

The generated class balance figure is available in `outputs/figures/class_balance.png`.

## Methodology

### Preprocessing

The dataset required very little cleaning. The first column in each file is a row index and was ignored when loading the data. The remaining fields were read with the `date` column parsed as a timestamp. No missing values were found, so imputation was not required.

The final feature set used for modeling was:

- Temperature
- Humidity
- Light
- CO2
- HumidityRatio

The target variable was the binary `Occupancy` label.

### Models

Three model variants were evaluated during the comparison stage:

1. Logistic Regression without feature scaling
2. Logistic Regression with standardization
3. Random Forest

The scaled Logistic Regression model used a `StandardScaler` inside a scikit-learn pipeline to avoid leakage. Random Forest was trained with 300 trees and a fixed random seed for reproducibility.

### Evaluation Metrics

The models were compared using the following metrics:

- Accuracy
- Precision
- Recall
- F1 score
- Confusion matrix

The validation split was used for model comparison. After selecting the best approach, the top models were retrained on the combined training and validation data and then evaluated once on the final holdout test split.

## Exploratory Data Analysis

Summary statistics from the training split showed that the feature scales are quite different. For example, `Light` ranges from 0 to 1546.333, while `HumidityRatio` remains near 0.004 on average. This difference in scale is one reason feature standardization was tested for Logistic Regression.

The correlation heatmap in `outputs/figures/correlation_heatmap.png` suggests that `Light` has the strongest direct association with occupancy, with `CO2` also showing a meaningful relationship. This pattern is consistent with the intuition that occupied spaces typically have more lighting activity and human-generated CO2.

## Results

### Validation Results

The validation results are shown below.

| Model | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression (scaled) | 0.9771 | 0.9470 | 0.9928 | 0.9694 |
| Logistic Regression (unscaled) | 0.9764 | 0.9469 | 0.9907 | 0.9683 |
| Random Forest | 0.9501 | 0.9458 | 0.9156 | 0.9305 |

These values are also saved in `outputs/validation_results.csv`.

The scaled Logistic Regression model was the best model on the validation split. Scaling only slightly improved the Logistic Regression result, but the improvement was consistent enough to justify using the scaled version for the final evaluation.

### Validation Confusion Matrices

The scaled Logistic Regression validation confusion matrix was:

| Actual \\ Predicted | Not Occupied | Occupied |
| --- | ---: | ---: |
| Not Occupied | 1639 | 54 |
| Occupied | 7 | 965 |

The Random Forest validation confusion matrix was:

| Actual \\ Predicted | Not Occupied | Occupied |
| --- | ---: | ---: |
| Not Occupied | 1642 | 51 |
| Occupied | 82 | 890 |

These matrices show that Logistic Regression produced far fewer false negatives than Random Forest on the validation split. That difference is reflected in the higher recall and F1 score for Logistic Regression.

### Final Holdout Test Results

After model selection, the scaled Logistic Regression and Random Forest models were retrained on the combined training and validation data and evaluated on the final holdout split.

| Model | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression (scaled) | 0.9925 | 0.9718 | 0.9932 | 0.9824 |
| Random Forest | 0.9867 | 0.9436 | 0.9961 | 0.9691 |

These values are saved in `outputs/test_results.csv`.

The scaled Logistic Regression model again produced the best F1 score, confirming that the simpler model was sufficient for this dataset. Random Forest achieved slightly higher recall, but it did so at the cost of lower precision and lower overall F1 score.

### Holdout Test Confusion Matrices

The scaled Logistic Regression holdout confusion matrix was:

| Actual \\ Predicted | Not Occupied | Occupied |
| --- | ---: | ---: |
| Not Occupied | 7644 | 59 |
| Occupied | 14 | 2035 |

The Random Forest holdout confusion matrix was:

| Actual \\ Predicted | Not Occupied | Occupied |
| --- | ---: | ---: |
| Not Occupied | 7581 | 122 |
| Occupied | 8 | 2041 |

The Logistic Regression model made fewer total errors on the holdout set, especially fewer false positives than Random Forest.

## Model Interpretation

### Logistic Regression Coefficients

The largest absolute Logistic Regression coefficients were:

| Feature | Coefficient |
| --- | ---: |
| Light | 4.2071 |
| CO2 | 1.7393 |
| Temperature | -1.3446 |
| Humidity | -0.5862 |
| HumidityRatio | 0.5789 |

This suggests that higher light levels and higher CO2 values were strong indicators of occupancy in the fitted model. The coefficient table is saved in `outputs/logistic_regression_coefficients.csv`.

### Random Forest Feature Importance

The Random Forest feature importance values were:

| Feature | Importance |
| --- | ---: |
| Light | 0.6127 |
| CO2 | 0.2272 |
| Temperature | 0.0962 |
| HumidityRatio | 0.0404 |
| Humidity | 0.0235 |

The feature importance plot is saved in `outputs/figures/random_forest_feature_importance.png`.

Both models indicate that `Light` and `CO2` are the most informative signals. This is a sensible result because these variables change noticeably when people are present in a room.

## Discussion

The main goal of the project was to compare Logistic Regression and Random Forest on the room occupancy detection task. Based on the results, Logistic Regression performed better than Random Forest on both the validation and holdout splits. This is an important result because it shows that a relatively simple model can work extremely well on this dataset.

There are several possible reasons for this outcome. First, the relationship between occupancy and the sensor variables may be close to linearly separable once the data is scaled properly. Second, the strongest features, especially `Light` and `CO2`, may already provide enough discriminative power that a more flexible model does not provide much additional benefit. Third, the Random Forest model may be more prone to small classification tradeoffs that increase recall but reduce precision.

Scaling had only a small effect on Logistic Regression, but the scaled version still outperformed the unscaled version on the validation split. That result supports the standard machine learning practice of scaling features before fitting linear models.

One limitation of this project is that the `date` field was not engineered into time-based features such as hour of day or day of week. Adding those could potentially improve performance further. Another limitation is that the experiments focused on two standard algorithms rather than a wider benchmark set. Even so, the project successfully met its original scope and provided a clear comparison between a simple baseline and a more complex ensemble method.

## Conclusion

This project demonstrated that room occupancy can be predicted accurately using environmental sensor data from a smart-building setting. Logistic Regression and Random Forest were both effective, but Logistic Regression with feature scaling gave the best overall performance. The best holdout result was an F1 score of 0.9824, which shows that the sensor features are highly informative for occupancy detection.

The results also showed that `Light` and `CO2` were the most influential predictors across both models. Overall, the project supports the idea that non-invasive sensors can be used to build accurate occupancy detection systems for practical IoT applications such as energy management and building automation.

## Reproducibility

The project can be rerun with:

```powershell
cd "C:\Saved Files\USD Third Year\CSC 444\occupancy+detection"
python train_models.py
```

All generated metrics, figures, and interpretation tables will be recreated in the `outputs` folder.

## References

1. Candanedo, L. M., & Feldheim, V. (2016). Accurate occupancy detection of an office room from light, temperature, humidity and CO2 measurements using statistical learning models. *Energy and Buildings, 112*, 28-39.
2. UCI Machine Learning Repository. Occupancy Detection Data Set.
3. Zhang, W., Han, K., Costa, G., & Li, J. (2022). A review on occupancy prediction through machine learning for enhancing energy efficiency, air quality and thermal comfort in the built environment. *Renewable and Sustainable Energy Reviews*.
