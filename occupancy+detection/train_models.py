from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"

TRAIN_FILE = BASE_DIR / "datatraining.txt"
VALIDATION_FILE = BASE_DIR / "datatest.txt"
TEST_FILE = BASE_DIR / "datatest2.txt"

FEATURE_COLUMNS = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]
TARGET_COLUMN = "Occupancy"


def load_split(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=["date"])


def build_models() -> dict[str, object]:
    return {
        "logistic_regression_unscaled": LogisticRegression(max_iter=1000, random_state=42),
        "logistic_regression_scaled": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
        ),
    }


def evaluate_split(model, x_train: pd.DataFrame, y_train: pd.Series, x_eval: pd.DataFrame, y_eval: pd.Series) -> tuple[dict[str, float], pd.Series]:
    fitted_model = clone(model)
    fitted_model.fit(x_train, y_train)
    predictions = pd.Series(fitted_model.predict(x_eval), index=y_eval.index)
    metrics = {
        "accuracy": accuracy_score(y_eval, predictions),
        "precision": precision_score(y_eval, predictions, zero_division=0),
        "recall": recall_score(y_eval, predictions, zero_division=0),
        "f1": f1_score(y_eval, predictions, zero_division=0),
    }
    return metrics, predictions


def save_confusion_matrix(y_true: pd.Series, y_pred: pd.Series, title: str, filename: str) -> None:
    matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / filename, dpi=200)
    plt.close()


def save_class_balance_plot(y_train: pd.Series, y_validation: pd.Series, y_test: pd.Series) -> None:
    counts = pd.DataFrame(
        {
            "train": y_train.value_counts().sort_index(),
            "validation": y_validation.value_counts().sort_index(),
            "test": y_test.value_counts().sort_index(),
        }
    ).fillna(0)
    counts.index = counts.index.map({0: "Not Occupied", 1: "Occupied"})
    plot_df = counts.reset_index(names="Occupancy").melt(id_vars="Occupancy", var_name="Split", value_name="Count")

    plt.figure(figsize=(8, 5))
    sns.barplot(data=plot_df, x="Split", y="Count", hue="Occupancy", palette="Set2")
    plt.title("Class Balance Across Dataset Splits")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "class_balance.png", dpi=200)
    plt.close()


def save_correlation_heatmap(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 6))
    sns.heatmap(df[FEATURE_COLUMNS + [TARGET_COLUMN]].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "correlation_heatmap.png", dpi=200)
    plt.close()


def save_feature_importance(model, feature_names: list[str]) -> pd.DataFrame:
    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance_df.to_csv(OUTPUT_DIR / "random_forest_feature_importance.csv", index=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=importance_df, x="importance", y="feature", hue="feature", palette="viridis", legend=False)
    plt.title("Random Forest Feature Importance")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "random_forest_feature_importance.png", dpi=200)
    plt.close()
    return importance_df


def save_logistic_coefficients(model, feature_names: list[str]) -> pd.DataFrame:
    if isinstance(model, Pipeline):
        coefficients = model.named_steps["model"].coef_[0]
    else:
        coefficients = model.coef_[0]

    coefficient_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
            "absolute_coefficient": [abs(value) for value in coefficients],
        }
    ).sort_values("absolute_coefficient", ascending=False)
    coefficient_df.to_csv(OUTPUT_DIR / "logistic_regression_coefficients.csv", index=False)
    return coefficient_df


def write_summary(validation_results: pd.DataFrame, test_results: pd.DataFrame, feature_importance: pd.DataFrame) -> None:
    best_validation_model = validation_results.sort_values("f1", ascending=False).iloc[0]
    best_test_model = test_results.sort_values("f1", ascending=False).iloc[0]
    top_features = ", ".join(feature_importance["feature"].head(3).tolist())
    scaled_lr_f1 = validation_results.loc["logistic_regression_scaled", "f1"]
    unscaled_lr_f1 = validation_results.loc["logistic_regression_unscaled", "f1"]

    lines = [
        "# Occupancy Detection Summary",
        "",
        "## Validation Split",
        f"- Best validation model: {best_validation_model.name}",
        f"- Validation F1: {best_validation_model['f1']:.4f}",
        f"- Scaled Logistic Regression F1: {scaled_lr_f1:.4f}",
        f"- Unscaled Logistic Regression F1: {unscaled_lr_f1:.4f}",
        "",
        "## Holdout Test Split",
        f"- Best holdout model: {best_test_model.name}",
        f"- Holdout F1: {best_test_model['f1']:.4f}",
        "",
        "## Interpretation",
        f"- Most important Random Forest features: {top_features}",
        "- Use the validation table to compare scaling effects and the holdout table for final reported results.",
    ]
    (OUTPUT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    sns.set_theme(style="whitegrid")
    OUTPUT_DIR.mkdir(exist_ok=True)
    FIGURE_DIR.mkdir(exist_ok=True)

    train_df = load_split(TRAIN_FILE)
    validation_df = load_split(VALIDATION_FILE)
    test_df = load_split(TEST_FILE)

    x_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]
    x_validation = validation_df[FEATURE_COLUMNS]
    y_validation = validation_df[TARGET_COLUMN]
    x_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    save_class_balance_plot(y_train, y_validation, y_test)
    save_correlation_heatmap(train_df)

    validation_rows = []
    validation_predictions = {}

    models = build_models()
    for name, model in models.items():
        metrics, predictions = evaluate_split(model, x_train, y_train, x_validation, y_validation)
        validation_rows.append({"model": name, **metrics})
        validation_predictions[name] = predictions

    validation_results = pd.DataFrame(validation_rows).set_index("model").sort_values("f1", ascending=False)
    validation_results.to_csv(OUTPUT_DIR / "validation_results.csv")

    save_confusion_matrix(
        y_validation,
        validation_predictions["logistic_regression_scaled"],
        "Validation Confusion Matrix: Logistic Regression",
        "validation_confusion_logistic_regression.png",
    )
    save_confusion_matrix(
        y_validation,
        validation_predictions["random_forest"],
        "Validation Confusion Matrix: Random Forest",
        "validation_confusion_random_forest.png",
    )

    combined_train = pd.concat([train_df, validation_df], ignore_index=True)
    x_combined = combined_train[FEATURE_COLUMNS]
    y_combined = combined_train[TARGET_COLUMN]

    test_rows = []
    test_predictions = {}
    final_models = {}
    for name in ["logistic_regression_scaled", "random_forest"]:
        final_model = clone(models[name])
        final_model.fit(x_combined, y_combined)
        final_models[name] = final_model
        predictions = pd.Series(final_model.predict(x_test), index=y_test.index)
        test_predictions[name] = predictions
        test_rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, predictions),
                "precision": precision_score(y_test, predictions, zero_division=0),
                "recall": recall_score(y_test, predictions, zero_division=0),
                "f1": f1_score(y_test, predictions, zero_division=0),
            }
        )

    test_results = pd.DataFrame(test_rows).set_index("model").sort_values("f1", ascending=False)
    test_results.to_csv(OUTPUT_DIR / "test_results.csv")

    save_confusion_matrix(
        y_test,
        test_predictions["logistic_regression_scaled"],
        "Holdout Confusion Matrix: Logistic Regression",
        "test_confusion_logistic_regression.png",
    )
    save_confusion_matrix(
        y_test,
        test_predictions["random_forest"],
        "Holdout Confusion Matrix: Random Forest",
        "test_confusion_random_forest.png",
    )

    rf_importance = save_feature_importance(final_models["random_forest"], FEATURE_COLUMNS)
    save_logistic_coefficients(final_models["logistic_regression_scaled"], FEATURE_COLUMNS)
    write_summary(validation_results, test_results, rf_importance)

    print("Outputs written to:", OUTPUT_DIR)
    print()
    print("Validation results")
    print(validation_results.round(4))
    print()
    print("Holdout test results")
    print(test_results.round(4))


if __name__ == "__main__":
    main()
