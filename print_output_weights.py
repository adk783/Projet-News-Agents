"""
Print final model weights for the main NLP outputs.

This reads the trained market impact classifier and shows how the numeric
signals (polarity, uncertainty, litigious, fundamental_strength, etc.) are
weighted for Bearish / Bullish / Neutral predictions.
"""

import json
import os

import pandas as pd
from joblib import load


MODEL_DIR = "./market_impact_model"
CLASSIFIER_FILE = os.path.join(MODEL_DIR, "classifier.joblib")
METRICS_FILE = os.path.join(MODEL_DIR, "metrics.json")
OUTPUT_DIR = "./metrics_plots"
NUMERIC_WEIGHTS_CSV = os.path.join(OUTPUT_DIR, "final_model_numeric_feature_weights.csv")
RAW_OUTPUT_WEIGHTS_CSV = os.path.join(OUTPUT_DIR, "base_output_weight_summary.csv")

RAW_OUTPUTS = [
    "polarity",
    "polarity_conf",
    "uncertainty",
    "litigious",
    "fundamental_strength",
]


def extract_numeric_weights():
    classifier = load(CLASSIFIER_FILE)
    with open(METRICS_FILE, "r", encoding="utf-8") as handle:
        metrics = json.load(handle)

    numeric_columns = metrics["numeric_classifier_columns"]
    feature_names = classifier.named_steps["features"].get_feature_names_out()
    logistic = classifier.named_steps["classifier"]

    rows = []
    for class_index, class_name in enumerate(logistic.classes_):
        coefficients = logistic.coef_[class_index]
        for feature in numeric_columns:
            feature_name = f"numeric__{feature}"
            feature_index = list(feature_names).index(feature_name)
            rows.append(
                {
                    "class": class_name,
                    "feature": feature,
                    "coefficient": float(coefficients[feature_index]),
                    "abs_coefficient": abs(float(coefficients[feature_index])),
                }
            )

    weights_df = pd.DataFrame(rows)
    raw_summary_df = (
        weights_df[weights_df["feature"].isin(RAW_OUTPUTS)]
        .groupby("feature", as_index=False)
        .agg(mean_abs_weight=("abs_coefficient", "mean"))
        .sort_values("mean_abs_weight", ascending=False)
    )
    return weights_df, raw_summary_df


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    weights_df, raw_summary_df = extract_numeric_weights()

    weights_df.to_csv(NUMERIC_WEIGHTS_CSV, index=False, encoding="utf-8")
    raw_summary_df.to_csv(RAW_OUTPUT_WEIGHTS_CSV, index=False, encoding="utf-8")

    print("\nPoids moyens des outputs de base dans le modele final")
    print("Plus le poids absolu moyen est haut, plus le signal pese dans la decision.")
    print(raw_summary_df.to_string(index=False))

    print("\nPoids detailles par classe")
    print(weights_df[weights_df["feature"].isin(RAW_OUTPUTS)].to_string(index=False))

    print(f"\nCSV ecrit : {NUMERIC_WEIGHTS_CSV}")
    print(f"CSV ecrit : {RAW_OUTPUT_WEIGHTS_CSV}")


if __name__ == "__main__":
    main()
