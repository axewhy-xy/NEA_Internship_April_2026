"""
Compare the results of two different models by reading their respective CSV files and generating a comparison report.

Author: AxeWhy
Date Created: 2026-08-11
Last Modified: 2026-08-11

Description:
This script reads two CSV files: one containing the predictions from a model and another containing the 
ground truth labels. It compares the predictions against the ground truth and generates a detailed 
comparison report, including metrics such as accuracy, precision, recall, F1 score, and confusion 
matrices for each feature.

Usage:
python compare_results.py --predictions-file <path_to_predictions_csv> --labels-file <path_to_labels_csv> --detail-output <path_to_detail_output_csv> --summary-output <path_to_summary_output_csv>

"""
import pandas as pd
import numpy as np
import argparse
from sklearn.metrics import(
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

PREDICTION_FILE = "predictions_simplified.csv"
LABELS_FILE = "Labelled_BuildingFeature.csv"

DETAIL_OUTPUT = "comparison_details.csv"
SUMMARY_OUTPUT = "comparison_summary.csv"

def binary_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

def run():
    pred = pd.read_csv(PREDICTION_FILE)
    labels = pd.read_csv(LABELS_FILE)

    pred["postal_code"] = pred["postal_code"].astype(str).str.strip()
    labels["Postal"] = labels["Postal"].astype(str).str.strip()

    merged = pred.merge(
        labels,
        left_on="postal_code",
        right_on="Postal",
        how="inner",
        validate="one_to_one",
    )

    print(f"Prediction rows: {len(pred)}")
    print(f"Ground truth rows: {len(labels)}")
    print(f"Matched rows: {len(merged)}")
    print(f"Unmatched prediction rows: {len(pred) - len(merged)}")
    print(f"Unmatched ground truth rows: {len(labels) - len(merged)}")
    print()

    summary = []

    y_true = merged["Water Tanks"].str.strip().eq("Yes")
    y_pred = merged["water_tank"].gt(0)

    metrics = binary_metrics(y_true, y_pred)
    summary.append({
        "feature": "Water Tanks",
        "comparison": "count > 0 vs Yes/No",
        **metrics,
        "coverage": 1.0,
    })
    
    y_true = merged["Solar Panels"].str.strip().eq("Yes")
    y_pred = merged["solar_panels"].gt(0)

    metrics = binary_metrics(y_true, y_pred)
    summary.append({
        "feature": "Solar Panels",
        "comparison": "count > 0 vs Yes/No",
        **metrics,
        "coverage": 1.0,
    })

    ac_pred = np.select(
        [
            merged["concrete_AC"] > merged["frame_AC"],
            merged["frame_AC"] > merged["concrete_AC"],
        ],
        [
            "Ledge",
            "Wall-mount",
        ],
        default="Unknown",
    )

    ac_true = merged["Dominant Type of Air-conditioning Mount"].str.strip()

    known = ac_pred != "Unknown"

    if known.any():
        ac_accuracy = accuracy_score(ac_true[known], ac_pred[known])
    else:
        ac_accuracy = np.nan

    summary.append({
        "feature": "Air-conditioning mount",
        "comparison": "higher of concrete_AC/frame_AC vs dominant type",
        "accuracy": ac_accuracy,
        "precision": np.nan,
        "recall": np.nan,
        "f1": np.nan,
        "coverage": known.mean(),
    })

    y_true = ~merged["Outdoor Laundry Rack Type"].str.strip().eq(
        "Not Observable"
    )
    y_pred = merged["drying_fixture"].gt(0)

    metrics = binary_metrics(y_true, y_pred)
    summary.append({
        "feature": "Drying fixture",
        "comparison": "count > 0 vs observable/not observable",
        **metrics,
        "coverage": 1.0,
    })

    merged["pred_water_tank_yes"] = merged["water_tank"] > 0
    merged["true_water_tank_yes"] = (
        merged["Water Tanks"].str.strip() == "Yes"
    )
    merged["water_tank_correct"] = (
        merged["pred_water_tank_yes"] == merged["true_water_tank_yes"]
    )

    merged["pred_solar_yes"] = merged["solar_panels"] > 0
    merged["true_solar_yes"] = (
        merged["Solar Panels"].str.strip() == "Yes"
    )
    merged["solar_correct"] = (
        merged["pred_solar_yes"] == merged["true_solar_yes"]
    )

    merged["pred_ac_type"] = ac_pred
    merged["ac_correct"] = (
        merged["pred_ac_type"] == merged["Dominant Type of Air-conditioning Mount"]
    )
    merged.loc[merged["pred_ac_type"] == "Unknown", "ac_correct"] = np.nan

    merged["pred_drying_observable"] = merged["drying_fixture"] > 0
    merged["true_drying_observable"] = (
        merged["Outdoor Laundry Rack Type"].str.strip() != "Not Observable"
    )
    merged["drying_correct"] = (
        merged["pred_drying_observable"]
        == merged["true_drying_observable"]
    )

    pd.DataFrame(summary).to_csv(SUMMARY_OUTPUT, index=False)
    merged.to_csv(DETAIL_OUTPUT, index=False)

    print("=== COMPARISON SUMMARY ===")
    for row in summary:
        print(f"\n{row['feature']}")
        print(f"  Method:    {row['comparison']}")
        print(f"  Accuracy:  {row['accuracy']:.4f}")
        if not pd.isna(row["precision"]):
            print(f"  Precision: {row['precision']:.4f}")
            print(f"  Recall:    {row['recall']:.4f}")
            print(f"  F1:        {row['f1']:.4f}")
        print(f"  Coverage:  {row['coverage']:.4f}")
    
    print("\n=== CONFUSION MATRICES ===")
    for name, true_col, pred_col in [
        ("Water Tanks", "true_water_tank_yes", "pred_water_tank_yes"),
        ("Solar Panels", "true_solar_yes", "pred_solar_yes"),
        ("Drying Fixture", "true_drying_observable", "pred_drying_observable"),
    ]:
        cm = confusion_matrix(
            merged[true_col],
            merged[pred_col],
            labels=[False, True],
        )
        print(f"\n{name}")
        print("             Pred No   Pred Yes")
        print(f"True No      {cm[0,0]:8d} {cm[0,1]:10d}")
        print(f"True Yes     {cm[1,0]:8d} {cm[1,1]:10d}")

    print(f"\nDetailed results written to: {DETAIL_OUTPUT}")
    print(f"Summary written to: {SUMMARY_OUTPUT}")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare the results of two different models by reading their respective CSV files and generating a comparison report."
    )
    parser.add_argument(
        "--predictions-file",
        type=str,
        default=PREDICTION_FILE,
        help="Path to the predictions CSV file (default: predictions_simplified.csv)."
    )
    parser.add_argument(
        "--labels-file",
        type=str,
        default=LABELS_FILE,
        help="Path to the ground truth labels CSV file (default: Labelled_BuildingFeature.csv)."
    )
    parser.add_argument(
        "--detail-output",
        type=str,
        default=DETAIL_OUTPUT,
        help="Path to the output CSV file for detailed comparison results (default: comparison_details.csv)."
    )
    parser.add_argument(
        "--summary-output",
        type=str,
        default=SUMMARY_OUTPUT,
        help="Path to the output CSV file for summary comparison results (default: comparison_summary.csv)."
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    PREDICTION_FILE = args.predictions_file
    LABELS_FILE = args.labels_file
    DETAIL_OUTPUT = args.detail_output
    SUMMARY_OUTPUT = args.summary_output

    run()