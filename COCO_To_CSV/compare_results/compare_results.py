"""
Compare the results of two different models by reading their respective CSV files and generating a comparison report.

Author: AxeWhy
Date Created: 2026-08-11
Last Modified: 2026-08-19

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

PREDICTION_FILE = "./COCO_To_CSV/compare_results/results/predictions_simplified_SUTD.csv"
LABELS_FILE = "./COCO_To_CSV/compare_results/results/Labelled_BuildingFeature_SUTD.csv"

DETAIL_OUTPUT = "./COCO_To_CSV/compare_results/results/comparison_details.csv"
SUMMARY_OUTPUT = "./COCO_To_CSV/compare_results/results/comparison_summary.csv"

def accuracy_score(y_true, y_pred):
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)

    if len(y_true) == 0:
        return np.nan

    return (y_true == y_pred).mean()


def precision_score(y_true, y_pred, zero_division=0):
    y_true = pd.Series(y_true).astype(bool).reset_index(drop=True)
    y_pred = pd.Series(y_pred).astype(bool).reset_index(drop=True)

    true_positive = (y_true & y_pred).sum()
    predicted_positive = y_pred.sum()

    if predicted_positive == 0:
        return zero_division

    return true_positive / predicted_positive


def recall_score(y_true, y_pred, zero_division=0):
    y_true = pd.Series(y_true).astype(bool).reset_index(drop=True)
    y_pred = pd.Series(y_pred).astype(bool).reset_index(drop=True)

    true_positive = (y_true & y_pred).sum()
    actual_positive = y_true.sum()

    if actual_positive == 0:
        return zero_division

    return true_positive / actual_positive


def f1_score(y_true, y_pred, zero_division=0):
    precision = precision_score(y_true, y_pred, zero_division=zero_division)
    recall = recall_score(y_true, y_pred, zero_division=zero_division)

    if precision + recall == 0:
        return zero_division

    return 2 * precision * recall / (precision + recall)


def confusion_matrix(y_true, y_pred, labels):
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)
    label_positions = {label: index for index, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)

    for true_value, pred_value in zip(y_true, y_pred):
        if true_value in label_positions and pred_value in label_positions:
            matrix[label_positions[true_value], label_positions[pred_value]] += 1

    return matrix


def binary_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

def report_column_differences(merged):
    comparisons = [
        {
            "name": "Water Tanks",
            "prediction_column": "water_tank",
            "label_column": "Water Tanks",
            "pred_value": merged["water_tank"].gt(0).map({True: "Yes", False: "No"}),
            "true_value": merged["Water Tanks"].str.strip(),
        },
        {
            "name": "Solar Panels",
            "prediction_column": "solar_panels",
            "label_column": "Solar Panels",
            "pred_value": merged["solar_panels"].gt(0).map({True: "Yes", False: "No"}),
            "true_value": merged["Solar Panels"].str.strip(),
        },
        {
            "name": "Air-conditioning mount",
            "prediction_column": "concrete_AC/frame_AC",
            "label_column": "Dominant Type of Air-conditioning Mount",
            "pred_value": np.select(
                [
                    merged["concrete_AC"] > merged["frame_AC"],
                    merged["frame_AC"] > merged["concrete_AC"],
                ],
                ["Ledge", "Wall-mount"],
                default="Unknown",
            ),
            "true_value": merged["Dominant Type of Air-conditioning Mount"].str.strip(),
        },
        {
            "name": "Drying fixture",
            "prediction_column": "drying_fixture",
            "label_column": "Outdoor Laundry Rack Type",
            "pred_value": merged["drying_fixture"].gt(0).map({True: "Observable", False: "Not Observable"}),
            "true_value": np.where(
                merged["Outdoor Laundry Rack Type"].str.strip().eq("Not Observable"),
                "Not Observable",
                "Observable",
            ),
        },
    ]

    print("=== COLUMN DIFFERENCES ===")
    differences = []

    for item in comparisons:
        pred_value = pd.Series(item["pred_value"], index=merged.index)
        true_value = pd.Series(item["true_value"], index=merged.index)

        comparable_mask = pred_value.notna() & true_value.notna()
        if item["name"] == "Air-conditioning mount":
            comparable_mask = comparable_mask & (pred_value != "Unknown")

        mismatch_mask = comparable_mask & (pred_value != true_value)
        mismatch_count = int(mismatch_mask.sum())
        comparable_count = int(comparable_mask.sum())

        if mismatch_count > 0:
            print(
                f"{item['name']}: {mismatch_count} differing rows out of {comparable_count} comparable rows "
                f"({item['prediction_column']} vs {item['label_column']})"
            )
            differences.append(
                {
                    "feature": item["name"],
                    "prediction_column": item["prediction_column"],
                    "label_column": item["label_column"],
                    "differences": mismatch_count,
                    "comparable_rows": comparable_count,
                }
            )

    if not differences:
        print("No column differences found among the compared feature columns.")

    print()
    return differences


def normalize_postal_code(series):
    normalized = series.fillna("").astype(str).str.strip()
    has_value = normalized.ne("")
    normalized.loc[has_value] = normalized.loc[has_value].str.zfill(6)
    return normalized


def require_columns(frame, columns, file_path):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(
            f"{file_path} is missing required column(s): {sorted(missing)}"
        )


def prepare_solar_labels(labels):
    require_columns(labels, {"POSTALCODE", "has_solar"}, LABELS_FILE)

    solar_labels = labels[["POSTALCODE", "has_solar"]].copy()
    solar_labels["POSTALCODE"] = normalize_postal_code(solar_labels["POSTALCODE"])
    solar_labels = solar_labels[solar_labels["POSTALCODE"].ne("")]
    solar_labels["has_solar"] = (
        pd.to_numeric(solar_labels["has_solar"], errors="coerce")
        .fillna(0)
        .gt(0)
        .astype(int)
    )

    return solar_labels.groupby("POSTALCODE", as_index=False)["has_solar"].max()


def report_solar_comparison(merged):
    y_true = pd.to_numeric(merged["has_solar"], errors="coerce").fillna(0).gt(0)
    y_pred = pd.to_numeric(merged["solar_panels"], errors="coerce").fillna(0).gt(0)

    merged["pred_solar_yes"] = y_pred
    merged["true_solar_yes"] = y_true
    merged["solar_correct"] = y_pred == y_true

    metrics = binary_metrics(y_true, y_pred)
    summary = [{
        "feature": "Solar Panels",
        "comparison": "solar_panels > 0 vs has_solar > 0",
        **metrics,
        "coverage": 1.0 if len(merged) else 0.0,
    }]

    mismatch_count = int((y_pred != y_true).sum())
    print("=== COLUMN DIFFERENCES ===")
    if mismatch_count:
        print(
            "Solar Panels: "
            f"{mismatch_count} differing rows out of {len(merged)} comparable rows "
            "(solar_panels vs has_solar)"
        )
    else:
        print("No column differences found among the compared feature columns.")
    print()

    pd.DataFrame(summary).to_csv(SUMMARY_OUTPUT, index=False)
    merged.to_csv(DETAIL_OUTPUT, index=False)

    print("=== COMPARISON SUMMARY ===")
    for row in summary:
        print(f"\n{row['feature']}")
        print(f"  Method:    {row['comparison']}")
        print(f"  Accuracy:  {row['accuracy']:.4f}")
        print(f"  Precision: {row['precision']:.4f}")
        print(f"  Recall:    {row['recall']:.4f}")
        print(f"  F1:        {row['f1']:.4f}")
        print(f"  Coverage:  {row['coverage']:.4f}")

    cm = confusion_matrix(y_true, y_pred, labels=[False, True])
    print("\n=== CONFUSION MATRICES ===")
    print("\nSolar Panels")
    print("             Pred No   Pred Yes")
    print(f"True No      {cm[0,0]:8d} {cm[0,1]:10d}")
    print(f"True Yes     {cm[1,0]:8d} {cm[1,1]:10d}")

    print(f"\nDetailed results written to: {DETAIL_OUTPUT}")
    print(f"Summary written to: {SUMMARY_OUTPUT}")


def run(compare_solar_panels=False):
    pred = pd.read_csv(PREDICTION_FILE)
    labels = pd.read_csv(LABELS_FILE)
    compare_solar_only = (
        compare_solar_panels
        or {"POSTALCODE", "has_solar"}.issubset(labels.columns)
    )

    require_columns(pred, {"postal_code"}, PREDICTION_FILE)
    pred["postal_code"] = normalize_postal_code(pred["postal_code"])
    pred = pred[pred["postal_code"].ne("")].copy()

    if compare_solar_only:
        require_columns(pred, {"solar_panels"}, PREDICTION_FILE)
        raw_label_rows = len(labels)
        labels = prepare_solar_labels(labels)
        merged = pred.merge(
            labels,
            left_on="postal_code",
            right_on="POSTALCODE",
            how="inner",
            validate="one_to_one",
        )

        print(f"Prediction rows: {len(pred)}")
        print(f"Ground truth rows: {raw_label_rows}")
        print(f"Ground truth postal codes: {len(labels)}")
        print(f"Matched rows: {len(merged)}")
        print(f"Unmatched prediction rows: {len(pred) - len(merged)}")
        print(f"Unmatched ground truth postal codes: {len(labels) - len(merged)}")
        print()

        report_solar_comparison(merged)
        return
    else:
        labels["Postal"] = normalize_postal_code(labels["Postal"])

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
    merged["ac_correct"] = np.where(
        merged["pred_ac_type"] == "Unknown",
        np.nan,
        merged["pred_ac_type"] == merged["Dominant Type of Air-conditioning Mount"],
    )

    merged["pred_drying_observable"] = merged["drying_fixture"] > 0
    merged["true_drying_observable"] = (
        merged["Outdoor Laundry Rack Type"].str.strip() != "Not Observable"
    )
    merged["drying_correct"] = (
        merged["pred_drying_observable"]
        == merged["true_drying_observable"]
    )

    report_column_differences(merged)

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
    parser.add_argument(
        "--compare-solar-panels",
        action="store_true",
        help="Include comparison for only solar panels in the report (default: False)."
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    PREDICTION_FILE = args.predictions_file
    LABELS_FILE = args.labels_file
    DETAIL_OUTPUT = args.detail_output
    SUMMARY_OUTPUT = args.summary_output

    run(args.compare_solar_panels)
