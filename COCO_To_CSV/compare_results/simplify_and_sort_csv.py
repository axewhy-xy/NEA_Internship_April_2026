"""
Simplify and sort a CSV file by removing unnecessary columns and sorting the rows based on a specific column.

Author: AxeWhy
Date Created: 2026-08-11
Last Modified: 2026-08-19

Description:
Converts an object detection CSV file into a simplified CSV file that keeps one row
per postal code and replaces the class label with a counter column per class.

Input columns:
file_path, xmin, ymin, xmax, ymax, class_label, confidence_score, postal_code

Output columns:
postal_code, water_tank, solar_panels, concrete_AC, frame_AC, drying_fixture

Usage:
python simplify_and_sort_csv.py --input <path_to_input_csv> --output <path_to_output_csv>

"""

import argparse
import csv

CLASS_TO_COLUMN = {
    "Water Tank": "water_tank",
    "Solar Panel": "solar_panels",
    "Concrete AC Mount": "concrete_AC",
    "Frame AC Mount": "frame_AC",
    "Drying Fixtures": "drying_fixture",
}

OUTPUT_COLUMNS = ["postal_code"] + list(dict.fromkeys(CLASS_TO_COLUMN.values()))

def simplify(input_path: str, output_path: str) -> None:
    with open(input_path, newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
 
        missing = {"postal_code", "class_label"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Input CSV is missing required column(s): {sorted(missing)}"
            )
 
        summary_rows = {}
        unmapped_labels = set()
 
        for row in reader:
            postal_code = (row["postal_code"] or "").strip()
            class_label = (row["class_label"] or "").strip()
            column = CLASS_TO_COLUMN.get(class_label)

            if not postal_code:
                continue

            postal_code = postal_code.zfill(6)

            if postal_code not in summary_rows:
                summary_rows[postal_code] = {col: 0 for col in OUTPUT_COLUMNS}
                summary_rows[postal_code]["postal_code"] = postal_code

            if column:
                summary_rows[postal_code][column] += 1
            else:
                unmapped_labels.add(class_label)

        with open(output_path, "w", newline="", encoding="utf-8") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()

            for postal_code in sorted(summary_rows):
                out_row = summary_rows[postal_code]
                writer.writerow(out_row)

    print(f"Wrote {len(summary_rows)} rows to {output_path}")
    if unmapped_labels:
        print(
            "Warning: encountered class_label value(s) with no matching "
            f"output column (left blank): {sorted(unmapped_labels)}"
        )
 
def parse_args():
    parser = argparse.ArgumentParser(
        description="Simplify and sort a CSV file by removing unnecessary columns and sorting the rows based on a specific column."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the output CSV file.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    simplify(args.input, args.output)
