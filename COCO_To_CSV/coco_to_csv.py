"""
Convert COCO JSON annotations to CSV format.

Author: AxeWhy
Date Created: 2026-07-27
Last Modified: 2026-07-27

Description:
This source file contains the implementation of a function that converts COCO JSON annotations to
csv format. The function reads a COCO JSON file, extracts the file path, xmin, ymin, xmax, ymax, class
label and postal code for each annotation, and saves the extracted data to a CSV file.

Usage:
python coco_to_csv.py --input_json <path_to_coco_json> --output_csv <path_to_output_csv>

"""

import pandas as pd
import json
import argparse
import os
import re

def parse_args():
    parser = argparse.ArgumentParser(description="Convert COCO JSON annotations to CSV format.")
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Path to the input COCO JSON file."
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to the output CSV file."
    )
    return parser.parse_args()

def coco_to_csv(input_file, output_file):
    with open(input_file, 'r') as f:
        coco_data = json.load(f)

    csv_data = []
    category_mapping = {category['id']: category['name'] for category in coco_data['categories']}
    image_mapping = {img['id']: img for img in coco_data['images']} 

    for annotation in coco_data['annotations']:
        image_id = annotation['image_id']
        bbox = annotation['bbox']
        category_id = annotation['category_id']

        image_info = image_mapping.get(image_id)
        if image_info is None:
            continue

        file_name = image_info['file_name']
        xmin, ymin, width, height = bbox
        xmax = xmin + width
        ymax = ymin + height
        class_label = category_mapping.get(category_id, "Unknown")

        match = re.search(r"_(\d{6})[_\.]", file_name)
        postal_code = match.group(1) if match else "Unknown"

        csv_data.append([file_name, xmin, ymin, xmax, ymax, class_label, postal_code])

    df = pd.DataFrame(csv_data, columns=['file_path', 'xmin', 'ymin', 'xmax', 'ymax', 'class_label', 'postal_code'])
    df.to_csv(output_file, index=False)

def main():
    args = parse_args()
    input_file = args.input_file
    output_file = args.output_file

    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' does not exist.")
        return

    coco_to_csv(input_file, output_file)
    print(f"Successfully converted COCO JSON annotations to CSV format and saved to '{output_file}'.")

if __name__ == "__main__":
    main()
