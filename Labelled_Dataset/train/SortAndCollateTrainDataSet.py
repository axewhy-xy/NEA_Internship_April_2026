"""
Sorting and collating script for the training dataset.

Author: AxeWhy
Date: 25/05/2026
Last Updated: 25/05/2026

Description:
This script is responsible for sorting and collating the training dataset for the project. Due to the size limitation of
label-studio, the training dataset is split into multiple parts. This means that the output COCO JSON file from label-studio
will be split into multiple JSON files, each containing a portion of the dataset and there will be a conflict between the image IDs
and annotation IDs across the different JSON files. This script will sort and collate the multiple JSON files into a single JSON file,
ensuring that the image IDs and annotation IDs are unique across the entire dataset. The script will
also ensure that the images are correctly associated with their corresponding annotations in the final collated JSON file.

Command Line Usage:
python SortAndCollateTrainDataSet.py -i <input_directory> -o <output_json_file>
"""

import argparse
import json
import os
import sys