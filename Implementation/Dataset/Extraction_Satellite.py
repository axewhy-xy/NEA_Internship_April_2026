
"""
Dataset extraction script for satellite image patches from Google Earth Engine.

Author: AxeWhy
Date: 2026-04-13

Description:
Builds a satellite patch dataset from a CSV of locations and labels.
Supports both coordinate and postal-code input formats, performs
train/validation/test splitting, downloads patches for each sample,
and writes detailed metadata logs.

Input CSV formats supported:
1) id,lat,lon,label
2) id,postal_code,label

Usage example:
python Dataset/Extraction_Satellite.py --input-csv Dataset/hdb_points_postal.csv

Earth Engine project:
Set GOOGLE_EE_PROJECT in your environment, or pass --ee-project.
"""

import argparse
import csv
import datetime
import logging
import os
import random
import re
import time
from pathlib import Path

import ee
import requests

# -----------------------------
# Defaults / constants
# -----------------------------
DEFAULT_INPUT_CSV = "Dataset/hdb_points_postal.csv"
DEFAULT_OUTPUT_ROOT = Path("Dataset/Satellite_Metadata")
DEFAULT_COMBINED_METADATA = "satellite_metadata.csv"

DEFAULT_EE_PROJECT = os.getenv("GOOGLE_EE_PROJECT", "quixotic-booth-493203-p6")
DEFAULT_COLLECTION_ID = "COPERNICUS/S2_SR_HARMONIZED"
DEFAULT_START_DATE = "2024-01-01"
DEFAULT_END_DATE = "2024-12-31"
DEFAULT_MAX_CLOUD = 20

DEFAULT_CRS = "EPSG:4326"
DEFAULT_SCALE_M = 10
DEFAULT_PATCH_SIZE_M = 320

DEFAULT_RANDOM_SEED = 42
DEFAULT_TRAIN_RATIO = 0.7
DEFAULT_VALID_RATIO = 0.15

DEFAULT_REQUEST_TIMEOUT_SECONDS = 120
DEFAULT_MAX_RETRIES = 4
DEFAULT_RETRY_BACKOFF_SECONDS = 1.2
DEFAULT_MIN_DELAY_SECONDS = 0.2
DEFAULT_GEOCODE_DELAY_SECONDS = 0.2

RETRY_HTTP_STATUSES = {429, 500, 502, 503, 504}
ONEMAP_GEOCODE_URL = "https://www.onemap.gov.sg/api/common/elastic/search"

METADATA_COLUMNS = [
    "id",
    "lat",
    "lon",
    "postal_code",
    "label",
    "split",
    "path",
    "status",
    "error",
    "collection_id",
    "start_date",
    "end_date",
    "max_cloud",
    "scale_m",
    "patch_size_m",
    "crs",
]

'''
------------------------------
parse_args()
------------------------------
Description:
Defines and parses command-line arguments for the dataset extraction script.
Includes validation for argument values and formats.

Arguments:
--input-csv: Path to the input CSV file containing sample locations and labels.
--output-root: Directory where the output dataset and metadata will be saved.
--metadata-csv: Filename for the combined metadata CSV in the output root.
--ee-project: Google Earth Engine project ID to use for API access.
--collection-id: Earth Engine image collection ID to query (default is Sentinel-2 SR).
--start-date: Start date for filtering Earth Engine images (YYYY-MM-DD).
--end-date: End date for filtering Earth Engine images (YYYY-MM-DD).
--max-cloud: Maximum cloud cover percentage for Earth Engine image filtering.
--crs: Coordinate reference system for downloaded patches (default EPSG:4326).
--scale-m: Scale in meters per pixel for downloaded patches (default 10m).
--patch-size-m: Size of the square patch in meters (default 320m).
--seed: Random seed for reproducible train/validation/test splitting.
--train-ratio: Proportion of samples to assign to the training set (default 0.7).
--valid-ratio: Proportion of samples to assign to the validation set (default 0.15).
--timeout-seconds: Timeout for HTTP requests in seconds (default 120s).
--max-retries: Maximum number of retries for failed HTTP requests (default 4).
--retry-backoff-seconds: Base seconds for exponential backoff between retries (default 1.2s).
--min-delay-seconds: Minimum delay between processing samples to avoid rate limits (default 0.2s).
--geocode-delay-seconds: Delay between geocoding requests to avoid rate limits (default 0.2s).
--metadata-only: If set, only generate metadata and skip downloading image patches.

Returns:
Namespace object containing parsed argument values.
'''
def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract Earth Engine satellite patch dataset with split folders and metadata."
    )
    parser.add_argument("--input-csv", default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--metadata-csv", default=DEFAULT_COMBINED_METADATA)

    parser.add_argument("--ee-project", default=DEFAULT_EE_PROJECT)
    parser.add_argument("--collection-id", default=DEFAULT_COLLECTION_ID)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--max-cloud", type=int, default=DEFAULT_MAX_CLOUD)

    parser.add_argument("--crs", default=DEFAULT_CRS)
    parser.add_argument("--scale-m", type=int, default=DEFAULT_SCALE_M)
    parser.add_argument("--patch-size-m", type=int, default=DEFAULT_PATCH_SIZE_M)

    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    parser.add_argument("--valid-ratio", type=float, default=DEFAULT_VALID_RATIO)

    parser.add_argument(
        "--timeout-seconds", type=int, default=DEFAULT_REQUEST_TIMEOUT_SECONDS
    )
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument(
        "--retry-backoff-seconds", type=float, default=DEFAULT_RETRY_BACKOFF_SECONDS
    )
    parser.add_argument("--min-delay-seconds", type=float, default=DEFAULT_MIN_DELAY_SECONDS)
    parser.add_argument(
        "--geocode-delay-seconds", type=float, default=DEFAULT_GEOCODE_DELAY_SECONDS
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Only verify URL generation and write metadata; skip patch downloads.",
    )
    return parser.parse_args()

'''
------------------------------
setup_logging()
------------------------------
Description:
Configures the logging settings for the script, including log level, format,
and timestamp format. logs will be printed to the console with INFO level and
above, showing timestamps, log levels, and messages.

Aruments:
None

Returns:
None
'''
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

'''
------------------------------
validate_args(args)
------------------------------
Description:
Validates the command-line arguments to ensure they meet expected formats and constraints.
The function checks for valid split ratios, cloud cover percentage, scale and patch sizes,
timeout and retry settings, and date formats. It raises ValueError with descriptive messages
if any argument is invalid.

Arguments:
args: Namespace object containing parsed command-line arguments.

Returns:
ValueError is raised if any argument is invalid; 
otherwise, the function returns None.
'''
def validate_args(args):
    if args.train_ratio <= 0 or args.valid_ratio < 0:
        raise ValueError("Split ratios must be non-negative and train > 0.")
    if args.train_ratio + args.valid_ratio >= 1.0:
        raise ValueError("train_ratio + valid_ratio must be < 1.0.")
    if args.max_cloud < 0 or args.max_cloud > 100:
        raise ValueError("--max-cloud must be between 0 and 100.")
    if args.scale_m <= 0:
        raise ValueError("--scale-m must be > 0.")
    if args.patch_size_m <= 0:
        raise ValueError("--patch-size-m must be > 0.")
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be > 0.")
    if args.max_retries < 0:
        raise ValueError("--max-retries must be >= 0.")
    if args.retry_backoff_seconds <= 0:
        raise ValueError("--retry-backoff-seconds must be > 0.")
    if args.min_delay_seconds < 0:
        raise ValueError("--min-delay-seconds must be >= 0.")
    if args.geocode_delay_seconds < 0:
        raise ValueError("--geocode-delay-seconds must be >= 0.")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", args.start_date):
        raise ValueError("--start-date must be YYYY-MM-DD.")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", args.end_date):
        raise ValueError("--end-date must be YYYY-MM-DD.")

    start = datetime.date.fromisoformat(args.start_date)
    end = datetime.date.fromisoformat(args.end_date)
    if start > end:
        raise ValueError("--start-date must be <= --end-date.")

'''
------------------------------
_row_value(row, *keys)
------------------------------
Description:
Helper function to extract a value from a dictionary row using multiple
possible keys. It iterates through the provided keys and returns the first
non-empty value found. If no valid value is found, it returns None.

Arguments:
row: Dictionary representing a row of data (e.g., from a CSV).
*keys: Variable number of string keys to check in the row for a valid value.

Returns:
The first non-empty value found for the provided keys.
None if no valid value is found.
'''
def _row_value(row, *keys):
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None

'''
------------------------------
request_with_retries(session, method, url, *, params=None,
                     timeout=30, max_retries=4, backoff_seconds=1.2)
------------------------------
Description:
Performs a HTTP request with retries and exponential backoff for handling transient errors.
The function uses a requests.Session to make the request and retries on specific HTTP status
codes (e.g., 429, 500, 502, 503, 504) or on request exceptions. It implements exponential
backoff between retries and logs warnings for each retry attempt. If all retries fail,
it raises a RuntimeError with the last encountered error.

Arguments:
session: requests.Session object to use for making the HTTP request.
method: HTTP method as a string (e.g., "GET", "POST").
url: The URL to which the request is sent.
params: (Optional) Dictionary of query parameters to include in the request.
timeout: Timeout in seconds
max_retries: Maximum number of retry attempts for failed requests.
backoff_seconds: Base seconds for exponential backoff between retries.

Returns:
The successful requests.Response object if the request is successful within the retry limit.
Raises RuntimeError if all retry attempts fail, with the last encountered error as context.
'''
def request_with_retries(
    session,
    method,
    url,
    *,
    params=None,
    timeout=30,
    max_retries=4,
    backoff_seconds=1.2,
):
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            response = session.request(method, url, params=params, timeout=timeout)

            if response.status_code in RETRY_HTTP_STATUSES and attempt < max_retries:
                wait_seconds = backoff_seconds ** (attempt + 1)
                logging.warning(
                    "HTTP %s from %s. Retrying in %.2fs (%s/%s).",
                    response.status_code,
                    url,
                    wait_seconds,
                    attempt + 1,
                    max_retries,
                )
                time.sleep(wait_seconds)
                continue

            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            wait_seconds = backoff_seconds ** (attempt + 1)
            logging.warning(
                "Request error on %s. Retrying in %.2fs (%s/%s): %s",
                url,
                wait_seconds,
                attempt + 1,
                max_retries,
                exc,
            )
            time.sleep(wait_seconds)

    raise RuntimeError(f"Request failed after retries: {url}") from last_error

'''
-------------------------------
geocode_postal_sg(session, postal_code, timeout,
                  max_retries, backoff_seconds)
-------------------------------
Description:
Geocodes a Singapore postal code to latitude and longitude using the OneMap API.
The function sends a GET request to the OneMap geocoding endpoint with the postal
code as a query parameter. It handles retries and backoff using the request_with_retries function.
If the geocoding is successful and returns results, it extracts the latitude and longitude
from the first result. If the postal code is invalid or geocoding fails, it returns (None, None).

Arguments:
session: requests.Session object to use for making the HTTP request.
postal_code: The Singapore postal code to geocode.
timeout: Timeout in seconds for the HTTP request.
max_retries: Maximum number of retry attempts for failed requests.
backoff_seconds: Base seconds for exponential backoff between retries.

Returns:
A tuple (lat, lon) with the latitude and longitude of the geocoded postal code.
Returns (None, None) if the postal code is invalid or geocoding fails.
'''
def geocode_postal_sg(
    session,
    postal_code,
    timeout,
    max_retries,
    backoff_seconds,
):
    postal = str(postal_code).strip()
    if not postal:
        return None, None

    params = {
        "searchVal": postal,
        "returnGeom": "Y",
        "getAddrDetails": "Y",
        "pageNum": 1,
    }
    response = request_with_retries(
        session,
        "GET",
        ONEMAP_GEOCODE_URL,
        params=params,
        timeout=timeout,
        max_retries=max_retries,
        backoff_seconds=backoff_seconds,
    )
    payload = response.json()

    if int(payload.get("found", 0)) == 0:
        return None, None

    results = payload.get("results") or []
    if not results:
        return None, None

    first = results[0]
    try:
        lat = float(first["LATITUDE"])
        lon = float(first["LONGITUDE"])
    except (KeyError, TypeError, ValueError):
        return None, None
    return lat, lon


def load_samples(
    csv_path,
    session,
    timeout,
    max_retries,
    backoff_seconds,
    geocode_delay_seconds,
):
    rows = []
    postal_cache = {}

    with open(csv_path, newline="", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)

        for line_no, row in enumerate(reader, start=2):
            sample_id = _row_value(row, "id", "ID") or f"row_{line_no}"
            label = _row_value(row, "label", "Label") or ""

            lat_raw = _row_value(row, "lat", "latitude", "LAT", "Latitude")
            lon_raw = _row_value(row, "lon", "lng", "longitude", "LON", "Longitude")
            postal_code = _row_value(
                row,
                "postal_code",
                "postal",
                "postcode",
                "postal code",
                "POSTAL",
            )

            lat = None
            lon = None

            if lat_raw is not None and lon_raw is not None:
                try:
                    lat = float(lat_raw)
                    lon = float(lon_raw)
                except ValueError:
                    logging.warning(
                        "Skipping line %s (%s): invalid lat/lon values.",
                        line_no,
                        sample_id,
                    )
                    continue
            elif postal_code:
                if postal_code not in postal_cache:
                    lat_lon = geocode_postal_sg(
                        session,
                        postal_code,
                        timeout=timeout,
                        max_retries=max_retries,
                        backoff_seconds=backoff_seconds,
                    )
                    postal_cache[postal_code] = lat_lon
                    time.sleep(geocode_delay_seconds)

                lat, lon = postal_cache[postal_code]
                if lat is None or lon is None:
                    logging.warning(
                        "Skipping line %s (%s): could not geocode postal code %s.",
                        line_no,
                        sample_id,
                        postal_code,
                    )
                    continue
            else:
                logging.warning(
                    "Skipping line %s (%s): missing both lat/lon and postal code.",
                    line_no,
                    sample_id,
                )
                continue

            rows.append(
                {
                    "id": sample_id,
                    "lat": lat,
                    "lon": lon,
                    "label": label,
                    "postal_code": postal_code or "",
                }
            )

    if not rows:
        raise ValueError(f"No valid samples loaded from {csv_path}.")
    return rows


def split_rows(rows, seed, train_ratio, valid_ratio):
    rng = random.Random(seed)
    rows_copy = rows[:]
    rng.shuffle(rows_copy)

    count = len(rows_copy)
    n_train = int(count * train_ratio)
    n_valid = int(count * valid_ratio)

    train = rows_copy[:n_train]
    valid = rows_copy[n_train : n_train + n_valid]
    test = rows_copy[n_train + n_valid :]
    return train, valid, test


def initialize_earth_engine(project):
    try:
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
    except Exception as exc:
        raise RuntimeError(
            "Earth Engine initialization failed. Authenticate and set --ee-project or GOOGLE_EE_PROJECT."
        ) from exc


def mask_s2_clouds(image):
    qa = image.select("QA60")
    cloud = qa.bitwiseAnd(1 << 10).eq(0)
    cirrus = qa.bitwiseAnd(1 << 11).eq(0)
    return image.updateMask(cloud.And(cirrus)).divide(10000)


def build_composite(aoi, collection_id, start_date, end_date, max_cloud):
    collection = (
        ee.ImageCollection(collection_id)
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
        .map(mask_s2_clouds)
    )
    return collection.median().select(["B4", "B3", "B2"])


def build_aoi(rows):
    features = [ee.Feature(ee.Geometry.Point([row["lon"], row["lat"]])) for row in rows]
    return ee.FeatureCollection(features).geometry().convexHull()


def square_region(lon, lat, patch_size_m):
    point = ee.Geometry.Point([lon, lat])
    return point.buffer(patch_size_m / 2).bounds()


def get_patch_download_url(image, lon, lat, *, scale_m, patch_size_m, crs):
    region = square_region(lon, lat, patch_size_m)
    return image.getDownloadURL(
        {
            "region": region,
            "scale": scale_m,
            "crs": crs,
            "format": "GEO_TIFF",
        }
    )


def download_patch(
    session,
    url,
    *,
    out_path,
    timeout,
    max_retries,
    backoff_seconds,
):
    response = request_with_retries(
        session,
        "GET",
        url,
        timeout=timeout,
        max_retries=max_retries,
        backoff_seconds=backoff_seconds,
    )
    out_path.write_bytes(response.content)


def sanitize_filename(value):
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(value))


def write_metadata_row(writer, row):
    writer.writerow([row.get(col, "") for col in METADATA_COLUMNS])


def export_split(
    rows,
    split_dir,
    split_name,
    *,
    session,
    image,
    collection_id,
    start_date,
    end_date,
    max_cloud,
    scale_m,
    patch_size_m,
    crs,
    timeout,
    max_retries,
    backoff_seconds,
    min_delay_seconds,
    metadata_only,
    combined_writer,
    split_writer,
):
    counters = {
        "samples": len(rows),
        "metadata_ok": 0,
        "metadata_not_ok": 0,
        "image_ok": 0,
        "image_failed": 0,
    }

    for row in rows:
        sample_id = row["id"]
        lat = row["lat"]
        lon = row["lon"]
        label = row["label"]
        postal_code = row.get("postal_code", "")

        try:
            download_url = get_patch_download_url(
                image,
                lon,
                lat,
                scale_m=scale_m,
                patch_size_m=patch_size_m,
                crs=crs,
            )
        except Exception as exc:
            counters["metadata_not_ok"] += 1
            error_row = {
                "id": sample_id,
                "lat": lat,
                "lon": lon,
                "postal_code": postal_code,
                "label": label,
                "split": split_name,
                "path": "",
                "status": "URL_FAILED",
                "error": str(exc),
                "collection_id": collection_id,
                "start_date": start_date,
                "end_date": end_date,
                "max_cloud": max_cloud,
                "scale_m": scale_m,
                "patch_size_m": patch_size_m,
                "crs": crs,
            }
            write_metadata_row(combined_writer, error_row)
            write_metadata_row(split_writer, error_row)
            logging.warning("%s | URL generation failed: %s", sample_id, exc)
            time.sleep(min_delay_seconds)
            continue

        counters["metadata_ok"] += 1

        if metadata_only:
            meta_only_row = {
                "id": sample_id,
                "lat": lat,
                "lon": lon,
                "postal_code": postal_code,
                "label": label,
                "split": split_name,
                "path": "",
                "status": "OK_METADATA_ONLY",
                "error": "",
                "collection_id": collection_id,
                "start_date": start_date,
                "end_date": end_date,
                "max_cloud": max_cloud,
                "scale_m": scale_m,
                "patch_size_m": patch_size_m,
                "crs": crs,
            }
            write_metadata_row(combined_writer, meta_only_row)
            write_metadata_row(split_writer, meta_only_row)
            time.sleep(min_delay_seconds)
            continue

        safe_id = sanitize_filename(sample_id)
        file_name = f"{safe_id}_{split_name}.tif"
        out_path = split_dir / file_name

        try:
            download_patch(
                session,
                download_url,
                out_path=out_path,
                timeout=timeout,
                max_retries=max_retries,
                backoff_seconds=backoff_seconds,
            )
            counters["image_ok"] += 1
            row_data = {
                "id": sample_id,
                "lat": lat,
                "lon": lon,
                "postal_code": postal_code,
                "label": label,
                "split": split_name,
                "path": str(out_path),
                "status": "OK",
                "error": "",
                "collection_id": collection_id,
                "start_date": start_date,
                "end_date": end_date,
                "max_cloud": max_cloud,
                "scale_m": scale_m,
                "patch_size_m": patch_size_m,
                "crs": crs,
            }
        except Exception as exc:
            counters["image_failed"] += 1
            row_data = {
                "id": sample_id,
                "lat": lat,
                "lon": lon,
                "postal_code": postal_code,
                "label": label,
                "split": split_name,
                "path": str(out_path),
                "status": "DOWNLOAD_FAILED",
                "error": str(exc),
                "collection_id": collection_id,
                "start_date": start_date,
                "end_date": end_date,
                "max_cloud": max_cloud,
                "scale_m": scale_m,
                "patch_size_m": patch_size_m,
                "crs": crs,
            }
            logging.warning("%s | download failed: %s", sample_id, exc)

        write_metadata_row(combined_writer, row_data)
        write_metadata_row(split_writer, row_data)
        time.sleep(min_delay_seconds)

    return counters


def ensure_output_dirs(output_root):
    train_dir = output_root / "Training_Set"
    valid_dir = output_root / "Validation_Set"
    test_dir = output_root / "Test_Set"

    for directory in (train_dir, valid_dir, test_dir):
        directory.mkdir(parents=True, exist_ok=True)

    return train_dir, valid_dir, test_dir


def run():
    setup_logging()
    args = parse_args()
    validate_args(args)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    train_dir, valid_dir, test_dir = ensure_output_dirs(output_root)

    combined_metadata_path = output_root / args.metadata_csv
    split_metadata_paths = {
        "train": output_root / "satellite_metadata_train.csv",
        "valid": output_root / "satellite_metadata_valid.csv",
        "test": output_root / "satellite_metadata_test.csv",
    }

    logging.info("Loading samples from %s", args.input_csv)
    with requests.Session() as session:
        samples = load_samples(
            args.input_csv,
            session,
            timeout=args.timeout_seconds,
            max_retries=args.max_retries,
            backoff_seconds=args.retry_backoff_seconds,
            geocode_delay_seconds=args.geocode_delay_seconds,
        )

        train_rows, valid_rows, test_rows = split_rows(
            samples,
            seed=args.seed,
            train_ratio=args.train_ratio,
            valid_ratio=args.valid_ratio,
        )

        logging.info(
            "Loaded %s samples | train=%s valid=%s test=%s",
            len(samples),
            len(train_rows),
            len(valid_rows),
            len(test_rows),
        )

        logging.info("Initializing Earth Engine")
        initialize_earth_engine(args.ee_project)

        logging.info("Building AOI and composite")
        aoi = build_aoi(samples)
        image = build_composite(
            aoi,
            collection_id=args.collection_id,
            start_date=args.start_date,
            end_date=args.end_date,
            max_cloud=args.max_cloud,
        )

        with open(combined_metadata_path, "w", newline="", encoding="utf-8") as combined_file, \
            open(split_metadata_paths["train"], "w", newline="", encoding="utf-8") as train_meta_file, \
            open(split_metadata_paths["valid"], "w", newline="", encoding="utf-8") as valid_meta_file, \
            open(split_metadata_paths["test"], "w", newline="", encoding="utf-8") as test_meta_file:

            combined_writer = csv.writer(combined_file)
            train_writer = csv.writer(train_meta_file)
            valid_writer = csv.writer(valid_meta_file)
            test_writer = csv.writer(test_meta_file)

            combined_writer.writerow(METADATA_COLUMNS)
            train_writer.writerow(METADATA_COLUMNS)
            valid_writer.writerow(METADATA_COLUMNS)
            test_writer.writerow(METADATA_COLUMNS)

            total = {
                "samples": 0,
                "metadata_ok": 0,
                "metadata_not_ok": 0,
                "image_ok": 0,
                "image_failed": 0,
            }

            split_tasks = [
                (train_rows, train_dir, "train", train_writer),
                (valid_rows, valid_dir, "valid", valid_writer),
                (test_rows, test_dir, "test", test_writer),
            ]

            for split_rows_data, split_dir, split_name, split_writer in split_tasks:
                logging.info("Exporting split: %s", split_name)
                stats = export_split(
                    split_rows_data,
                    split_dir,
                    split_name,
                    session=session,
                    image=image,
                    collection_id=args.collection_id,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    max_cloud=args.max_cloud,
                    scale_m=args.scale_m,
                    patch_size_m=args.patch_size_m,
                    crs=args.crs,
                    timeout=args.timeout_seconds,
                    max_retries=args.max_retries,
                    backoff_seconds=args.retry_backoff_seconds,
                    min_delay_seconds=args.min_delay_seconds,
                    metadata_only=args.metadata_only,
                    combined_writer=combined_writer,
                    split_writer=split_writer,
                )

                for key in total:
                    total[key] += stats[key]

                logging.info(
                    "Split %s complete | samples=%s metadata_ok=%s metadata_not_ok=%s image_ok=%s image_failed=%s",
                    split_name,
                    stats["samples"],
                    stats["metadata_ok"],
                    stats["metadata_not_ok"],
                    stats["image_ok"],
                    stats["image_failed"],
                )

    logging.info("All splits complete.")
    logging.info(
        "TOTAL | samples=%s metadata_ok=%s metadata_not_ok=%s image_ok=%s image_failed=%s",
        total["samples"],
        total["metadata_ok"],
        total["metadata_not_ok"],
        total["image_ok"],
        total["image_failed"],
    )
    logging.info("Combined metadata: %s", combined_metadata_path)
    logging.info("Per-split metadata: %s", split_metadata_paths)


if __name__ == "__main__":
    run()
