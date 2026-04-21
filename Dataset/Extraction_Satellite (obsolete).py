
"""
Dataset extraction script for satellite image patches from Google Earth Engine.

Author: AxeWhy
Date: 2026-04-13
Last updated: 2026-04-15

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

# ----------------------------------------------------------------------------------------------------
# Defaults / constants
# ----------------------------------------------------------------------------------------------------
DEFAULT_INPUT_CSV                                       = "Dataset/Input/hdb_points_postal.csv"
DEFAULT_OUTPUT_ROOT                                     = Path("Dataset/Satellite")
DEFAULT_COMBINED_METADATA                               = "satellite_metadata.csv"

DEFAULT_EE_PROJECT                                      = os.getenv("GOOGLE_EE_PROJECT", "quixotic-booth-493203-p6")
DEFAULT_COLLECTION_ID                                   = "COPERNICUS/S2_SR_HARMONIZED"
DEFAULT_START_DATE                                      = "2024-01-01"
DEFAULT_END_DATE                                        = "2024-12-31"
DEFAULT_MAX_CLOUD                                       = 20

DEFAULT_CRS                                             = "EPSG:4326"
DEFAULT_SCALE_M                                         = 10
DEFAULT_PATCH_SIZE_M                                    = 320

DEFAULT_RANDOM_SEED                                     = 42
DEFAULT_TRAIN_RATIO                                     = 0.7
DEFAULT_VALID_RATIO                                     = 0.15

DEFAULT_REQUEST_TIMEOUT_SECONDS                         = 120
DEFAULT_MAX_RETRIES                                     = 4
DEFAULT_RETRY_BACKOFF_SECONDS                           = 1.2
DEFAULT_MIN_DELAY_SECONDS                               = 0.2
DEFAULT_GEOCODE_DELAY_SECONDS                           = 0.2

RETRY_HTTP_STATUSES                                     = {429, 500, 502, 503, 504}
ONEMAP_GEOCODE_URL                                      = "https://www.onemap.gov.sg/api/common/elastic/search"

METADATA_COLUMNS                                        = [
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
# ----------------------------------------------------------------------------------------------------
# parse_args()
# ----------------------------------------------------------------------------------------------------
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
# ----------------------------------------------------------------------------------------------------
# setup_logging()
# ----------------------------------------------------------------------------------------------------
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
# ----------------------------------------------------------------------------------------------------
# validate_args(args)
# ----------------------------------------------------------------------------------------------------
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
# ----------------------------------------------------------------------------------------------------
# _row_value(row, *keys)
# ----------------------------------------------------------------------------------------------------
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
# ----------------------------------------------------------------------------------------------------
# request_with_retries(session, method, url, *, params=None,
#                      timeout=30, max_retries=4, backoff_seconds=1.2)
# ----------------------------------------------------------------------------------------------------
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
# ----------------------------------------------------------------------------------------------------
# geocode_postal_sg(session, postal_code, timeout,
#                   max_retries, backoff_seconds)
# ----------------------------------------------------------------------------------------------------
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

'''
# ----------------------------------------------------------------------------------------------------
# load_samples(csv_path, session, timeout, max_retries, backoff_seconds,
#              geocode_delay_seconds)
# ----------------------------------------------------------------------------------------------------
Description:
Loads samples from the input CSV file, extracting IDs, labels, and either
latitude/longitude or postal codes. The function supports multiple column
name variations for each field. If latitude and longitude are provided, it
parses them as floats. If only a postal code is provided, it uses the 
geocode_postal_sg function to obtain the corresponding latitude and longitude.
The function handles errors gracefully, logging warnings for invalid rows and
skipping them. It returns a list of valid sample dictionaries with keys:
id, lat, lon, label, and postal_code (if applicable).

Arguments:
csv_path: Path to the input CSV file containing sample data.
session: requests.Session object to use for making HTTP requests (e.g., for geocoding).
timeout: Timeout in seconds for HTTP requests.
max_retries: Maximum number of retry attempts for failed HTTP requests.
backoff_seconds: Base seconds for exponential backoff between retries for HTTP requests.
geocode_delay_seconds: Delay in seconds between geocoding requests to avoid rate limits.

Returns:
A list of dictionaries, each representing a valid sample with keys:
- id: Sample identifier (string).
- lat: Latitude of the sample location (float).
- lon: Longitude of the sample location (float).
- label: Label associated with the sample (string).
- postal_code: Postal code associated with the sample (string, may be empty if not applicable).

Raises:
ValueError if no valid samples are loaded from the CSV file.
'''
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

'''
# ----------------------------------------------------------------------------------------------------
# split_rows(rows, seed, train_ratio, valid_ratio)
# ----------------------------------------------------------------------------------------------------
Description:
Splits a list of sample rows into training, validation, and test sets based
on specified ratios. The function uses a random seed for reproducibility and
shuffles the rows before splitting. It calculates the number of samples for
each split based on the provided ratios and returns three seperate list for
train, validation, and test splits.

Arguments:
rows: List of sample dictionaries to be split.
seed: Random seed for shuffling the rows to ensure reproducibility.
train_ratio: Proportion of samples to assign to the training set (e.g., 0.7).
valid_ratio: Proportion of samples to assign to the validation set (e.g., 0.15).

Returns:
A tuple containing three lists: (train_rows, valid_rows, test_rows), where each list
contains the sample dictionaries assigned to the respective split.
'''
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

'''
# ----------------------------------------------------------------------------------------------------
# initialize_earth_engine(project)
# ----------------------------------------------------------------------------------------------------
Description:
Initializes the Google Earth Engine API with the specified project. If a project ID is provided,
it attempts to initialize with that project; otherwise, it uses the default initialization. If
initialization fails (e.g., due to authentication issues), it raises a RuntimeError with a 
descriptive message instructing the user to authenticate and set the appropriate project.
This function is essential for ensuring that the Earth Engine API is ready for subsequent
operations such as building composites and downloading patches.

Arguments:
project: (Optional) Google Earth Engine project ID to use for API access. 
If None, default initialization is attempted.

Returns:
None if initialization is successful; 
Raises RuntimeError if initialization fails.
'''
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

'''
# ----------------------------------------------------------------------------------------------------
# mask_s2_clouds(image)
# ----------------------------------------------------------------------------------------------------
Description:
Applies a cloud and cirrus mask to a Sentinel-2 image using the QA60 band. The function
selects the QA60 band, identifies cloud and cirrus pixels based on specific bits, and 
updates the image mask to exclude those pixels. The resulting masked image is then scaled
by dividing by 10000 to convert from integer reflectance value to float reflectance. This
function is used in the mapping step when building the composite image to ensure that only
cloud-free pixels are included in the final composite.

Arguments:
image: ee.Image object representing a Sentinel-2 image to be masked.

Returns:
An ee.Image object with clouds and cirrus masked out and reflectance values scaled.
'''
def mask_s2_clouds(image):
    qa = image.select("QA60")
    cloud = qa.bitwiseAnd(1 << 10).eq(0)
    cirrus = qa.bitwiseAnd(1 << 11).eq(0)
    return image.updateMask(cloud.And(cirrus)).divide(10000)

'''
# ----------------------------------------------------------------------------------------------------
# build_composite(aoi, collection_id, start_date, end_date, max_cloud)
# ----------------------------------------------------------------------------------------------------
Description:
Builds a composite image from an Earth Engine image collection filtered by the specified
parameters. The function filters the collection by the area of interest (AOI), date range,
and maximum cloud cover percentage. It then applies the mask_s2_clouds function to each image
in the collection to remove cloudy pixels. Finally, it computes the median composite of the
filtered and masked collection and selects the RGB bands (B4, B3, B2) for the final output.
This composite image is used as the source for downloading patches corresponding to the sample
locations.

Arguments:
aoi: ee.Geometry object representing the area of interest for filtering the image collection.
collection_id: String ID of the Earth Engine image collection to query (e.g., "COPERNICUS/S2_SR_HARMONIZED").
start_date: String representing the start date for filtering images (YYYY-MM-DD).
end_date: String representing the end date for filtering images (YYYY-MM-DD).
max_cloud: Integer representing the maximum cloud cover percentage for filtering images.

Returns:
An ee.Image object representing the median composite of the filtered and masked image collection, with RGB bands selected.
'''
def build_composite(aoi, collection_id, start_date, end_date, max_cloud):
    collection = (
        ee.ImageCollection(collection_id)
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
        .map(mask_s2_clouds)
    )
    return collection.median().select(["B4", "B3", "B2"])

'''
# ----------------------------------------------------------------------------------------------------
# build_aoi(rows)
# ----------------------------------------------------------------------------------------------------
Description:
Constructs an area of interest (AOI) geometry from a list of sample rows. The function creates
a point geometry for each sample using its latitude and longitude, then combines all points into
a single feature collection. It computes the convex hull of the feature collection to creat a polygon
that encompasses all sample locations. This AOI is used for filtering the Earth Engine image collection
to ensure that only images covering the sample locations are included in the composite.

Arguments:
rows: List of sample dictionaries, each containing 'lat' and 'lon' keys for the sample's location.

Returns:
An ee.Geometry object representing the convex hull of the sample locations, which serves as the area of
interest for Earth Engine queries.
'''
def build_aoi(rows):
    features = [ee.Feature(ee.Geometry.Point([row["lon"], row["lat"]])) for row in rows]
    return ee.FeatureCollection(features).geometry().convexHull()

'''
# ----------------------------------------------------------------------------------------------------
# square_region(lon, lat, patch_size_m)
# ----------------------------------------------------------------------------------------------------
Description:
Generates a square region geometry centered around a given longitude and latitude with a specified size
in meters. The function creates a point geometry at the specified coordinates, then applies a buffer
to create a circular area with a radius equal to half the patch size. Finally, it computes the bounding
box of the buffered area to produce a square region. This region is used as the area parameter when
requesting downloads from Earth Engine, ensuring that the downloaded patch is centered on the sample location
and has the correct dimensions.

Arguments:
lon: Longitude of the center point for the region.
lat: Latitude of the center point for the region.
patch_size_m: Size of the square patch in meters (the function will create a region that is patch_size_m x patch_size_m).

Returns:
An ee.Geometry object representing the square region centered on the specified coordinates.
'''
def square_region(lon, lat, patch_size_m):
    point = ee.Geometry.Point([lon, lat])
    return point.buffer(patch_size_m / 2).bounds()

'''
# ----------------------------------------------------------------------------------------------------
# get_patch_download_url(image, lon, lat, *, scale_m, patch_size_m, crs)
# ----------------------------------------------------------------------------------------------------
Description:
Generates a download URL for a satellite image patch centered on the specified longitude
and latitude. The function creates a square region around the coordinates using the square_region
function, then calls the getDownloadURL function on the Earth Engine image object with the specified
parameters for region, scale, coordinate reference system (CRS), and format. The resulting URL can
be used to download the image patch corresponding to the sample location. This function is called 
for each sample to obtain the appropriate download URL based on its coordinates and the desired patch 
specifications.

Arguments:
image: ee.Image object representing the composite image from which to download patches.
lon: Longitude of the center point for the patch.
lat: Latitude of the center point for the patch.
scale_m: Scale in meters per pixel for the downloaded patch.
patch_size_m: Size of the square patch in meters.
crs: Coordinate reference system for the downloaded patch (e.g., "EPSG:4326").

Returns:
A string URL that can be used to download the specified image patch from Earth Engine.
'''
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

'''
# ----------------------------------------------------------------------------------------------------
# download_patch(session, url, *, out_path, timeout, max_retries, backoff_seconds)
# ----------------------------------------------------------------------------------------------------
Description:
Downloads an image patch from the specified URL and saves it to the given output path.
The function uses the request_with_retries function to handle the HTTP request with retries
and backoff in case of transient errors. If the download is successful, it writes the content
of the response to the specified output path as a binary file. If the download fails after all
retries, it raises a RuntimeError with a descriptive message.

Arguments:
session: requests.Session object to use for making the HTTP request.
url: The URL from which to download the image patch.
out_path: Path object representing the file path where the downloaded patch should be saved.
timeout: Timeout in seconds for the HTTP request.
max_retries: Maximum number of retry attempts for failed HTTP requests.
backoff_seconds: Base seconds for exponential backoff between retries for HTTP requests.

Returns:
None if the download is successful and the file is saved;
Raises RuntimeError if the download fails after all retry attempts, with a descriptive message.
'''
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

'''
# ----------------------------------------------------------------------------------------------------
# sanitize_filename(value)
# ----------------------------------------------------------------------------------------------------
Description:
Sanitizes a string value to create a safe filename by replacing any characters that are not
alphanumeric, dot, underscore, or hyphen with an underscore. This function is used to ensure 
that sample IDs or other values used in filenames do not contain characters that could cause 
issue in file systems. It uses a regular expression to identify and replace invalid characters.

Arguments:
value: The input value (e.g., sample ID) to be sanitized for use in a filename.

Returns:
A sanitized string that is safe to use as a filename, with invalid characters replaced by underscores.
'''
def sanitize_filename(value):
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(value))

'''
# ----------------------------------------------------------------------------------------------------
# write_metadata_row(writer, row)
# ----------------------------------------------------------------------------------------------------
Description:
Writes a single row of metadata to a CSV file using the provided csv.writer object. The function
ensures that the values are written in the order specified by the METADATA_COLUMNS list. It retrieves
each value from the row dictionary using the get function, providing an empty string as a default if
the key is missing. This function is used to write metadata for each sample to both the combined metadata
CSV and the split-specific metadata CSV, ensuring consistent formatting and handling of missing values.

Arguments:
writer: A csv.writer object that has been initialized with the appropriate file and header.
row: A dictionary containing metadata values for a sample, with keys corresponding to the 
     columns defined in METADATA_COLUMNS.

Returns:
None; the function writes a row to the CSV file using the provided writer.
'''
def write_metadata_row(writer, row):
    writer.writerow([row.get(col, "") for col in METADATA_COLUMNS])

'''
# ----------------------------------------------------------------------------------------------------
# export_split(rows, split_dir, split_name, *, session, image, collection_id,
#              start_date, end_date, max_cloud, scale_m, patch_size_m, crs,
#              timeout, max_retries, backoff_seconds, min_delay_seconds,
#              metadata_only, combined_writer, split_writer)
# ----------------------------------------------------------------------------------------------------
Description:
Processes a list of sample rows for a specific data split (train/validation/test), generating download URLs
for satellite image patches, downloading the patches, and writing metadata for each sample. The function iterates
through each sample row, attempts to generate a download URL using the get_patch_download_url function, and 
handles any exceptions that occur during URL generation by logging a warning and writing an error entry to the
metadata CSVs. If URL generation is successful and metadata_only is not set, it proceeds to download the patch 
using the download_patch function, handling any exceptions during download similarly. For each sample, it writes
a metadata row to both the combined metadata CSV and the split-specific metadata CSV, including the status of the
operation ("OK", "URL_FAILED", "DOWNLOAD_FAILED", etc.) and any error messages. The function also maintains counters
for the number of samples processed, metadata successes/failures, and image download successes/failures, which it
returns at the end for logging purposes.

Arguments:
rows: List of sample dictionaries to process for the split.
split_dir: Path object representing the directory where downloaded patches for this split should be saved.
split_name: String name of the split (e.g., "train", "valid", "test") for metadata recording.
session: requests.Session object to use for making HTTP requests.
image: ee.Image object representing the composite image from which to download patches.
collection_id: String ID of the Earth Engine image collection used for metadata recording.
start_date: String representing the start date for filtering images, used for metadata recording.
end_date: String representing the end date for filtering images, used for metadata recording.
max_cloud: Integer representing the maximum cloud cover percentage for filtering images, used for metadata recording.
scale_m: Scale in meters per pixel for the downloaded patches, used for metadata recording.
patch_size_m: Size of the square patch in meters, used for metadata recording.
crs: Coordinate reference system for the downloaded patches, used for metadata recording.
timeout: Timeout in seconds for HTTP requests.
max_retries: Maximum number of retry attempts for failed HTTP requests.
backoff_seconds: Base seconds for exponential backoff between retries for HTTP requests.
min_delay_seconds: Minimum delay in seconds between processing samples to avoid rate limits.
metadata_only: Boolean flag indicating whether to only generate metadata without downloading patches.
combined_writer: csv.writer object for writing to the combined metadata CSV.
split_writer: csv.writer object for writing to the split-specific metadata CSV.

Returns:
A dictionary containing counters for the number of samples processed, metadata successes/failures, and image
download successes/failures, which can be used for logging and analysis of the export process.
'''
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

'''
# ----------------------------------------------------------------------------------------------------
# ensure_output_dirs(output_root)
# ----------------------------------------------------------------------------------------------------
Description:
Ensures that the necessary output directories for the training, validation, and test splits 
exist under the specified output root directory. The function constructs the paths for each 
split directory, creates them if they do not already exist, and returns the Path object for
each directory. This function is called at the beginning of the run function to set up the
directory structure for saving downloaded patches and metadata files.

Arguments:
output_root: Path object representing the root directory under which the split directories should be created.

Returns:
A tuple containing Path objects for the training, validation, and test directories: (train_dir, valid_dir, test_dir).
'''
def ensure_output_dirs(output_root):
    train_dir = output_root / "Satellite_Training_Set"
    valid_dir = output_root / "Satellite_Validation_Set"
    test_dir = output_root / "Satellite_Test_Set"

    for directory in (train_dir, valid_dir, test_dir):
        directory.mkdir(parents=True, exist_ok=True)

    return train_dir, valid_dir, test_dir

'''
# ----------------------------------------------------------------------------------------------------
# run()
# ----------------------------------------------------------------------------------------------------
Description:
The main execute function that orchestrates the entire process of loading samples, 
splitting them into train/validation/test sets, initializing Earth Engine, building 
the composite image, and exporting the patches and metadata for each split. The function
sets up logging, parses and validates command-line arguments, ensures output directories exist,
and manages the overall flow of data processing. It uses a requests.Session for efficient HTTP
requests and maintains counters for tracking the number of samples processed, metadata successes/failures,
and image download successes/failures, which it logs at the end of the process. The function
also handles the opening and closing of metadata CSV files for writing, ensuring that all metadata
is recorded properly for both the combined metadata file and the split-specific metadata files.

Arguments:
None; the function relies on command-line arguments for configuration.

Returns:
None; the function performs processing and logging but does not return a value.
'''
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

# Entry point for the script
if __name__ == "__main__":
    run()
