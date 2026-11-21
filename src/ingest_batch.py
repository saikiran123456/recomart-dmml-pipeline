import os
import time
import json
import logging
from datetime import datetime
import pandas as pd
import requests
import boto3
from botocore.exceptions import ClientError

# --- Configuration ---
TODAY = datetime.now().strftime("%Y-%m-%d")
LANDING_ZONE = "landing_zone"
LOCAL_RAW_ROOT = "data/raw"
LOG_FILE = "logs/ingestion.log"

MINIO_ENDPOINT = "http://localhost:9000"
MINIO_ACCESS_KEY = "admin"
MINIO_SECRET_KEY = "password123"
BUCKET_NAME = "recomart-raw"

# --- Logging Setup ---
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)

def get_s3_client():
    s3 = boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
    )
    # Auto-create bucket if missing
    try:
        s3.head_bucket(Bucket=BUCKET_NAME)
    except ClientError:
        s3.create_bucket(Bucket=BUCKET_NAME)
        logging.info(f"Created MinIO bucket: {BUCKET_NAME}")
    return s3

def upload_to_minio(s3, local_path, s3_key):
    s3.upload_file(local_path, BUCKET_NAME, s3_key)
    logging.info(f"Uploaded to MinIO -> s3://{BUCKET_NAME}/{s3_key}")

def fetch_api_with_retry(url, max_retries=3, backoff=2):
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logging.warning(f"Attempt {attempt}/{max_retries} failed for {url}: {e}")
            if attempt == max_retries:
                logging.error(f"Permanent failure fetching {url}")
                raise
            time.sleep(backoff ** attempt)

def ingest_csvs(s3):
    csv_mapping = {
        "events.csv": ("csv/clickstream", 50000),  # Sample 50k rows for fast local processing
        "olist_orders_dataset.csv": ("csv/transactions", None),
        "olist_order_items_dataset.csv": ("csv/transactions", None),
        "olist_order_reviews_dataset.csv": ("csv/transactions", None),
        "olist_customers_dataset.csv": ("csv/transactions", None),
        "olist_products_dataset.csv": ("csv/catalog", None),
        "product_category_name_translation.csv": ("csv/catalog", None),
    }

    for filename, (subpath, nrows) in csv_mapping.items():
        src_path = os.path.join(LANDING_ZONE, filename)
        if not os.path.exists(src_path):
            logging.error(f"Missing file in landing_zone: {src_path}")
            continue

        df = pd.read_csv(src_path, nrows=nrows)
        dest_dir = os.path.join(LOCAL_RAW_ROOT, subpath, f"dt={TODAY}")
        os.makedirs(dest_dir, exist_ok=True)
        local_dest = os.path.join(dest_dir, filename)
        df.to_csv(local_dest, index=False)

        s3_key = f"{subpath}/dt={TODAY}/{filename}"
        upload_to_minio(s3, local_dest, s3_key)
        logging.info(f"Ingested CSV {filename} ({len(df):,} rows)")

def ingest_apis(s3):
    apis = {
        "products.json": ("api/catalog", "https://dummyjson.com/products?limit=100"),
        "fakestore_popularity.json": ("api/external_scores", "https://fakestoreapi.com/products"),
        "dummyjson_comments.json": ("api/external_scores", "https://dummyjson.com/comments?limit=100"),
    }

    for filename, (subpath, url) in apis.items():
        data = fetch_api_with_retry(url)
        dest_dir = os.path.join(LOCAL_RAW_ROOT, subpath, f"dt={TODAY}")
        os.makedirs(dest_dir, exist_ok=True)
        local_dest = os.path.join(dest_dir, filename)

        with open(local_dest, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        s3_key = f"{subpath}/dt={TODAY}/{filename}"
        upload_to_minio(s3, local_dest, s3_key)
        logging.info(f"Ingested API {url} -> {filename}")

if __name__ == "__main__":
    logging.info("=== Starting Batch Ingestion ===")
    s3_client = get_s3_client()
    ingest_csvs(s3_client)
    ingest_apis(s3_client)
    logging.info("=== Batch Ingestion Completed Successfully ===")