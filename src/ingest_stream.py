import os
import time
import json
import logging
from datetime import datetime
import pandas as pd
import boto3
import redis

# --- Configuration ---
TODAY = datetime.now().strftime("%Y-%m-%d")
EVENTS_FILE = "landing_zone/events.csv"
LOCAL_STREAM_DIR = f"data/raw/stream/clickstream/dt={TODAY}"
LOG_FILE = "logs/stream_ingestion.log"

MINIO_ENDPOINT = "http://localhost:9000"
BUCKET_NAME = "recomart-raw"

# --- Logging Setup ---
os.makedirs(LOCAL_STREAM_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)

def get_clients():
    s3 = boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id="admin",
        aws_secret_access_key="password123",
    )
    r = redis.Redis(host="localhost", port=6379, decode_responses=True)
    return s3, r

def run_stream(num_batches=5, batch_size=20, interval_sec=3):
    s3, r = get_clients()
    logging.info(f"=== Starting Near-Real-Time Stream ({num_batches} micro-batches) ===")

    df = pd.read_csv(EVENTS_FILE, nrows=num_batches * batch_size)

    for i in range(num_batches):
        batch_df = df.iloc[i * batch_size : (i + 1) * batch_size]
        ts = datetime.now().strftime("%H%M%S")
        filename = f"batch_{ts}.json"
        local_path = os.path.join(LOCAL_STREAM_DIR, filename)

        records = batch_df.to_dict(orient="records")
        with open(local_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)

        # 1. Push micro-batch to MinIO Raw Data Lake
        s3_key = f"stream/clickstream/dt={TODAY}/{filename}"
        s3.upload_file(local_path, BUCKET_NAME, s3_key)

        # 2. Update live user click counts in Redis (online feature store)
        for rec in records:
            user_key = f"user:{rec['visitorid']}:live_events"
            r.incr(user_key)

        logging.info(f"Micro-batch {i+1}/{num_batches} -> Uploaded {s3_key} & updated {len(records)} Redis keys")
        if i < num_batches - 1:
            time.sleep(interval_sec)

    logging.info("=== Near-Real-Time Stream Completed ===")

if __name__ == "__main__":
    run_stream()