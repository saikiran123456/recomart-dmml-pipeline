import os
import sys
import subprocess
import logging
from datetime import datetime
from prefect import flow, task, get_run_logger

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler("logs/orchestration.log"), logging.StreamHandler()]
)

def run_script(script_path):
    """Execute a modular pipeline script using the active .venv Python interpreter."""
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True,
        text=True,
        check=True
    )
    return result.stdout

@task(name="1a_Batch_Ingestion", retries=2, retry_delay_seconds=5)
def task_ingest_batch():
    logger = get_run_logger()
    logger.info("Running Batch Ingestion (CSVs + REST APIs -> MinIO)...")
    run_script("src/ingest_batch.py")
    return "batch_ingested"

@task(name="1b_Near_Real_Time_Stream", retries=2, retry_delay_seconds=5)
def task_ingest_stream():
    logger = get_run_logger()
    logger.info("Running Near-Real-Time Micro-Batch Stream (Clickstreams -> MinIO + Redis)...")
    run_script("src/ingest_stream.py")
    return "stream_ingested"

@task(name="2_Data_Validation", retries=1, retry_delay_seconds=5)
def task_validate(batch_status, stream_status):
    logger = get_run_logger()
    logger.info(f"Validating raw partitions ({batch_status}, {stream_status})...")
    run_script("src/validate.py")
    return "validated"

@task(name="3_Data_Preparation_EDA", retries=1, retry_delay_seconds=5)
def task_prepare(val_status):
    logger = get_run_logger()
    logger.info(f"Preparing & cleaning datasets ({val_status})...")
    run_script("src/prepare.py")
    return "prepared"

@task(name="4_Feature_Engineering_Warehouse", retries=1, retry_delay_seconds=5)
def task_transform(prep_status):
    logger = get_run_logger()
    logger.info(f"Engineering features & loading PostgreSQL warehouse ({prep_status})...")
    run_script("src/transform.py")
    return "transformed"

@task(name="5_Feature_Store_Feast_Redis", retries=2, retry_delay_seconds=5)
def task_feature_store(trans_status):
    logger = get_run_logger()
    logger.info(f"Applying Feast registry & materializing to Redis ({trans_status})...")
    run_script("src/feature_store_demo.py")
    return "feast_ready"

@task(name="6_Data_Versioning_DVC_Lineage", retries=1, retry_delay_seconds=5)
def task_version_lineage(feast_status):
    logger = get_run_logger()
    logger.info(f"Updating DVC dataset snapshots & lineage metadata ({feast_status})...")
    subprocess.run([sys.executable, "-m", "dvc", "add", "data/raw", "data/processed", "data/features"], check=True)
    subprocess.run([sys.executable, "-m", "dvc", "push"], check=True)
    run_script("src/lineage_tracker.py")
    return "versioned"

@task(name="7_Model_Training_MLflow", retries=1, retry_delay_seconds=5)
def task_train(ver_status):
    logger = get_run_logger()
    logger.info(f"Training SVD & Content-Based models and logging to MLflow ({ver_status})...")
    run_script("src/train.py")
    return "trained"

@task(name="8_Live_Inference_Verification", retries=1, retry_delay_seconds=5)
def task_infer(train_status):
    logger = get_run_logger()
    logger.info(f"Verifying end-to-end live recommendation inference ({train_status})...")
    run_script("src/infer.py")
    return "inference_verified"

@flow(name="RecoMart_End_To_End_Data_Pipeline", log_prints=True)
def recomart_pipeline():
    start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f"=== Starting Prefect Orchestrated Pipeline at {start} ===")

    b_status = task_ingest_batch()
    s_status = task_ingest_stream()
    v_status = task_validate(b_status, s_status)
    p_status = task_prepare(v_status)
    t_status = task_transform(p_status)
    f_status = task_feature_store(t_status)
    d_status = task_version_lineage(f_status)
    m_status = task_train(d_status)
    i_status = task_infer(m_status)

    logging.info(f"=== Prefect Pipeline Finished Successfully: {i_status} ===")
    return i_status

if __name__ == "__main__":
    recomart_pipeline()