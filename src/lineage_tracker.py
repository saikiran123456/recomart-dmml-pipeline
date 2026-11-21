import os
import json
import yaml
import logging
from datetime import datetime

os.makedirs("logs", exist_ok=True)
os.makedirs("reports", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler("logs/lineage.log"), logging.StreamHandler()]
)

def parse_dvc_file(dvc_path):
    if not os.path.exists(dvc_path):
        return {"md5": "NOT_TRACKED", "size": 0, "nfiles": 0}
    with open(dvc_path, "r", encoding="utf-8") as f:
        meta = yaml.safe_load(f)
    out = meta.get("outs", [{}])[0]
    return {
        "md5": out.get("md5", "N/A"),
        "size_mb": round(out.get("size", 0) / (1024 * 1024), 2),
        "nfiles": out.get("nfiles", 0),
    }

def generate_lineage_docs():
    logging.info("=== Starting Task 8: Data Versioning & Lineage Tracking ===")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    stages = [
        {
            "stage": "1. Raw Data Lake",
            "path": "data/raw",
            "dvc_file": "data/raw.dvc",
            "sources": "Kaggle Retailrocket CSV, Olist CSVs (6 files), DummyJSON & FakeStore REST APIs",
            "transformations": "Automated batch & near-real-time micro-batch ingestion; partitioned by source/type/dt",
        },
        {
            "stage": "2. Cleaned & Prepared Datasets",
            "path": "data/processed",
            "dvc_file": "data/processed.dvc",
            "sources": "data/raw (validated via Great Expectations)",
            "transformations": "Deduplication (reviews & clickstreams), missing category imputation, label encoding, Min-Max scaling",
        },
        {
            "stage": "3. Engineered Feature Tables",
            "path": "data/features",
            "dvc_file": "data/features.dvc",
            "sources": "data/processed + External API sentiment/popularity JSONs",
            "transformations": "User/item aggregations, co-occurrence cosine similarity, PostgreSQL load, Feast Parquet export",
        },
    ]

    lineage_records = []
    for s in stages:
        dvc_info = parse_dvc_file(s["dvc_file"])
        lineage_records.append({**s, **dvc_info, "recorded_at": now})

    with open("data/lineage_metadata.json", "w", encoding="utf-8") as f:
        json.dump(lineage_records, f, indent=2)

    md_path = "reports/08_data_versioning_lineage.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# RecoMart Data Versioning & Lineage Documentation\n\n")
        f.write(f"- **Versioning Tool:** DVC (Data Version Control) + Git\n")
        f.write(f"- **Remote Store:** `local_storage` (`./dvc_storage`)\n")
        f.write(f"- **Last Recorded:** {now}\n\n")
        f.write("## Dataset Lineage & Version Hashes\n\n")
        f.write("| Stage | Dataset Path | Files | Size (MB) | DVC MD5 Hash | Data Source | Applied Transformations |\n")
        f.write("| :--- | :--- | :---: | :---: | :--- | :--- | :--- |\n")
        for r in lineage_records:
            f.write(
                f"| {r['stage']} | `{r['path']}` | {r['nfiles']} | {r['size_mb']} | "
                f"`{r['md5']}` | {r['sources']} | {r['transformations']} |\n"
            )
        f.write("\n## Reproducibility Workflow\n")
        f.write("1. `git checkout <commit-hash>` to select code and `.dvc` pointer versions.\n")
        f.write("2. `dvc checkout` (or `dvc pull`) to restore exact dataset snapshots.\n")
        f.write("3. Run `dvc status` to detect any untracked data drift.\n")

    logging.info(f"Saved lineage metadata -> data/lineage_metadata.json & {md_path}")
    logging.info("=== Task 8 Completed Successfully ===")

if __name__ == "__main__":
    generate_lineage_docs()