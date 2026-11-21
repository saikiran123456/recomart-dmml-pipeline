import os
import textwrap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

os.makedirs("reports", exist_ok=True)
FENCE = "`" * 3


def create_problem_formulation_pdf():
    pdf_path = "reports/01_problem_formulation.pdf"
    sections = [
        (
            "1. Business Problem Statement",
            "RecoMart, an e-commerce startup, collects user behavior and catalog data across disconnected channels "
            "but lacks an automated data pipeline to continuously power personalized product recommendations. "
            "The goal is to design and implement an end-to-end, modular, automated data management pipeline "
            "supporting batch and near-real-time ingestion to increase cross-sell conversion and customer engagement.",
        ),
        (
            "2. Objectives",
            "- Automate periodic batch ingestion (CSVs + REST APIs) and near-real-time clickstream micro-batching.\n"
            "- Enforce automated data quality gates (schema, missing values, duplicates, 1-5 rating range checks).\n"
            "- Build a structured PostgreSQL warehouse and Feast + Redis feature store for consistent training & serving.\n"
            "- Track dataset lineage via DVC, experiment metrics via MLflow, and orchestrate the full DAG via Prefect.",
        ),
        (
            "3. Key Data Sources & Attributes",
            "- Clickstream Logs (Retailrocket events.csv): timestamp, visitorid, event (view/addtocart/transaction), itemid.\n"
            "- Transactional Purchase History (Olist CSVs): order_id, customer_unique_id, product_id, price, review_score (1-5).\n"
            "- Product Metadata (Olist Catalog + DummyJSON REST API): product_id, category_english, price, dimensions.\n"
            "- External APIs (FakeStore & DummyJSON Comments): external popularity ratings and NLP sentiment polarity.",
        ),
        (
            "4. Expected Pipeline Outputs",
            "- Clean, validated Parquet datasets and Exploratory Data Analysis (EDA) visual summaries.\n"
            "- Engineered user, item, and co-occurrence features stored in PostgreSQL and served via Feast + Redis.\n"
            "- Deployable recommendation models (SVD & Content-Based TF-IDF) with a live inference interface (src/infer.py).",
        ),
        (
            "5. Evaluation Metrics",
            "- Precision@10: Fraction of Top-10 recommended items relevant to the user.\n"
            "- Recall@10: Fraction of held-out user purchases captured within the Top-10 recommendations.\n"
            "- NDCG@10: Position-sensitive ranking quality rewarding relevant items placed higher in the Top-10 list.",
        ),
    ]

    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")
    y = 0.94

    fig.text(0.5, y, "Data Management for Machine Learning (S1-26_DSEZG529)", ha="center", fontsize=11, color="#4A5568")
    y -= 0.032
    fig.text(0.5, y, "Task 1: Problem Formulation Report — RecoMart", ha="center", fontsize=15, fontweight="bold", color="#1B365D")
    y -= 0.025
    fig.text(0.5, y, "End-to-End Data Management Pipeline for a Recommendation System", ha="center", fontsize=11, style="italic", color="#2E6B9E")
    y -= 0.045

    for heading, body in sections:
        fig.text(0.08, y, heading, fontsize=11.5, fontweight="bold", color="#1B365D")
        y -= 0.024
        wrapped_lines = []
        for para in body.split("\n"):
            wrapped_lines.extend(textwrap.wrap(para, width=92))
        block = "\n".join(wrapped_lines)
        fig.text(0.09, y, block, fontsize=9.5, va="top", color="#1A202C", linespacing=1.45)
        y -= (len(wrapped_lines) * 0.020) + 0.032

    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig)
    plt.close(fig)
    print(f"Created -> {pdf_path}")


def create_storage_structure_doc():
    path = "reports/03_storage_structure.md"
    content = (
        "# Task 3: Raw Data Storage Structure Documentation\n\n"
        "## Storage Architecture\n"
        "- **Object Store (Data Lake):** Local S3-compatible MinIO container (`recomart-minio`, ports `9000`/`9001`), bucket `recomart-raw`.\n"
        "- **Local Mirror:** `data/raw/` tracked and versioned via DVC (`data/raw.dvc`).\n"
        "- **Partitioning Scheme:** Partitioned hierarchically by `<source>/<type>/dt=<YYYY-MM-DD>/` to preserve immutable daily snapshots.\n\n"
        "## Partitioned Folder & Bucket Layout\n"
        f"{FENCE}text\n"
        "s3://recomart-raw/  (mirrored at data/raw/)\n"
        "├── csv/\n"
        "│   ├── clickstream/dt=YYYY-MM-DD/events.csv\n"
        "│   ├── transactions/dt=YYYY-MM-DD/\n"
        "│   │   ├── olist_orders_dataset.csv\n"
        "│   │   ├── olist_order_items_dataset.csv\n"
        "│   │   ├── olist_order_reviews_dataset.csv\n"
        "│   │   └── olist_customers_dataset.csv\n"
        "│   └── catalog/dt=YYYY-MM-DD/\n"
        "│       ├── olist_products_dataset.csv\n"
        "│       └── product_category_name_translation.csv\n"
        "├── api/\n"
        "│   ├── catalog/dt=YYYY-MM-DD/products.json\n"
        "│   └── external_scores/dt=YYYY-MM-DD/\n"
        "│       ├── fakestore_popularity.json\n"
        "│       └── dummyjson_comments.json\n"
        "└── stream/\n"
        "    └── clickstream/dt=YYYY-MM-DD/batch_HHMMSS.json\n"
        f"{FENCE}\n\n"
        "## Upload Scripts & Configuration\n"
        "- `src/ingest_batch.py`: Automated batch ingestion with exponential backoff retries (`max_retries=3`), `boto3` S3 upload, and audit logging (`logs/ingestion.log`).\n"
        "- `src/ingest_stream.py`: Near-real-time micro-batch streamer pushing JSON clickstream windows to MinIO and updating live Redis event counters (`logs/stream_ingestion.log`).\n"
        "- `docker-compose.yml`: Defines MinIO (`recomart-minio`), PostgreSQL (`recomart-postgres`), and Redis (`recomart-redis`).\n"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Created -> {path}")


def create_feature_logic_doc():
    path = "reports/06_feature_logic_summary.md"
    content = (
        "# Task 6: Feature Engineering & Warehouse Logic Summary\n\n"
        "## Structured Warehouse (`recomart_dw` in PostgreSQL)\n"
        "Defined in `sql/schema.sql` and populated by `src/transform.py`:\n"
        "- **`dim_users` (95,420 rows):** Aggregated per `customer_unique_id`. Computes `user_order_count` (activity frequency), `user_total_spend`, `user_avg_rating`, label-encoded `state_encoded`, and estimated `user_click_events`.\n"
        "- **`dim_items` (32,951 rows):** Aggregated per `product_id`. Computes `item_purchase_count` (popularity), `item_avg_rating`, `avg_price`, Min-Max `price_normalized`, label-encoded `category_encoded`, `external_popularity_score` (from FakeStore API), and `external_sentiment_score` (TextBlob NLP polarity from DummyJSON Comments API).\n"
        "- **`fact_interactions` (112,650 rows):** Cleaned user-item transactional fact table linking orders, prices, normalized timestamps, and 1–5 review scores.\n"
        "- **`item_co_occurrence` (4,058 rows):** Pairwise co-purchased product counts from multi-item orders with normalized cosine similarity score: `co_occurrence_count / sqrt(count_A * count_B)`.\n\n"
        "## Downstream Consumers\n"
        "- Exported as Parquet files (`data/features/*.parquet`) for DVC versioning and Feast offline historical retrieval.\n"
        "- Active entities pushed to Redis online store (`recomart-redis`) for sub-millisecond inference lookups.\n"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Created -> {path}")


def create_readme():
    content = (
        "# RecoMart End-to-End Data Management Pipeline (DMML — S1-26_DSEZG529)\n\n"
        "Automated, modular data management pipeline for an e-commerce recommendation system using "
        "**Docker Desktop (MinIO, PostgreSQL, Redis)**, **Great Expectations**, **Feast**, **DVC**, **MLflow**, and **Prefect**.\n\n"
        "## Key Results\n"
        "- **Warehouse Loaded:** 95,420 users, 32,951 items, 112,650 interactions, 4,058 co-occurrence pairs.\n"
        "- **Matrix Sparsity:** 99.9964% (diagnosed via EDA plots).\n"
        "- **Model Performance (Top-10 Ranking, 500 Held-Out Users, Tracked in MLflow):**\n"
        "  - **Content-Based (TF-IDF + Metadata):** Precision@10 = `0.0610`, Recall@10 = `0.6100`, NDCG@10 = `0.4510`\n"
        "  - **Collaborative Filtering (SVD, k=25):** Precision@10 = `0.0466`, Recall@10 = `0.4660`, NDCG@10 = `0.2946`\n"
        "- **Orchestration:** Full 9-task Prefect DAG completed end-to-end in `3m 31s`.\n\n"
        "## Quick Start\n"
        f"{FENCE}powershell\n"
        "docker compose up -d\n"
        ".\\.venv\\Scripts\\Activate.ps1\n"
        "# Terminal 1:\n"
        "prefect server start\n"
        "# Terminal 2:\n"
        '$env:PREFECT_API_URL="http://127.0.0.1:4200/api"\n'
        "python src/pipeline_flow.py\n"
        f"{FENCE}\n"
    )
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(content)
    print("Created -> README.md")


if __name__ == "__main__":
    create_problem_formulation_pdf()
    create_storage_structure_doc()
    create_feature_logic_doc()
    create_readme()