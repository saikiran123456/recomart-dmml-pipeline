# Task 3: Raw Data Storage Structure Documentation

## Storage Architecture
- **Object Store (Data Lake):** Local S3-compatible MinIO container (`recomart-minio`, ports `9000`/`9001`), bucket `recomart-raw`.
- **Local Mirror:** `data/raw/` tracked and versioned via DVC (`data/raw.dvc`).
- **Partitioning Scheme:** Partitioned hierarchically by `<source>/<type>/dt=<YYYY-MM-DD>/` to preserve immutable daily snapshots.

## Partitioned Folder & Bucket Layout
```text
s3://recomart-raw/  (mirrored at data/raw/)
├── csv/
│   ├── clickstream/dt=YYYY-MM-DD/events.csv
│   ├── transactions/dt=YYYY-MM-DD/
│   │   ├── olist_orders_dataset.csv
│   │   ├── olist_order_items_dataset.csv
│   │   ├── olist_order_reviews_dataset.csv
│   │   └── olist_customers_dataset.csv
│   └── catalog/dt=YYYY-MM-DD/
│       ├── olist_products_dataset.csv
│       └── product_category_name_translation.csv
├── api/
│   ├── catalog/dt=YYYY-MM-DD/products.json
│   └── external_scores/dt=YYYY-MM-DD/
│       ├── fakestore_popularity.json
│       └── dummyjson_comments.json
└── stream/
    └── clickstream/dt=YYYY-MM-DD/batch_HHMMSS.json
```

## Upload Scripts & Configuration
- `src/ingest_batch.py`: Automated batch ingestion with exponential backoff retries (`max_retries=3`), `boto3` S3 upload, and audit logging (`logs/ingestion.log`).
- `src/ingest_stream.py`: Near-real-time micro-batch streamer pushing JSON clickstream windows to MinIO and updating live Redis event counters (`logs/stream_ingestion.log`).
- `docker-compose.yml`: Defines MinIO (`recomart-minio`), PostgreSQL (`recomart-postgres`), and Redis (`recomart-redis`).
