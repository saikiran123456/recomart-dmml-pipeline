# RecoMart End-to-End Data Management Pipeline (DMML — S1-26_DSEZG529)

Automated, modular data management pipeline for an e-commerce recommendation system using **Docker Desktop (MinIO, PostgreSQL, Redis)**, **Great Expectations**, **Feast**, **DVC**, **MLflow**, and **Prefect**.

## Key Results
- **Warehouse Loaded:** 95,420 users, 32,951 items, 112,650 interactions, 4,058 co-occurrence pairs.
- **Matrix Sparsity:** 99.9964% (diagnosed via EDA plots).
- **Model Performance (Top-10 Ranking, 500 Held-Out Users, Tracked in MLflow):**
  - **Content-Based (TF-IDF + Metadata):** Precision@10 = `0.0610`, Recall@10 = `0.6100`, NDCG@10 = `0.4510`
  - **Collaborative Filtering (SVD, k=25):** Precision@10 = `0.0466`, Recall@10 = `0.4660`, NDCG@10 = `0.2946`
- **Orchestration:** Full 9-task Prefect DAG completed end-to-end in `3m 31s`.

## Quick Start
```powershell
docker compose up -d
.\.venv\Scripts\Activate.ps1
# Terminal 1:
prefect server start
# Terminal 2:
$env:PREFECT_API_URL="http://127.0.0.1:4200/api"
python src/pipeline_flow.py
```
