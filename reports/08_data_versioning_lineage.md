# RecoMart Data Versioning & Lineage Documentation

- **Versioning Tool:** DVC (Data Version Control) + Git
- **Remote Store:** `local_storage` (`./dvc_storage`)
- **Last Recorded:** 2026-11-21 16:13:37

## Dataset Lineage & Version Hashes

| Stage | Dataset Path | Files | Size (MB) | DVC MD5 Hash | Data Source | Applied Transformations |
| :--- | :--- | :---: | :---: | :--- | :--- | :--- |
| 1. Raw Data Lake | `data/raw` | 45 | 114.68 | `e64eefe16a47cf558ab35cd029e77b61.dir` | Kaggle Retailrocket CSV, Olist CSVs (6 files), DummyJSON & FakeStore REST APIs | Automated batch & near-real-time micro-batch ingestion; partitioned by source/type/dt |
| 2. Cleaned & Prepared Datasets | `data/processed` | 3 | 17.61 | `eb18d3cfdf54b69d88d48ca897044e53.dir` | data/raw (validated via Great Expectations) | Deduplication (reviews & clickstreams), missing category imputation, label encoding, Min-Max scaling |
| 3. Engineered Feature Tables | `data/features` | 3 | 4.79 | `0e637fb1165a38f63700cf0e7a3c3784.dir` | data/processed + External API sentiment/popularity JSONs | User/item aggregations, co-occurrence cosine similarity, PostgreSQL load, Feast Parquet export |

## Reproducibility Workflow
1. `git checkout <commit-hash>` to select code and `.dvc` pointer versions.
2. `dvc checkout` (or `dvc pull`) to restore exact dataset snapshots.
3. Run `dvc status` to detect any untracked data drift.
