# Task 6: Feature Engineering & Warehouse Logic Summary

## Structured Warehouse (`recomart_dw` in PostgreSQL)
Defined in `sql/schema.sql` and populated by `src/transform.py`:
- **`dim_users` (95,420 rows):** Aggregated per `customer_unique_id`. Computes `user_order_count` (activity frequency), `user_total_spend`, `user_avg_rating`, label-encoded `state_encoded`, and estimated `user_click_events`.
- **`dim_items` (32,951 rows):** Aggregated per `product_id`. Computes `item_purchase_count` (popularity), `item_avg_rating`, `avg_price`, Min-Max `price_normalized`, label-encoded `category_encoded`, `external_popularity_score` (from FakeStore API), and `external_sentiment_score` (TextBlob NLP polarity from DummyJSON Comments API).
- **`fact_interactions` (112,650 rows):** Cleaned user-item transactional fact table linking orders, prices, normalized timestamps, and 1–5 review scores.
- **`item_co_occurrence` (4,058 rows):** Pairwise co-purchased product counts from multi-item orders with normalized cosine similarity score: `co_occurrence_count / sqrt(count_A * count_B)`.

## Downstream Consumers
- Exported as Parquet files (`data/features/*.parquet`) for DVC versioning and Feast offline historical retrieval.
- Active entities pushed to Redis online store (`recomart-redis`) for sub-millisecond inference lookups.
