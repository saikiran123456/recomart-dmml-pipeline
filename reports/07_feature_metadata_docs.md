# RecoMart Feature Store Metadata Documentation

- **Feature Store Engine:** Feast (Local SQLite Registry + Parquet Offline Store + Redis Online Store)
- **Entities:** `customer` (`customer_unique_id`), `product` (`product_id`)

| Feature View | Feature Name | Data Type | Source Table | Transformation Logic |
| :--- | :--- | :--- | :--- | :--- |
| `user_stats_features` | `state_encoded` | `Int64` | `dim_users` | Label-encoded customer state |
| `user_stats_features` | `user_order_count` | `Int64` | `dim_users` | Count of distinct orders per user |
| `user_stats_features` | `user_total_spend` | `Float64` | `dim_users` | Sum of item prices purchased by user |
| `user_stats_features` | `user_avg_rating` | `Float64` | `dim_users` | Mean review score (1–5) submitted by user |
| `user_stats_features` | `user_click_events` | `Int64` | `dim_users` | Estimated session click frequency |
| `item_stats_features` | `category_encoded` | `Int64` | `dim_items` | Label-encoded English product category |
| `item_stats_features` | `avg_price` | `Float64` | `dim_items` | Mean unit price of product |
| `item_stats_features` | `price_normalized` | `Float64` | `dim_items` | Min-Max scaled unit price `[0, 1]` |
| `item_stats_features` | `item_purchase_count` | `Int64` | `dim_items` | Total order frequency per product |
| `item_stats_features` | `item_avg_rating` | `Float64` | `dim_items` | Mean review score (1–5) received by product |
| `item_stats_features` | `external_popularity_score` | `Float64` | `dim_items` | Scaled rating enriched with FakeStore API popularity |
| `item_stats_features` | `external_sentiment_score` | `Float64` | `dim_items` | TextBlob NLP polarity from DummyJSON comments API |
