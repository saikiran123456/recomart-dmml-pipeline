import os
import subprocess
import logging
import pandas as pd
from feast import FeatureStore

LOG_FILE = "logs/feature_store.log"
os.makedirs("logs", exist_ok=True)
os.makedirs("reports", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)

def run_feature_store_demo():
    logging.info("=== Starting Task 7: Feast Feature Store Registration & Retrieval ===")

    # 1. Register Feature Views in Feast Registry
    subprocess.run(["feast", "apply"], cwd="feature_repo", check=True)
    logging.info("Executed `feast apply` -> Registry updated")

    store = FeatureStore(repo_path="feature_repo")

    # 2. Load Engineered Feature Tables & Push Top Active Entities to Redis Online Store
    user_df = pd.read_parquet("data/features/user_features.parquet")
    item_df = pd.read_parquet("data/features/item_features.parquet")

    # Ensure sample rows from clean_interactions are included in the online store
    interactions = pd.read_parquet("data/processed/clean_interactions.parquet").head(10)
    top_users = pd.concat([
        user_df[user_df["customer_unique_id"].isin(interactions["customer_unique_id"])],
        user_df.head(1000)
    ]).drop_duplicates(subset=["customer_unique_id"])

    top_items = pd.concat([
        item_df[item_df["product_id"].isin(interactions["product_id"])],
        item_df.head(1000)
    ]).drop_duplicates(subset=["product_id"])

    store.write_to_online_store("user_stats_features", top_users)
    store.write_to_online_store("item_stats_features", top_items)
    logging.info(f"Pushed {len(top_users):,} users and {len(top_items):,} items into Redis online store")

    # 3. Sample Offline Historical Retrieval (for Model Training)
    sample_inter = interactions.head(5)
    entity_df = pd.DataFrame({
        "customer_unique_id": sample_inter["customer_unique_id"].tolist(),
        "product_id": sample_inter["product_id"].tolist(),
        "event_timestamp": [pd.Timestamp.now(tz="UTC")] * len(sample_inter),
    })

    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "user_stats_features:user_order_count",
            "user_stats_features:user_avg_rating",
            "item_stats_features:item_purchase_count",
            "item_stats_features:item_avg_rating",
            "item_stats_features:external_popularity_score",
        ],
    ).to_df()

    logging.info(f"Offline Historical Feature Retrieval Sample ({len(training_df)} rows):\n{training_df.to_string(index=False)}")

    # 4. Sample Online Low-Latency Retrieval (for Real-Time Inference)
    sample_user = entity_df["customer_unique_id"].iloc[0]
    sample_item = entity_df["product_id"].iloc[0]

    online_vec = store.get_online_features(
        features=[
            "user_stats_features:user_order_count",
            "user_stats_features:user_total_spend",
            "user_stats_features:user_avg_rating",
            "item_stats_features:avg_price",
            "item_stats_features:item_avg_rating",
            "item_stats_features:external_popularity_score",
        ],
        entity_rows=[{"customer_unique_id": sample_user, "product_id": sample_item}],
    ).to_dict()

    logging.info(f"Online Inference Feature Vector (from Redis): {online_vec}")

    # 5. Generate Feature Metadata Documentation
    doc_path = "reports/07_feature_metadata_docs.md"
    with open(doc_path, "w", encoding="utf-8") as f:
        f.write("# RecoMart Feature Store Metadata Documentation\n\n")
        f.write("- **Feature Store Engine:** Feast (Local SQLite Registry + Parquet Offline Store + Redis Online Store)\n")
        f.write("- **Entities:** `customer` (`customer_unique_id`), `product` (`product_id`)\n\n")
        f.write("| Feature View | Feature Name | Data Type | Source Table | Transformation Logic |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        f.write("| `user_stats_features` | `state_encoded` | `Int64` | `dim_users` | Label-encoded customer state |\n")
        f.write("| `user_stats_features` | `user_order_count` | `Int64` | `dim_users` | Count of distinct orders per user |\n")
        f.write("| `user_stats_features` | `user_total_spend` | `Float64` | `dim_users` | Sum of item prices purchased by user |\n")
        f.write("| `user_stats_features` | `user_avg_rating` | `Float64` | `dim_users` | Mean review score (1–5) submitted by user |\n")
        f.write("| `user_stats_features` | `user_click_events` | `Int64` | `dim_users` | Estimated session click frequency |\n")
        f.write("| `item_stats_features` | `category_encoded` | `Int64` | `dim_items` | Label-encoded English product category |\n")
        f.write("| `item_stats_features` | `avg_price` | `Float64` | `dim_items` | Mean unit price of product |\n")
        f.write("| `item_stats_features` | `price_normalized` | `Float64` | `dim_items` | Min-Max scaled unit price `[0, 1]` |\n")
        f.write("| `item_stats_features` | `item_purchase_count` | `Int64` | `dim_items` | Total order frequency per product |\n")
        f.write("| `item_stats_features` | `item_avg_rating` | `Float64` | `dim_items` | Mean review score (1–5) received by product |\n")
        f.write("| `item_stats_features` | `external_popularity_score` | `Float64` | `dim_items` | Scaled rating enriched with FakeStore API popularity |\n")
        f.write("| `item_stats_features` | `external_sentiment_score` | `Float64` | `dim_items` | TextBlob NLP polarity from DummyJSON comments API |\n")

    logging.info(f"Generated Feature Metadata Documentation -> {doc_path}")
    logging.info("=== Task 7 Completed Successfully ===")

if __name__ == "__main__":
    run_feature_store_demo()