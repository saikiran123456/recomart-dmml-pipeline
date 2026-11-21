import os
import json
import logging
from datetime import datetime, timezone
from itertools import combinations
from collections import Counter
import numpy as np
import pandas as pd
from textblob import TextBlob
from sqlalchemy import create_engine, text

# --- Configuration ---
TODAY = datetime.now().strftime("%Y-%m-%d")
RAW_ROOT = "data/raw"
PROC_DIR = "data/processed"
FEAT_DIR = "data/features"
SCHEMA_FILE = "sql/schema.sql"
LOG_FILE = "logs/transformation.log"
DB_URI = "postgresql+psycopg2://recomart:password123@localhost:5432/recomart_dw"

os.makedirs(FEAT_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)

def compute_external_api_signals():
    """Extract popularity scores and NLP sentiment polarity from ingested REST APIs."""
    pop_path = f"{RAW_ROOT}/api/external_scores/dt={TODAY}/fakestore_popularity.json"
    comm_path = f"{RAW_ROOT}/api/external_scores/dt={TODAY}/dummyjson_comments.json"

    with open(pop_path, "r", encoding="utf-8") as f:
        pop_data = json.load(f)
    pop_scores = [item["rating"]["rate"] for item in pop_data if "rating" in item]
    mean_pop = float(np.mean(pop_scores)) if pop_scores else 4.0

    with open(comm_path, "r", encoding="utf-8") as f:
        comm_data = json.load(f).get("comments", [])
    sentiments = [TextBlob(c["body"]).sentiment.polarity for c in comm_data if "body" in c]
    mean_sent = float(np.mean(sentiments)) if sentiments else 0.15

    return mean_pop, mean_sent

def transform_and_load():
    logging.info("=== Starting Task 6: Feature Engineering & Warehouse Load ===")
    now_utc = pd.Timestamp.now(tz="UTC")

    interactions = pd.read_parquet(f"{PROC_DIR}/clean_interactions.parquet")
    clicks = pd.read_parquet(f"{PROC_DIR}/clean_clickstream.parquet")

    # 1. User Features (Grouped strictly by customer_unique_id to guarantee PK uniqueness)
    clicks_per_visitor = clicks.groupby("visitorid").size().values
    avg_click_baseline = int(np.mean(clicks_per_visitor)) if len(clicks_per_visitor) > 0 else 1

    user_features = (
        interactions.groupby("customer_unique_id", as_index=False)
        .agg(
            customer_state=("customer_state", "first"),
            state_encoded=("state_encoded", "first"),
            user_order_count=("order_id", "nunique"),
            user_total_spend=("price", "sum"),
            user_avg_rating=("review_score", "mean"),
        )
    )
    user_features["user_click_events"] = user_features["user_order_count"] * avg_click_baseline
    user_features["updated_at"] = now_utc

    # 2. Item Features (Grouped strictly by product_id to guarantee PK uniqueness)
    mean_pop, mean_sent = compute_external_api_signals()
    item_features = (
        interactions.groupby("product_id", as_index=False)
        .agg(
            category_english=("category_english", "first"),
            category_encoded=("category_encoded", "first"),
            avg_price=("price", "mean"),
            price_normalized=("price_normalized", "mean"),
            item_purchase_count=("order_id", "count"),
            item_avg_rating=("review_score", "mean"),
        )
    )
    item_features["external_popularity_score"] = (
        (item_features["item_avg_rating"] / 5.0) * mean_pop
    ).round(3)
    item_features["external_sentiment_score"] = round(mean_sent, 4)
    item_features["updated_at"] = now_utc

    # 3. Item Co-Occurrence & Cosine Similarity Features
    baskets = interactions.groupby("order_id")["product_id"].apply(lambda x: sorted(set(x)))
    multi_baskets = [b for b in baskets if len(b) > 1]

    pair_counts = Counter()
    for basket in multi_baskets:
        for pair in combinations(basket, 2):
            pair_counts[pair] += 1

    item_counts = dict(zip(item_features["product_id"], item_features["item_purchase_count"]))
    co_rows = []
    for (prod_a, prod_b), cnt in pair_counts.most_common(5000):
        denom = np.sqrt(item_counts.get(prod_a, 1) * item_counts.get(prod_b, 1))
        sim = float(cnt / denom) if denom > 0 else 0.0
        co_rows.append((prod_a, prod_b, cnt, round(sim, 4)))

    co_occurrence_df = pd.DataFrame(
        co_rows, columns=["product_id_a", "product_id_b", "co_occurrence_count", "similarity_score"]
    )

    # 4. Prepare Fact Interactions Table
    fact_df = interactions[[
        "customer_unique_id", "product_id", "order_id",
        "review_score", "price", "price_normalized",
        "timestamp_normalized", "order_purchase_timestamp"
    ]].rename(columns={"order_purchase_timestamp": "event_timestamp"}).copy()
    fact_df["event_timestamp"] = pd.to_datetime(fact_df["event_timestamp"])

    # 5. Execute SQL Schema & Load into PostgreSQL Data Warehouse
    engine = create_engine(DB_URI)
    with open(SCHEMA_FILE, "r", encoding="utf-8") as f:
        ddl = f.read()

    with engine.begin() as conn:
        for stmt in ddl.split(";"):
            if stmt.strip():
                conn.execute(text(stmt))
        conn.execute(text("TRUNCATE TABLE dim_users, dim_items, fact_interactions, item_co_occurrence RESTART IDENTITY;"))

    user_features.to_sql("dim_users", engine, if_exists="append", index=False, chunksize=5000)
    item_features.to_sql("dim_items", engine, if_exists="append", index=False, chunksize=5000)
    fact_df.to_sql("fact_interactions", engine, if_exists="append", index=False, chunksize=5000)
    if not co_occurrence_df.empty:
        co_occurrence_df.to_sql("item_co_occurrence", engine, if_exists="append", index=False, chunksize=5000)

    logging.info(
        f"Loaded PostgreSQL Warehouse -> dim_users ({len(user_features):,}), "
        f"dim_items ({len(item_features):,}), fact_interactions ({len(fact_df):,}), "
        f"co_occurrence ({len(co_occurrence_df):,})"
    )

    # 6. Export Parquet Feature Tables for Feast Feature Store & DVC
    user_features.to_parquet(f"{FEAT_DIR}/user_features.parquet", index=False)
    item_features.to_parquet(f"{FEAT_DIR}/item_features.parquet", index=False)
    co_occurrence_df.to_parquet(f"{FEAT_DIR}/item_co_occurrence.parquet", index=False)
    logging.info(f"Saved feature Parquet files to {FEAT_DIR}/")
    logging.info("=== Task 6 Completed Successfully ===")

if __name__ == "__main__":
    transform_and_load()