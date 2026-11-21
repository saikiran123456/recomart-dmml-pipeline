import os
import sys
import pickle
import logging
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import normalize
from feast import FeatureStore

LOG_FILE = "logs/inference.log"
os.makedirs("logs", exist_ok=True)
os.makedirs("reports", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)

def recommend_top_k(customer_id=None, top_k=10):
    logging.info("=== Starting Live Recommendation Inference ===")

    interactions = pd.read_parquet("data/processed/clean_interactions.parquet")
    item_features = pd.read_parquet("data/features/item_features.parquet")

    # Default to a known active customer materialized in Redis
    if not customer_id:
        customer_id = interactions["customer_unique_id"].iloc[0]

    # 1. Fetch live user features from Feast Online Store (Redis)
    store = FeatureStore(repo_path="feature_repo")
    user_online = store.get_online_features(
        features=[
            "user_stats_features:user_order_count",
            "user_stats_features:user_total_spend",
            "user_stats_features:user_avg_rating",
            "user_stats_features:user_click_events",
        ],
        entity_rows=[{"customer_unique_id": customer_id}],
    ).to_dict()

    logging.info(f"Live Online User Profile (Redis/Feast) for {customer_id}: {user_online}")

    # 2. Load trained Content-Based TF-IDF artifact (winning model from MLflow)
    with open("models/content_based_model.pkl", "rb") as f:
        cb_bundle = pickle.load(f)
    tfidf = cb_bundle["vectorizer"]

    cat_mat = tfidf.transform(item_features["category_english"].fillna("unknown"))
    num_mat = item_features[["price_normalized", "item_avg_rating", "external_popularity_score"]].fillna(0).values
    num_mat = num_mat / (np.linalg.norm(num_mat, axis=0, keepdims=True) + 1e-9)
    item_content_mat = normalize(hstack([cat_mat, csr_matrix(num_mat)]).tocsr())

    # 3. Build user preference vector from purchase history
    bought_ids = set(interactions.loc[interactions["customer_unique_id"] == customer_id, "product_id"])
    item_to_idx = {pid: idx for idx, pid in enumerate(item_features["product_id"])}
    bought_indices = [item_to_idx[p] for p in bought_ids if p in item_to_idx]

    if bought_indices:
        user_profile = np.asarray(item_content_mat[bought_indices].mean(axis=0))
        scores = (item_content_mat @ user_profile.T).ravel()
    else:
        # Cold-start fallback using external API popularity & item rating
        scores = item_features["external_popularity_score"].fillna(0).values

    # Exclude already-purchased items
    for idx in bought_indices:
        scores[idx] = -1.0

    top_indices = np.argsort(-scores)[:top_k]
    recs = item_features.iloc[top_indices][[
        "product_id", "category_english", "avg_price", "item_avg_rating", "external_popularity_score"
    ]].copy()
    recs["match_score"] = np.round(scores[top_indices], 4)
    recs["avg_price"] = recs["avg_price"].round(2)
    recs["item_avg_rating"] = recs["item_avg_rating"].round(2)

    output_text = (
        f"Customer ID: {customer_id}\n"
        f"Online Features (Redis): orders={user_online['user_order_count'][0]}, "
        f"spend={user_online['user_total_spend'][0]}, avg_rating={user_online['user_avg_rating'][0]}\n\n"
        f"Top-{top_k} Recommended Products:\n"
        f"{recs.to_string(index=False)}\n"
    )
    print("\n" + output_text)

    with open("reports/inference_sample_output.txt", "w", encoding="utf-8") as f:
        f.write(output_text)

    logging.info("Saved sample inference output -> reports/inference_sample_output.txt")
    logging.info("=== Live Inference Completed Successfully ===")
    return recs

if __name__ == "__main__":
    cid = sys.argv[1] if len(sys.argv) > 1 else None
    recommend_top_k(customer_id=cid, top_k=10)