import os
import pandas as pd

os.makedirs("app_data", exist_ok=True)

interactions = pd.read_parquet("data/processed/clean_interactions.parquet")
user_features = pd.read_parquet("data/features/user_features.parquet")
item_features = pd.read_parquet("data/features/item_features.parquet")
co_occurrence = pd.read_parquet("data/features/item_co_occurrence.parquet")

# Keep our verified demo customer + top active repeat shoppers (200 users total)
demo_uid = "871766c5855e863f6eccc05f988b23cb"
top_users = (
    user_features.sort_values(["user_order_count", "user_total_spend"], ascending=False)
    .head(200)["customer_unique_id"]
    .tolist()
)
selected_users = list(dict.fromkeys([demo_uid] + top_users))[:200]

sample_inter = interactions[interactions["customer_unique_id"].isin(selected_users)].copy()
sample_users = user_features[user_features["customer_unique_id"].isin(selected_users)].copy()

# Include all items purchased by selected users + top 3,000 popular items
bought_items = set(sample_inter["product_id"].unique())
popular_items = set(
    item_features.sort_values("item_purchase_count", ascending=False)
    .head(3000)["product_id"]
    .unique()
)
keep_items = bought_items | popular_items
sample_items = item_features[item_features["product_id"].isin(keep_items)].copy()
sample_co = co_occurrence[
    co_occurrence["product_id_a"].isin(keep_items) | co_occurrence["product_id_b"].isin(keep_items)
].head(1000).copy()

sample_inter.to_parquet("app_data/interactions_sample.parquet", index=False)
sample_users.to_parquet("app_data/user_features_sample.parquet", index=False)
sample_items.to_parquet("app_data/item_features_sample.parquet", index=False)
sample_co.to_parquet("app_data/item_co_occurrence_sample.parquet", index=False)

print(
    f"Exported app_data/ -> users: {len(sample_users)}, items: {len(sample_items)}, "
    f"interactions: {len(sample_inter)}, co-occurrence: {len(sample_co)}"
)