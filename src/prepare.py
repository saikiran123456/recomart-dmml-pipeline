import os
import logging
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# --- Configuration ---
TODAY = datetime.now().strftime("%Y-%m-%d")
RAW_ROOT = "data/raw"
PROC_DIR = "data/processed"
PLOT_PNG = "reports/05_eda_summary_plots.png"
PLOT_PDF = "reports/05_eda_summary_plots.pdf"
LOG_FILE = "logs/preparation.log"

os.makedirs(PROC_DIR, exist_ok=True)
os.makedirs("reports", exist_ok=True)
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)

def prepare_and_eda():
    logging.info("=== Starting Task 5: Data Preparation & EDA ===")

    # 1. Load Raw Datasets
    orders = pd.read_csv(f"{RAW_ROOT}/csv/transactions/dt={TODAY}/olist_orders_dataset.csv")
    items = pd.read_csv(f"{RAW_ROOT}/csv/transactions/dt={TODAY}/olist_order_items_dataset.csv")
    reviews = pd.read_csv(f"{RAW_ROOT}/csv/transactions/dt={TODAY}/olist_order_reviews_dataset.csv")
    customers = pd.read_csv(f"{RAW_ROOT}/csv/transactions/dt={TODAY}/olist_customers_dataset.csv")
    products = pd.read_csv(f"{RAW_ROOT}/csv/catalog/dt={TODAY}/olist_products_dataset.csv")
    translations = pd.read_csv(f"{RAW_ROOT}/csv/catalog/dt={TODAY}/product_category_name_translation.csv")
    clicks = pd.read_csv(f"{RAW_ROOT}/csv/clickstream/dt={TODAY}/events.csv")

    # 2. Clean Duplicates & Missing Values (Addressing Task 4 Findings)
    reviews_clean = reviews.drop_duplicates(subset=["review_id"]).copy()
    # Keep average review score per order if multiple reviews exist for one order
    order_ratings = (
        reviews_clean.groupby("order_id", as_index=False)
        .agg({"review_score": "mean", "review_comment_message": "first"})
    )
    order_ratings["review_score"] = order_ratings["review_score"].round().astype(int)

    # Impute 610 missing product categories and translate Portuguese -> English
    products["product_category_name"] = products["product_category_name"].fillna("unknown")
    products_clean = products.merge(translations, on="product_category_name", how="left")
    products_clean["category_english"] = products_clean["product_category_name_english"].fillna(
        products_clean["product_category_name"]
    )

    # Deduplicate clickstream events while preserving distinct event stages
    clicks_clean = clicks.drop_duplicates(subset=["visitorid", "itemid", "event"]).copy()
    logging.info(f"Clickstream cleaned: {len(clicks):,} -> {len(clicks_clean):,} rows")

    # 3. Build Unified User-Item Interaction Table
    interactions = (
        items[["order_id", "product_id", "price"]]
        .merge(orders[["order_id", "customer_id", "order_status", "order_purchase_timestamp"]], on="order_id", how="inner")
        .merge(customers[["customer_id", "customer_unique_id", "customer_city", "customer_state"]], on="customer_id", how="inner")
        .merge(order_ratings, on="order_id", how="left")
        .merge(products_clean[["product_id", "category_english"]], on="product_id", how="left")
    )

    # Handle missing ratings (impute with median rating for unreviewed orders)
    median_rating = int(interactions["review_score"].median())
    interactions["review_score"] = interactions["review_score"].fillna(median_rating).astype(int)
    interactions["category_english"] = interactions["category_english"].fillna("unknown")
    interactions = interactions.dropna(subset=["customer_unique_id", "product_id", "price"]).copy()

    # 4. Encode Categorical Attributes
    cat_encoder = LabelEncoder()
    state_encoder = LabelEncoder()
    interactions["category_encoded"] = cat_encoder.fit_transform(interactions["category_english"])
    interactions["state_encoded"] = state_encoder.fit_transform(interactions["customer_state"])
    products_clean["category_encoded"] = cat_encoder.transform(products_clean["category_english"])

    # 5. Normalize Numerical Variables (Price & Timestamp -> [0, 1])
    scaler = MinMaxScaler()
    interactions["price_normalized"] = scaler.fit_transform(interactions[["price"]])

    ts = pd.to_datetime(interactions["order_purchase_timestamp"], errors="coerce")
    epoch_sec = ((ts - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")).fillna(0).values.reshape(-1, 1)
    interactions["timestamp_normalized"] = scaler.fit_transform(epoch_sec)

    # 6. Calculate Matrix Sparsity & Save Prepared Parquet Datasets
    n_users = interactions["customer_unique_id"].nunique()
    n_items = interactions["product_id"].nunique()
    n_inter = len(interactions)
    sparsity = (1.0 - (n_inter / (n_users * n_items))) * 100
    logging.info(f"Prepared Interactions: {n_inter:,} rows | Users: {n_users:,} | Items: {n_items:,} | Sparsity: {sparsity:.4f}%")

    interactions.to_parquet(f"{PROC_DIR}/clean_interactions.parquet", index=False)
    products_clean.to_parquet(f"{PROC_DIR}/clean_products.parquet", index=False)
    clicks_clean.to_parquet(f"{PROC_DIR}/clean_clickstream.parquet", index=False)
    logging.info(f"Saved cleaned Parquet datasets to {PROC_DIR}/")

    # 7. Exploratory Data Analysis (EDA) Summary Plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"RecoMart Exploratory Data Analysis (EDA) Summary\n"
        f"Users: {n_users:,} | Items: {n_items:,} | Interactions: {n_inter:,} | Matrix Sparsity: {sparsity:.4f}%",
        fontsize=13, fontweight="bold"
    )

    # Plot 1: Rating Distribution (Histogram / Bar)
    rating_counts = interactions["review_score"].value_counts().sort_index()
    axes[0, 0].bar(rating_counts.index.astype(str), rating_counts.values, color="#1B365D")
    axes[0, 0].set_title("1. User Rating Distribution (1–5 Scale)", fontweight="bold")
    axes[0, 0].set_xlabel("Review Score")
    axes[0, 0].set_ylabel("Interaction Count")

    # Plot 2: Clickstream Event Distribution
    event_counts = clicks_clean["event"].value_counts()
    axes[0, 1].bar(event_counts.index, event_counts.values, color="#008080")
    axes[0, 1].set_title("2. Clickstream Event Funnel Distribution", fontweight="bold")
    axes[0, 1].set_xlabel("Event Type")
    axes[0, 1].set_ylabel("Unique User-Item Events")

    # Plot 3: Top 10 Popular Categories
    top_cats = interactions["category_english"].value_counts().head(10)[::-1]
    axes[1, 0].barh(top_cats.index, top_cats.values, color="#2E6B9E")
    axes[1, 0].set_title("3. Top 10 Product Categories (Item Popularity)", fontweight="bold")
    axes[1, 0].set_xlabel("Purchases")

    # Plot 4: Sparsity Heatmap (Top 25 Active Users x Top 25 Popular Items)
    top_users = interactions["customer_unique_id"].value_counts().head(25).index
    top_items = interactions["product_id"].value_counts().head(25).index
    sub = interactions[
        interactions["customer_unique_id"].isin(top_users) &
        interactions["product_id"].isin(top_items)
    ]
    pivot = sub.pivot_table(
        index="customer_unique_id", columns="product_id",
        values="review_score", aggfunc="count", fill_value=0
    ).reindex(index=top_users, columns=top_items, fill_value=0)

    im = axes[1, 1].imshow(pivot.values, cmap="Blues", aspect="auto")
    axes[1, 1].set_title("4. Interaction Sparsity Heatmap (Top 25 Users × Top 25 Items)", fontweight="bold")
    axes[1, 1].set_xlabel("Top 25 Products")
    axes[1, 1].set_ylabel("Top 25 Users")
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04, label="Interactions")

    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    plt.savefig(PLOT_PNG, dpi=200)
    with PdfPages(PLOT_PDF) as pdf:
        pdf.savefig(fig)
    plt.close(fig)
    logging.info(f"Saved EDA summary plots -> {PLOT_PNG} and {PLOT_PDF}")
    logging.info("=== Task 5 Completed Successfully ===")

if __name__ == "__main__":
    prepare_and_eda()