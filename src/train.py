import os
import pickle
import logging
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import mlflow
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# --- Configuration ---
PROC_DIR = "data/processed"
FEAT_DIR = "data/features"
MODEL_DIR = "models"
REPORT_PDF = "reports/09_model_performance_report.pdf"
LOG_FILE = "logs/training.log"

TOP_K = 10
N_FACTORS = 25
EVAL_USERS_LIMIT = 500
NUM_NEG_CANDIDATES = 99
RANDOM_SEED = 42

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs("reports", exist_ok=True)
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)

def ranking_metrics_at_k(ranked_items, true_item, k=10):
    top_k = ranked_items[:k]
    if true_item in top_k:
        rank = top_k.index(true_item) + 1
        precision = 1.0 / k
        recall = 1.0
        ndcg = 1.0 / np.log2(rank + 1)
        return precision, recall, ndcg
    return 0.0, 0.0, 0.0

def prepare_train_test(interactions):
    # Deduplicate to distinct user-item pairs with mean rating
    ui = (
        interactions.groupby(["customer_unique_id", "product_id"], as_index=False)
        .agg(rating=("review_score", "mean"))
    )

    user_counts = ui["customer_unique_id"].value_counts()
    multi_users = user_counts[user_counts >= 2].index

    # Leave-one-out split for multi-item users (1 held-out test item per user)
    multi_df = ui[ui["customer_unique_id"].isin(multi_users)]
    test_df = multi_df.groupby("customer_unique_id", as_index=False).tail(1)
    train_df = ui.drop(index=test_df.index)

    logging.info(
        f"Dataset Split -> Train pairs: {len(train_df):,} | "
        f"Test held-out pairs: {len(test_df):,} (Multi-item users: {len(multi_users):,})"
    )
    return train_df, test_df

def train_and_evaluate():
    logging.info("=== Starting Task 9: Model Training, Evaluation & MLflow Tracking ===")
    rng = np.random.default_rng(RANDOM_SEED)

    interactions = pd.read_parquet(f"{PROC_DIR}/clean_interactions.parquet")
    item_features = pd.read_parquet(f"{FEAT_DIR}/item_features.parquet")
    train_df, test_df = prepare_train_test(interactions)

    # Index mappings
    all_users = train_df["customer_unique_id"].unique()
    all_items = item_features["product_id"].unique()
    user_to_idx = {u: i for i, u in enumerate(all_users)}
    item_to_idx = {p: i for i, p in enumerate(all_items)}

    train_valid = train_df[train_df["product_id"].isin(item_to_idx)].copy()
    train_valid["u_idx"] = train_valid["customer_unique_id"].map(user_to_idx)
    train_valid["i_idx"] = train_valid["product_id"].map(item_to_idx)

    # Build User-Item Sparse Matrix
    n_users = len(all_users)
    n_items = len(all_items)
    global_mean = float(train_valid["rating"].mean())
    R = csr_matrix(
        (train_valid["rating"].values - global_mean, (train_valid["u_idx"].values, train_valid["i_idx"].values)),
        shape=(n_users, n_items),
        dtype=np.float64,
    )

    # Map seen items per user for candidate sampling
    user_seen = train_valid.groupby("u_idx")["i_idx"].apply(set).to_dict()

    # Select reproducible evaluation subset present in both train & item catalog
    eval_df = test_df[
        test_df["customer_unique_id"].isin(user_to_idx) &
        test_df["product_id"].isin(item_to_idx)
    ].copy()
    if len(eval_df) > EVAL_USERS_LIMIT:
        eval_df = eval_df.sample(n=EVAL_USERS_LIMIT, random_state=RANDOM_SEED)

    # Pre-generate identical candidate pools (1 true + 99 random unseen) for fair comparison
    eval_cases = []
    for _, row in eval_df.iterrows():
        u_idx = user_to_idx[row["customer_unique_id"]]
        true_i = item_to_idx[row["product_id"]]
        seen = user_seen.get(u_idx, set()) | {true_i}
        negs = []
        while len(negs) < NUM_NEG_CANDIDATES:
            cand = int(rng.integers(0, n_items))
            if cand not in seen:
                negs.append(cand)
        eval_cases.append((u_idx, true_i, [true_i] + negs))

    # --- Configure MLflow ---
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("RecoMart_Recommendation_Models")
    results_summary = []

    # ==========================================================
    # MODEL 1: Collaborative Filtering (Matrix Factorization SVD)
    # ==========================================================
    with mlflow.start_run(run_name="Collaborative_Filtering_SVD") as run_svd:
        logging.info(f"Training Collaborative Filtering (SVD, k={N_FACTORS})...")
        U, sigma, Vt = svds(R, k=N_FACTORS)
        U_sigma = U * sigma  # shape: (n_users, k)

        # Item popularity bias to break ties smoothly
        item_pop = np.log1p(item_features["item_purchase_count"].values)
        item_pop_norm = item_pop / (item_pop.max() + 1e-9)

        precs, recs, ndcgs = [], [], []
        for u_idx, true_i, candidates in eval_cases:
            scores = (U_sigma[u_idx] @ Vt[:, candidates]) + 0.15 * item_pop_norm[candidates]
            order = np.argsort(-scores)
            ranked = [candidates[idx] for idx in order]
            p, r, n = ranking_metrics_at_k(ranked, true_i, k=TOP_K)
            precs.append(p); recs.append(r); ndcgs.append(n)

        svd_metrics = {
            "precision_at_10": round(float(np.mean(precs)), 4),
            "recall_at_10": round(float(np.mean(recs)), 4),
            "ndcg_at_10": round(float(np.mean(ndcgs)), 4),
        }

        svd_artifact = f"{MODEL_DIR}/svd_model.pkl"
        with open(svd_artifact, "wb") as f:
            pickle.dump({"U_sigma": U_sigma[:1000], "Vt": Vt, "global_mean": global_mean, "k": N_FACTORS}, f)

        mlflow.log_params({
            "model_type": "Collaborative_Filtering_SVD",
            "n_factors": N_FACTORS,
            "top_k": TOP_K,
            "eval_users": len(eval_cases),
            "candidate_pool_size": NUM_NEG_CANDIDATES + 1,
        })
        mlflow.log_metrics(svd_metrics)
        mlflow.log_artifact(svd_artifact)

        logging.info(f"[SVD] Run ID: {run_svd.info.run_id} | Metrics: {svd_metrics}")
        results_summary.append([
            "Collaborative Filtering (SVD)",
            run_svd.info.run_id[:10] + "...",
            f"k={N_FACTORS}",
            f"{svd_metrics['precision_at_10']:.4f}",
            f"{svd_metrics['recall_at_10']:.4f}",
            f"{svd_metrics['ndcg_at_10']:.4f}",
        ])

    # ==========================================================
    # MODEL 2: Content-Based Filtering (TF-IDF + Item Features)
    # ==========================================================
    with mlflow.start_run(run_name="Content_Based_TFIDF") as run_cb:
        logging.info("Training Content-Based Filtering (Category TF-IDF + Numeric Features)...")
        tfidf = TfidfVectorizer()
        cat_mat = tfidf.fit_transform(item_features["category_english"].fillna("unknown"))
        num_mat = item_features[["price_normalized", "item_avg_rating", "external_popularity_score"]].fillna(0).values
        num_mat = num_mat / (np.linalg.norm(num_mat, axis=0, keepdims=True) + 1e-9)

        item_content_mat = normalize(hstack([cat_mat, csr_matrix(num_mat)]).tocsr())

        precs, recs, ndcgs = [], [], []
        for u_idx, true_i, candidates in eval_cases:
            seen_items = list(user_seen.get(u_idx, []))
            if not seen_items:
                scores = np.zeros(len(candidates))
            else:
                user_profile = np.asarray(item_content_mat[seen_items].mean(axis=0))
                cand_vecs = item_content_mat[candidates]
                scores = (cand_vecs @ user_profile.T).ravel()
            order = np.argsort(-scores)
            ranked = [candidates[idx] for idx in order]
            p, r, n = ranking_metrics_at_k(ranked, true_i, k=TOP_K)
            precs.append(p); recs.append(r); ndcgs.append(n)

        cb_metrics = {
            "precision_at_10": round(float(np.mean(precs)), 4),
            "recall_at_10": round(float(np.mean(recs)), 4),
            "ndcg_at_10": round(float(np.mean(ndcgs)), 4),
        }

        cb_artifact = f"{MODEL_DIR}/content_based_model.pkl"
        with open(cb_artifact, "wb") as f:
            pickle.dump({"vectorizer": tfidf, "n_items": n_items}, f)

        mlflow.log_params({
            "model_type": "Content_Based_TFIDF_Cosine",
            "features": "category_tfidf + price_norm + rating + ext_popularity",
            "top_k": TOP_K,
            "eval_users": len(eval_cases),
            "candidate_pool_size": NUM_NEG_CANDIDATES + 1,
        })
        mlflow.log_metrics(cb_metrics)
        mlflow.log_artifact(cb_artifact)

        logging.info(f"[Content-Based] Run ID: {run_cb.info.run_id} | Metrics: {cb_metrics}")
        results_summary.append([
            "Content-Based (TF-IDF + Metadata)",
            run_cb.info.run_id[:10] + "...",
            "TF-IDF + Cosine",
            f"{cb_metrics['precision_at_10']:.4f}",
            f"{cb_metrics['recall_at_10']:.4f}",
            f"{cb_metrics['ndcg_at_10']:.4f}",
        ])

    # --- Generate Model Performance Report PDF ---
    fig, (ax_tbl, ax_bar) = plt.subplots(2, 1, figsize=(11, 7.5), gridspec_kw={"height_ratios": [1, 1.3]})
    ax_tbl.axis("off")
    ax_tbl.set_title(
        f"RecoMart Model Training & Evaluation Report ({datetime.now().strftime('%Y-%m-%d')})\n"
        f"Evaluated on {len(eval_cases)} Held-Out Multi-Item Users | Top-{TOP_K} Candidate Ranking (100 Items) | Tracked in MLflow",
        fontsize=11.5, fontweight="bold", pad=12
    )

    cols = ["Model", "MLflow Run ID", "Config", "Precision@10", "Recall@10", "NDCG@10"]
    tbl = ax_tbl.table(cellText=results_summary, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.8)
    for (r_idx, c_idx), cell in tbl.get_celld().items():
        if r_idx == 0:
            cell.set_facecolor("#1B365D")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        elif r_idx % 2 == 0:
            cell.set_facecolor("#F2F5F9")

    # Bar chart comparison
    metrics_names = ["Precision@10", "Recall@10", "NDCG@10"]
    x = np.arange(len(metrics_names))
    w = 0.32
    svd_vals = [float(results_summary[0][3]), float(results_summary[0][4]), float(results_summary[0][5])]
    cb_vals = [float(results_summary[1][3]), float(results_summary[1][4]), float(results_summary[1][5])]

    ax_bar.bar(x - w/2, svd_vals, w, label="Collaborative Filtering (SVD)", color="#1B365D")
    ax_bar.bar(x + w/2, cb_vals, w, label="Content-Based (TF-IDF + Metadata)", color="#008080")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(metrics_names, fontweight="bold")
    ax_bar.set_ylabel("Score")
    ax_bar.set_title("Model Ranking Performance Comparison (Top-10)", fontweight="bold")
    ax_bar.legend()
    ax_bar.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    with PdfPages(REPORT_PDF) as pdf:
        pdf.savefig(fig)
    plt.close(fig)

    logging.info(f"Generated Model Performance Report -> {REPORT_PDF}")
    logging.info("=== Task 9 Completed Successfully ===")

if __name__ == "__main__":
    train_and_evaluate()