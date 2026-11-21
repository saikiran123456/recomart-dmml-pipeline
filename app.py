import os
import json
import socket
import pickle
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

st.set_page_config(
    page_title="RecoMart ML Data Pipeline & Recommender",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom Styling ---
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .hero-box {
        background: linear-gradient(135deg, #0F2744 0%, #1E4E79 60%, #0D7377 100%);
        padding: 1.4rem 1.8rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.2rem;
    }
    .hero-box h1 { margin: 0; font-size: 1.8rem; color: white; }
    .hero-box p { margin: 0.4rem 0 0 0; opacity: 0.9; font-size: 0.98rem; }
    .status-pill {
        display: inline-block;
        padding: 0.2rem 0.65rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 0.4rem;
    }
    .pill-up { background-color: #DEF7EC; color: #03543F; }
    .pill-off { background-color: #F3F4F6; color: #4B5563; }
    </style>
    """,
    unsafe_allow_html=True,
)


def check_port(host: str, port: int, timeout: float = 0.25) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@st.cache_data(show_spinner=False)
def load_datasets():
    """Load full local Parquet datasets if available, otherwise cloud-ready app_data snapshot."""
    full_paths = (
        "data/processed/clean_interactions.parquet",
        "data/features/user_features.parquet",
        "data/features/item_features.parquet",
        "data/features/item_co_occurrence.parquet",
    )
    sample_paths = (
        "app_data/interactions_sample.parquet",
        "app_data/user_features_sample.parquet",
        "app_data/item_features_sample.parquet",
        "app_data/item_co_occurrence_sample.parquet",
    )

    if all(os.path.exists(p) for p in full_paths):
        mode = "Full Local Data Lake"
        paths = full_paths
    else:
        mode = "Cloud Demo Snapshot"
        paths = sample_paths

    interactions = pd.read_parquet(paths[0])
    users = pd.read_parquet(paths[1])
    items = pd.read_parquet(paths[2])
    co_occ = pd.read_parquet(paths[3])
    return mode, interactions, users, items, co_occ


@st.cache_resource(show_spinner=False)
def build_content_matrix(items_df: pd.DataFrame):
    """Load trained TF-IDF model if Python/sklearn versions match, else fit seamlessly."""
    categories = items_df["category_english"].fillna("unknown")
    tfidf = None
    if os.path.exists("models/content_based_model.pkl"):
        try:
            with open("models/content_based_model.pkl", "rb") as f:
                bundle = pickle.load(f)
            tfidf = bundle["vectorizer"]
            cat_mat = tfidf.transform(categories)
        except Exception:
            tfidf = None

    if tfidf is None:
        tfidf = TfidfVectorizer()
        cat_mat = tfidf.fit_transform(categories)

    num_mat = items_df[["price_normalized", "item_avg_rating", "external_popularity_score"]].fillna(0).values
    num_mat = num_mat / (np.linalg.norm(num_mat, axis=0, keepdims=True) + 1e-9)
    item_content_mat = normalize(hstack([cat_mat, csr_matrix(num_mat)]).tocsr())
    return item_content_mat


def fetch_user_profile(customer_id: str, users_df: pd.DataFrame, redis_online: bool):
    """Try live Feast + Redis online lookup first; fall back to warehouse Parquet."""
    if redis_online and os.path.exists("feature_repo/feature_store.yaml"):
        try:
            from feast import FeatureStore

            store = FeatureStore(repo_path="feature_repo")
            vec = store.get_online_features(
                features=[
                    "user_stats_features:user_order_count",
                    "user_stats_features:user_total_spend",
                    "user_stats_features:user_avg_rating",
                    "user_stats_features:user_click_events",
                ],
                entity_rows=[{"customer_unique_id": customer_id}],
            ).to_dict()
            if vec["user_order_count"][0] is not None:
                return {
                    "source": "Feast + Redis Online Store (Live)",
                    "orders": int(vec["user_order_count"][0]),
                    "spend": float(vec["user_total_spend"][0]),
                    "avg_rating": float(vec["user_avg_rating"][0]),
                    "clicks": int(vec["user_click_events"][0]),
                }
        except Exception:
            pass

    row = users_df[users_df["customer_unique_id"] == customer_id]
    if not row.empty:
        r = row.iloc[0]
        return {
            "source": "Engineered Feature Store (Offline Parquet)",
            "orders": int(r["user_order_count"]),
            "spend": float(r["user_total_spend"]),
            "avg_rating": float(r["user_avg_rating"]),
            "clicks": int(r["user_click_events"]),
        }
    return {"source": "Cold-Start Visitor", "orders": 0, "spend": 0.0, "avg_rating": 0.0, "clicks": 0}


def generate_recommendations(
    customer_id: str,
    interactions_df: pd.DataFrame,
    items_df: pd.DataFrame,
    item_content_mat,
    top_k: int = 10,
    cold_start: bool = False,
    category_filter: str = "All Categories",
):
    bought_ids = set()
    if not cold_start and customer_id:
        bought_ids = set(interactions_df.loc[interactions_df["customer_unique_id"] == customer_id, "product_id"])

    item_to_idx = {pid: idx for idx, pid in enumerate(items_df["product_id"])}
    bought_indices = [item_to_idx[p] for p in bought_ids if p in item_to_idx]

    if bought_indices and not cold_start:
        user_profile = np.asarray(item_content_mat[bought_indices].mean(axis=0))
        scores = (item_content_mat @ user_profile.T).ravel()
        strategy = "Content-Based TF-IDF + Metadata Cosine Similarity"
    else:
        scores = (
            0.6 * items_df["external_popularity_score"].fillna(0).values
            + 0.4 * items_df["item_avg_rating"].fillna(0).values
        )
        scores = scores / (scores.max() + 1e-9)
        strategy = "Cold-Start Popularity + External API Rating Fallback"

    mask = np.ones(len(items_df), dtype=bool)
    for idx in bought_indices:
        mask[idx] = False
    if category_filter != "All Categories":
        mask &= (items_df["category_english"] == category_filter).values

    filtered_scores = np.where(mask, scores, -1.0)
    top_indices = np.argsort(-filtered_scores)[:top_k]
    top_indices = [i for i in top_indices if filtered_scores[i] >= 0]

    recs = items_df.iloc[top_indices][
        ["product_id", "category_english", "avg_price", "item_avg_rating", "item_purchase_count", "external_popularity_score"]
    ].copy()
    recs["match_score"] = np.round(filtered_scores[top_indices], 4)
    recs["avg_price"] = recs["avg_price"].round(2)
    recs["item_avg_rating"] = recs["item_avg_rating"].round(2)
    recs["external_popularity_score"] = recs["external_popularity_score"].round(3)
    return recs.reset_index(drop=True), strategy, bought_ids


# --- Load Data & Check Live Services ---
data_mode, interactions_df, users_df, items_df, co_occ_df = load_datasets()
item_content_mat = build_content_matrix(items_df)

services = {
    "MinIO Data Lake (:9000)": check_port("127.0.0.1", 9000),
    "PostgreSQL DW (:5432)": check_port("127.0.0.1", 5432),
    "Redis Online Store (:6379)": check_port("127.0.0.1", 6379),
    "Prefect Server (:4200)": check_port("127.0.0.1", 4200),
    "MLflow UI (:5000)": check_port("127.0.0.1", 5000),
}

# --- Sidebar ---
with st.sidebar:
    st.markdown("### 🛒 RecoMart Control Panel")
    st.caption(f"Data Mode: **{data_mode}**")
    st.divider()
    st.markdown("**Live Local Stack Status**")
    for svc, is_up in services.items():
        badge = "🟢 Online" if is_up else "⚪ Offline / Cloud"
        st.markdown(f"{badge} — `{svc}`")

    st.divider()
    st.markdown("**Project Metadata**")
    st.markdown(
        "- **Course:** DMML (`S1-26_DSEZG529`)\n"
        "- **Author:** Saikirankumar Dasari\n"
        "- **Orchestrator:** Prefect (`sandy-toucanet`)\n"
        "- **Winning Model:** Content-Based TF-IDF"
    )
    st.link_button("View GitHub Repo", "https://github.com/saikiran123456/recomart-dmml-pipeline", use_container_width=True)

# --- Hero Header ---
st.markdown(
    """
    <div class="hero-box">
        <h1>🛒 RecoMart — End-to-End ML Data Pipeline & Recommendation Engine</h1>
        <p>Automated Ingestion (MinIO) • Great Expectations • PostgreSQL Warehouse • Feast + Redis • DVC Lineage • MLflow • Prefect DAG</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Top KPI Strip ---
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Unique Users (DW)", "95,420")
k2.metric("Catalog Items (DW)", "32,951")
k3.metric("Clean Interactions", "112,650", "99.9964% Sparse")
k4.metric("Best NDCG@10 (TF-IDF)", "0.4510", "+53.1% vs SVD")
k5.metric("Prefect DAG Runtime", "3m 31s", "9 / 9 Tasks Passed")

# --- Main Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🎯 Live Recommender",
        "⚙️ Pipeline & Prefect DAG",
        "📊 Data Quality & EDA",
        "🧪 MLflow, Feast & DVC Lineage",
    ]
)

# ==============================================================================
# TAB 1: LIVE RECOMMENDER
# ==============================================================================
with tab1:
    st.subheader("Personalized Product Recommendation Interface")

    demo_uid = "871766c5855e863f6eccc05f988b23cb"
    top_sample_users = (
        users_df.sort_values(["user_order_count", "user_total_spend"], ascending=False)
        .head(100)["customer_unique_id"]
        .tolist()
    )
    if demo_uid in users_df["customer_unique_id"].values:
        selectable_users = [demo_uid] + [u for u in top_sample_users if u != demo_uid]
    else:
        selectable_users = top_sample_users

    c1, c2, c3, c4 = st.columns([2.6, 1.3, 1.0, 1.1])
    with c1:
        selected_user = st.selectbox("Select Customer (`customer_unique_id`)", selectable_users)
    with c2:
        all_cats = ["All Categories"] + sorted(items_df["category_english"].dropna().unique().tolist())
        cat_filter = st.selectbox("Category Filter", all_cats)
    with c3:
        top_k = st.slider("Top-K Items", min_value=5, max_value=20, value=10)
    with c4:
        st.write("")
        cold_start_mode = st.toggle("Cold-Start Mode", value=False, help="Simulate a brand-new shopper with zero order history")

    # Fetch profile (Live Redis/Feast or Offline Feature Store)
    profile = fetch_user_profile(
        selected_user if not cold_start_mode else "",
        users_df,
        redis_online=services["Redis Online Store (:6379)"],
    )

    st.info(
        f"**Feature Source:** `{profile['source']}` | "
        f"**Orders:** `{profile['orders']}` | "
        f"**Total Spend:** `${profile['spend']:.2f}` | "
        f"**Avg Rating:** `{profile['avg_rating']:.2f} / 5.0` | "
        f"**Click Events:** `{profile['clicks']}`"
    )

    recs_df, strategy_used, bought_ids = generate_recommendations(
        selected_user,
        interactions_df,
        items_df,
        item_content_mat,
        top_k=top_k,
        cold_start=cold_start_mode,
        category_filter=cat_filter,
    )

    left_col, right_col = st.columns([1.8, 1.0])
    with left_col:
        st.markdown(f"**Top-{top_k} Recommendations** — *{strategy_used}*")
        st.dataframe(
            recs_df.rename(
                columns={
                    "product_id": "Product ID",
                    "category_english": "Category",
                    "avg_price": "Avg Price ($)",
                    "item_avg_rating": "Avg Rating",
                    "item_purchase_count": "Purchases",
                    "external_popularity_score": "Ext. Popularity",
                    "match_score": "Match Score",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    with right_col:
        st.markdown("**Customer Purchase History**")
        if bought_ids and not cold_start_mode:
            hist = interactions_df[interactions_df["customer_unique_id"] == selected_user][
                ["product_id", "category_english", "price", "review_score"]
            ].drop_duplicates()
            st.dataframe(hist, use_container_width=True, hide_index=True)
        else:
            st.caption("No prior orders (Cold-Start fallback active).")

        st.markdown("**Frequently Bought Together (Co-Occurrence)**")
        if not co_occ_df.empty:
            top_pairs = co_occ_df.sort_values("co_occurrence_count", ascending=False).head(8)
            st.dataframe(
                top_pairs.rename(
                    columns={
                        "product_id_a": "Item A",
                        "product_id_b": "Item B",
                        "co_occurrence_count": "Co-Purchases",
                        "similarity_score": "Cosine Sim",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

# ==============================================================================
# TAB 2: PIPELINE & PREFECT DAG
# ==============================================================================
with tab2:
    st.subheader("Automated 9-Stage Prefect Orchestration DAG (`sandy-toucanet`)")
    st.caption("Flow: `RecoMart_End_To_End_Data_Pipeline` | Run ID: `bb54bada-2c08-44ac-8c15-2e629544c24e` | Total Duration: **3m 31s**")

    dag_data = pd.DataFrame(
        [
            {"Step": 1, "Task": "1a_Batch_Ingestion", "Duration (s)": 20.0, "Stage": "Ingestion (CSVs + REST APIs -> MinIO)", "Status": "Completed"},
            {"Step": 2, "Task": "1b_Near_Real_Time_Stream", "Duration (s)": 16.1, "Stage": "Stream Micro-Batches -> MinIO + Redis", "Status": "Completed"},
            {"Step": 3, "Task": "2_Data_Validation", "Duration (s)": 24.9, "Stage": "Great Expectations Quality Gate", "Status": "Completed"},
            {"Step": 4, "Task": "3_Data_Preparation_EDA", "Duration (s)": 15.4, "Stage": "Cleaning, Encoding, Normalization & EDA", "Status": "Completed"},
            {"Step": 5, "Task": "4_Feature_Engineering_Warehouse", "Duration (s)": 40.5, "Stage": "PostgreSQL Star Schema Load", "Status": "Completed"},
            {"Step": 6, "Task": "5_Feature_Store_Feast_Redis", "Duration (s)": 31.3, "Stage": "Feast Registry + Redis Materialization", "Status": "Completed"},
            {"Step": 7, "Task": "6_Data_Versioning_DVC_Lineage", "Duration (s)": 12.7, "Stage": "DVC Snapshots + Lineage Docs", "Status": "Completed"},
            {"Step": 8, "Task": "7_Model_Training_MLflow", "Duration (s)": 30.6, "Stage": "SVD & TF-IDF Training + MLflow", "Status": "Completed"},
            {"Step": 9, "Task": "8_Live_Inference_Verification", "Duration (s)": 17.9, "Stage": "Redis Online Lookup + Top-10 Recs", "Status": "Completed"},
        ]
    )

    chart = (
        alt.Chart(dag_data)
        .mark_bar(cornerRadiusTopRight=5, cornerRadiusBottomRight=5)
        .encode(
            x=alt.X("Duration (s):Q", title="Execution Duration (seconds)"),
            y=alt.Y("Task:N", sort=dag_data["Task"].tolist(), title=None),
            color=alt.value("#0D7377"),
            tooltip=["Task", "Stage", "Duration (s)", "Status"],
        )
        .properties(height=340)
    )
    st.altair_chart(chart, use_container_width=True)
    st.dataframe(dag_data, use_container_width=True, hide_index=True)

    st.markdown("**PostgreSQL Warehouse (`recomart_dw`) Table Counts**")
    w1, w2, w3, w4 = st.columns(4)
    w1.metric("dim_users", "95,420 rows")
    w2.metric("dim_items", "32,951 rows")
    w3.metric("fact_interactions", "112,650 rows")
    w4.metric("item_co_occurrence", "4,058 pairs")

# ==============================================================================
# TAB 3: DATA QUALITY & EDA
# ==============================================================================
with tab3:
    st.subheader("Task 4: Automated Data Quality Report (Great Expectations + pandas)")
    dq_df = pd.DataFrame(
        [
            ["clickstream_events", "50,000", 0, 9457, "PASS", "PASS (enum, bad=0)", "WARN -> Cleaned"],
            ["orders", "99,441", 0, 0, "PASS", "N/A", "PASS"],
            ["order_items", "112,650", 0, 0, "PASS", "PASS (>0, bad=0)", "PASS"],
            ["order_reviews", "99,224", 0, 814, "PASS", "PASS (1–5, bad=0)", "WARN -> Cleaned"],
            ["customers", "99,441", 0, 0, "PASS", "N/A", "PASS"],
            ["products", "32,951", 610, 0, "PASS", "N/A", "WARN -> Imputed"],
            ["api_catalog_json", "100", 0, 0, "PASS", "PASS (price > 0)", "PASS"],
        ],
        columns=["Dataset", "Rows", "Missing (Key Cols)", "Duplicates", "Schema", "Range / Format Check", "Gate Status"],
    )
    st.dataframe(dq_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Task 5: Interactive EDA & Summary Visualizations")

    e1, e2 = st.columns(2)
    with e1:
        top_cats = (
            items_df.groupby("category_english", as_index=False)["item_purchase_count"]
            .sum()
            .sort_values("item_purchase_count", ascending=False)
            .head(10)
        )
        cat_chart = (
            alt.Chart(top_cats)
            .mark_bar(color="#1E4E79")
            .encode(
                x=alt.X("item_purchase_count:Q", title="Total Purchases"),
                y=alt.Y("category_english:N", sort="-x", title="Category"),
                tooltip=["category_english", "item_purchase_count"],
            )
            .properties(title="Top 10 Product Categories", height=320)
        )
        st.altair_chart(cat_chart, use_container_width=True)

    with e2:
        rating_dist = (
            interactions_df["review_score"].value_counts().sort_index().reset_index()
        )
        rating_dist.columns = ["Review Score", "Count"]
        rat_chart = (
            alt.Chart(rating_dist)
            .mark_bar(color="#0F2744")
            .encode(
                x=alt.X("Review Score:O", title="Review Score (1–5 Scale)"),
                y=alt.Y("Count:Q", title="Interactions"),
                tooltip=["Review Score", "Count"],
            )
            .properties(title="Rating Distribution", height=320)
        )
        st.altair_chart(rat_chart, use_container_width=True)

    if os.path.exists("reports/05_eda_summary_plots.png"):
        with st.expander("View Full Generated EDA Summary Report (`reports/05_eda_summary_plots.png`)", expanded=False):
            st.image("reports/05_eda_summary_plots.png", use_container_width=True)

# ==============================================================================
# TAB 4: MLFLOW, FEAST & DVC LINEAGE
# ==============================================================================
with tab4:
    st.subheader("Task 9: MLflow Model Evaluation Comparison (500 Held-Out Users, Top-10)")

    mlflow_df = pd.DataFrame(
        [
            {"Model": "Content-Based (TF-IDF + Metadata)", "Metric": "Precision@10", "Score": 0.0610},
            {"Model": "Content-Based (TF-IDF + Metadata)", "Metric": "Recall@10", "Score": 0.6100},
            {"Model": "Content-Based (TF-IDF + Metadata)", "Metric": "NDCG@10", "Score": 0.4510},
            {"Model": "Collaborative Filtering (SVD, k=25)", "Metric": "Precision@10", "Score": 0.0466},
            {"Model": "Collaborative Filtering (SVD, k=25)", "Metric": "Recall@10", "Score": 0.4660},
            {"Model": "Collaborative Filtering (SVD, k=25)", "Metric": "NDCG@10", "Score": 0.2946},
        ]
    )

    m_chart = (
        alt.Chart(mlflow_df)
        .mark_bar()
        .encode(
            x=alt.X("Metric:N", title=None),
            y=alt.Y("Score:Q", title="Score"),
            color=alt.Color("Model:N", scale=alt.Scale(range=["#0D7377", "#1E4E79"])),
            xOffset="Model:N",
            tooltip=["Model", "Metric", "Score"],
        )
        .properties(height=320)
    )
    st.altair_chart(m_chart, use_container_width=True)

    st.success(
        "**Key Finding:** Under **99.9964% matrix sparsity**, Content-Based Filtering (`a416d532...`) achieved "
        "**0.4510 NDCG@10** and **0.6100 Recall@10**, beating Truncated SVD (`c494a210...`) by **+53.1% relative NDCG** "
        "because catalog metadata and FakeStore API popularity carry stronger signal for 1–2 purchase shoppers."
    )

    st.divider()
    col_dvc, col_feast = st.columns(2)
    with col_dvc:
        st.markdown("### Task 8: DVC Dataset Lineage")
        if os.path.exists("data/lineage_metadata.json"):
            with open("data/lineage_metadata.json", "r", encoding="utf-8") as f:
                lineage_df = pd.DataFrame(json.load(f))
            st.dataframe(
                lineage_df[["stage", "path", "nfiles", "size_mb", "md5"]],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("Run `python src/lineage_tracker.py` to populate lineage metadata.")

    with col_feast:
        st.markdown("### Task 7: Feast Feature Store Registry")
        feast_meta = pd.DataFrame(
            [
                ["user_stats_features", "state_encoded, user_order_count, user_total_spend, user_avg_rating, user_click_events", "dim_users -> Redis"],
                ["item_stats_features", "category_encoded, avg_price, price_normalized, item_purchase_count, item_avg_rating, external_popularity_score, external_sentiment_score", "dim_items -> Redis"],
            ],
            columns=["Feature View", "Registered Features", "Serving Path"],
        )
        st.dataframe(feast_meta, use_container_width=True, hide_index=True)