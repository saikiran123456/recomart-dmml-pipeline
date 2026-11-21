CREATE TABLE IF NOT EXISTS dim_users (
    customer_unique_id VARCHAR(64) PRIMARY KEY,
    customer_state VARCHAR(10),
    state_encoded INT,
    user_order_count INT,
    user_total_spend DOUBLE PRECISION,
    user_avg_rating DOUBLE PRECISION,
    user_click_events INT,
    updated_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS dim_items (
    product_id VARCHAR(64) PRIMARY KEY,
    category_english VARCHAR(100),
    category_encoded INT,
    avg_price DOUBLE PRECISION,
    price_normalized DOUBLE PRECISION,
    item_purchase_count INT,
    item_avg_rating DOUBLE PRECISION,
    external_popularity_score DOUBLE PRECISION,
    external_sentiment_score DOUBLE PRECISION,
    updated_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS fact_interactions (
    interaction_id SERIAL PRIMARY KEY,
    customer_unique_id VARCHAR(64),
    product_id VARCHAR(64),
    order_id VARCHAR(64),
    review_score INT,
    price DOUBLE PRECISION,
    price_normalized DOUBLE PRECISION,
    timestamp_normalized DOUBLE PRECISION,
    event_timestamp TIMESTAMP
);

CREATE TABLE IF NOT EXISTS item_co_occurrence (
    product_id_a VARCHAR(64),
    product_id_b VARCHAR(64),
    co_occurrence_count INT,
    similarity_score DOUBLE PRECISION,
    PRIMARY KEY (product_id_a, product_id_b)
);