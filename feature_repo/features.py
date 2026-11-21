from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float64, Int64, String
from feast.value_type import ValueType

# --- Entities ---
customer = Entity(
    name="customer",
    join_keys=["customer_unique_id"],
    value_type=ValueType.STRING,
    description="Unique RecoMart shopper identifier",
)

product = Entity(
    name="product",
    join_keys=["product_id"],
    value_type=ValueType.STRING,
    description="Unique RecoMart product catalog identifier",
)

# --- Offline File Sources ---
user_source = FileSource(
    name="user_features_source",
    path="../data/features/user_features.parquet",
    timestamp_field="updated_at",
)

item_source = FileSource(
    name="item_features_source",
    path="../data/features/item_features.parquet",
    timestamp_field="updated_at",
)

# --- Feature Views ---
user_stats_fv = FeatureView(
    name="user_stats_features",
    entities=[customer],
    ttl=timedelta(days=30),
    schema=[
        Field(name="state_encoded", dtype=Int64),
        Field(name="user_order_count", dtype=Int64),
        Field(name="user_total_spend", dtype=Float64),
        Field(name="user_avg_rating", dtype=Float64),
        Field(name="user_click_events", dtype=Int64),
    ],
    online=True,
    source=user_source,
)

item_stats_fv = FeatureView(
    name="item_stats_features",
    entities=[product],
    ttl=timedelta(days=30),
    schema=[
        Field(name="category_encoded", dtype=Int64),
        Field(name="avg_price", dtype=Float64),
        Field(name="price_normalized", dtype=Float64),
        Field(name="item_purchase_count", dtype=Int64),
        Field(name="item_avg_rating", dtype=Float64),
        Field(name="external_popularity_score", dtype=Float64),
        Field(name="external_sentiment_score", dtype=Float64),
    ],
    online=True,
    source=item_source,
)