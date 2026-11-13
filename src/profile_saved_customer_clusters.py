
import pandas as pd
import joblib

# Load customer-level data and saved model artifacts
data = pd.read_csv(
    "data/processed/customer_segmentation_data.csv"
)

scaler = joblib.load(
    "models/customer_segmentation_scaler.pkl"
)

kmeans = joblib.load(
    "models/customer_kmeans.pkl"
)

feature_columns = [
    "order_count",
    "total_spend",
    "avg_order_value",
    "total_quantity",
    "avg_discount",
    "unique_products",
    "cancelled_orders",
    "returned_orders"
]

# Assign clusters using the exact saved model
scaled_features = scaler.transform(data[feature_columns])
data["cluster"] = kmeans.predict(scaled_features)

# Summarize the actual profiles for each saved-model cluster
profile = data.groupby("cluster").agg(
    customer_count=("customer_id", "count"),
    avg_orders=("order_count", "mean"),
    avg_total_spend=("total_spend", "mean"),
    avg_order_value=("avg_order_value", "mean"),
    avg_unique_products=("unique_products", "mean"),
    avg_cancelled_orders=("cancelled_orders", "mean"),
    avg_returned_orders=("returned_orders", "mean")
).round(2)

print("=" * 60)
print("SAVED MODEL - CUSTOMER CLUSTER PROFILES")
print("=" * 60)
print(profile.to_string())

# Save results for project documentation
profile.to_csv(
    "data/processed/saved_model_cluster_profiles.csv"
)

print("\nProfiles saved successfully!")
