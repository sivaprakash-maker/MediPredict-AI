
import pandas as pd
from sklearn.preprocessing import StandardScaler

input_path = "data/processed/customer_segmentation_data.csv"
output_path = "data/processed/customer_features_scaled.csv"

# Load customer-level dataset
df = pd.read_csv(input_path)

# Keep customer ID separately; don't use it for clustering
customer_ids = df["customer_id"]

# Select behavioral features
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

X = df[feature_columns].copy()

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert scaled data back to a DataFrame
scaled_df = pd.DataFrame(
    X_scaled,
    columns=feature_columns
)

# Add customer ID only for tracking results
scaled_df.insert(0, "customer_id", customer_ids.values)

# Save scaled dataset
scaled_df.to_csv(output_path, index=False)

print("=" * 50)
print("CUSTOMER FEATURES SCALED")
print("=" * 50)

print("\nScaled dataset shape:", scaled_df.shape)

print("\nFirst 5 scaled records:")
print(scaled_df.head().round(3).to_string(index=False))

print("\nFeature means after scaling:")
print(scaled_df[feature_columns].mean().round(3))

print("\nFeature standard deviations after scaling:")
print(scaled_df[feature_columns].std(ddof=0).round(3))

print("\nSaved to:", output_path)
