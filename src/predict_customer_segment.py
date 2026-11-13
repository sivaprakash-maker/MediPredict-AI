
import pandas as pd
import joblib

# Load saved artifacts
scaler = joblib.load(
    "models/customer_segmentation_scaler.pkl"
)

kmeans = joblib.load(
    "models/customer_kmeans.pkl"
)

# Example customer profile using synthetic project data
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

new_customer = pd.DataFrame([{
    "order_count": 7,
    "total_spend": 18000,
    "avg_order_value": 2571.43,
    "total_quantity": 35,
    "avg_discount": 10,
    "unique_products": 5,
    "cancelled_orders": 0,
    "returned_orders": 0
}])[feature_columns]

# Apply the same fitted scaler used by the model
new_customer_scaled = scaler.transform(new_customer)

# Predict the cluster
cluster = kmeans.predict(new_customer_scaled)[0]

print("=" * 50)
print("CUSTOMER SEGMENT PREDICTION")
print("=" * 50)

print("\nCustomer profile:")
print(new_customer.to_string(index=False))

print("\nPredicted cluster:", cluster)
print("Prediction completed successfully!")
print("\nNote: Cluster IDs are numeric labels, not rankings.")
