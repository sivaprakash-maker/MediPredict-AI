
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load customer-level behavioral features
df = pd.read_csv(
    "data/processed/customer_segmentation_data.csv"
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

X = df[feature_columns]

# Fit and save the scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

joblib.dump(
    scaler,
    "models/customer_segmentation_scaler.pkl"
)

# Fit and save the K-Means model
kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_scaled)

joblib.dump(
    kmeans,
    "models/customer_kmeans.pkl"
)

print("Customer segmentation model saved!")
print("Scaler: models/customer_segmentation_scaler.pkl")
print("Model: models/customer_kmeans.pkl")

print("\nCluster counts:")
print(pd.Series(clusters).value_counts().sort_index())
