
import pandas as pd

customer_path = "data/processed/customer_segmentation_data.csv"
cluster_path = "data/processed/customer_clusters.csv"

customers = pd.read_csv(customer_path)
clusters = pd.read_csv(cluster_path)

# Join customer behavior with cluster assignments
df = customers.merge(
    clusters,
    on="customer_id",
    how="inner",
    validate="one_to_one"
)

profile_columns = [
    "order_count",
    "total_spend",
    "avg_order_value",
    "total_quantity",
    "avg_discount",
    "unique_products",
    "cancelled_orders",
    "returned_orders"
]

# Show average behavior for each cluster
profiles = df.groupby("cluster")[profile_columns].mean().round(2)

print("=" * 60)
print("CUSTOMER SEGMENT PROFILES — FOUR CLUSTERS")
print("=" * 60)

print("\nCluster sizes:")
print(df["cluster"].value_counts().sort_index())

print("\nAverage feature values per cluster:")
print(profiles.to_string())

print("\nMedian total spend per cluster:")
print(df.groupby("cluster")["total_spend"].median().round(2))

print("\nCustomer segmentation analysis completed.")

# Save results for later reporting
profiles.to_csv(
    "data/processed/customer_segment_profiles.csv"
)

print("\nSaved to: data/processed/customer_segment_profiles.csv")
