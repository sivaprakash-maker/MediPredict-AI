
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

input_path = "data/processed/customer_features_scaled.csv"

df = pd.read_csv(input_path)

# Keep IDs for reporting, but exclude them from clustering
customer_ids = df["customer_id"]

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

# Evaluate candidate cluster counts
inertias = []
silhouette_scores = []
k_values = range(2, 9)

print("=" * 55)
print("K-MEANS CUSTOMER CLUSTER EVALUATION")
print("=" * 55)

for k in k_values:
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X)

    inertias.append(model.inertia_)
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)

    print(
        f"k={k} | "
        f"Inertia={model.inertia_:.2f} | "
        f"Silhouette Score={score:.4f}"
    )

# Save evaluation results
evaluation_df = pd.DataFrame({
    "k": list(k_values),
    "inertia": inertias,
    "silhouette_score": silhouette_scores
})

evaluation_df.to_csv(
    "data/processed/kmeans_evaluation.csv",
    index=False
)

# Plot the Elbow Method
plt.figure(figsize=(8, 5))
plt.plot(list(k_values), inertias, marker="o")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Customer Segmentation")
plt.xticks(list(k_values))
plt.grid(True)
plt.tight_layout()
plt.savefig(
    "data/processed/kmeans_elbow.png",
    dpi=150
)
plt.close()

# Train initial candidate model with 4 clusters
final_model = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

cluster_labels = final_model.fit_predict(X)

# Save the cluster assignment for every customer
results_df = pd.DataFrame({
    "customer_id": customer_ids,
    "cluster": cluster_labels
})

results_df.to_csv(
    "data/processed/customer_clusters.csv",
    index=False
)

print("\nInitial model trained with 4 clusters.")
print("\nCustomer count per cluster:")
print(results_df["cluster"].value_counts().sort_index())

print("\nSaved files:")
print("- data/processed/kmeans_evaluation.csv")
print("- data/processed/kmeans_elbow.png")
print("- data/processed/customer_clusters.csv")
