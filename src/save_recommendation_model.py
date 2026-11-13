
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# Load customer-product purchase matrix
purchase_matrix = pd.read_csv(
    "data/processed/customer_product_matrix.csv",
    index_col="customer_id"
)

# Calculate final customer similarity
similarity_array = cosine_similarity(purchase_matrix)

similarity_df = pd.DataFrame(
    similarity_array,
    index=purchase_matrix.index,
    columns=purchase_matrix.index
)

# Save recommendation artifacts
joblib.dump(
    purchase_matrix,
    "models/customer_product_matrix.pkl"
)

joblib.dump(
    similarity_df,
    "models/customer_similarity.pkl"
)

print("=" * 60)
print("RECOMMENDATION MODEL SAVED")
print("=" * 60)

print("\nCustomer count:", len(purchase_matrix))
print("Product count:", len(purchase_matrix.columns))

print("\nSaved files:")
print("models/customer_product_matrix.pkl")
print("models/customer_similarity.pkl")

print("\nRecommendation model saved successfully!")
