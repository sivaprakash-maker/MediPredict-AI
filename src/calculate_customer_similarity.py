
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load customer-product matrix
purchase_matrix = pd.read_csv(
    "data/processed/customer_product_matrix.csv",
    index_col="customer_id"
)

print("=" * 60)
print("CUSTOMER SIMILARITY ANALYSIS")
print("=" * 60)

# Calculate cosine similarity between customers
similarity_array = cosine_similarity(purchase_matrix)

# Convert similarity array into DataFrame
similarity_df = pd.DataFrame(
    similarity_array,
    index=purchase_matrix.index,
    columns=purchase_matrix.index
)

# Save similarity results
similarity_df.to_csv(
    "data/processed/customer_similarity.csv"
)

# Select a sample customer
customer_id = 1002

print("\nSimilarity scores for customer:", customer_id)

similar_customers = (
    similarity_df[customer_id]
    .drop(customer_id)
    .sort_values(ascending=False)
    .head(5)
)

print("\nTop 5 similar customers:")
print(similar_customers)

print("\nSimilarity analysis completed!")
print("\nSimilarity matrix shape:", similarity_df.shape)
print("\nFile saved: data/processed/customer_similarity.csv")
