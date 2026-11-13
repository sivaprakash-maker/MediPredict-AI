
import pandas as pd

# Load purchase matrix and similarity matrix
purchase_matrix = pd.read_csv(
    "data/processed/customer_product_matrix.csv",
    index_col="customer_id"
)

similarity_df = pd.read_csv(
    "data/processed/customer_similarity.csv",
    index_col="customer_id"
)

# Convert customer IDs from strings to integers
similarity_df.index = similarity_df.index.astype(int)
similarity_df.columns = similarity_df.columns.astype(int)

# Function to recommend products
def recommend_products(customer_id, top_n=5, similar_users=5):

    if customer_id not in purchase_matrix.index:
        print("Customer not found in purchase history.")
        return

    # Find most similar customers
    similar_customers = (
        similarity_df[customer_id]
        .drop(customer_id)
        .sort_values(ascending=False)
        .head(similar_users)
    )

    # Get products already purchased by target customer
    purchased_products = set(
        purchase_matrix.columns[
            purchase_matrix.loc[customer_id] > 0
        ]
    )

    # Calculate recommendation scores
    recommendation_scores = {}

    for other_customer, similarity_score in similar_customers.items():

        other_products = purchase_matrix.loc[other_customer]

        for product_id, purchased in other_products.items():

            if purchased > 0 and product_id not in purchased_products:

                recommendation_scores[product_id] = (
                    recommendation_scores.get(product_id, 0)
                    + similarity_score
                )

    # Sort recommended products
    recommendations = sorted(
        recommendation_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]

    print("\nCustomer ID:", customer_id)
    print("\nPreviously purchased products:", purchased_products)

    print("\nRecommended products:")

    if not recommendations:
        print("No new recommendations available.")
    else:
        for product_id, score in recommendations:
            print(
                f"Product: {product_id} | "
                f"Recommendation Score: {score:.4f}"
            )


# Test recommendation engine
recommend_products(customer_id=1002, top_n=5)

print("\nProduct recommendation completed!")
