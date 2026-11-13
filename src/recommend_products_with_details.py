
import pandas as pd

purchase_matrix = pd.read_csv(
    "data/processed/customer_product_matrix.csv",
    index_col="customer_id"
)

similarity_df = pd.read_csv(
    "data/processed/customer_similarity.csv",
    index_col="customer_id"
)

similarity_df.index = similarity_df.index.astype(int)
similarity_df.columns = similarity_df.columns.astype(int)

# Load product details
orders = pd.read_csv(
    "data/processed/cleaned_medical_orders.csv"
)

product_details = (
    orders[["product_id", "product_name", "category", "unit_price"]]
    .drop_duplicates(subset="product_id")
    .set_index("product_id")
)

def recommend_products(customer_id, top_n=5, similar_users=5):

    if customer_id not in purchase_matrix.index:
        print("Customer not found!")
        return

    similar_customers = (
        similarity_df.loc[customer_id]
        .drop(customer_id)
        .sort_values(ascending=False)
        .head(similar_users)
    )

    purchased_products = set(
        purchase_matrix.columns[
            purchase_matrix.loc[customer_id] > 0
        ]
    )

    scores = {}

    for other_customer, similarity_score in similar_customers.items():

        other_products = purchase_matrix.loc[other_customer]

        for product_id, purchased in other_products.items():

            if purchased > 0 and product_id not in purchased_products:

                scores[product_id] = (
                    scores.get(product_id, 0) + similarity_score
                )

    recommendations = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]

    print("\nCustomer ID:", customer_id)
    print("\nRecommended Products:")

    for product_id, score in recommendations:

        details = product_details.loc[product_id]

        print("\nProduct ID:", product_id)
        print("Product Name:", details["product_name"])
        print("Category:", details["category"])
        print("Price: Rs.", details["unit_price"])
        print("Recommendation Score:", round(score, 4))


recommend_products(1002)

print("\nRecommendation completed successfully!")
