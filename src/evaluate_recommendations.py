
import pandas as pd
import random

# Load purchase matrix
purchase_matrix = pd.read_csv(
    "data/processed/customer_product_matrix.csv",
    index_col="customer_id"
)

# Load product similarity data
similarity_df = pd.read_csv(
    "data/processed/customer_similarity.csv",
    index_col="customer_id"
)

similarity_df.index = similarity_df.index.astype(int)
similarity_df.columns = similarity_df.columns.astype(int)

random.seed(42)

def recommend_for_customer(
    customer_id,
    matrix,
    similarity,
    top_n=3,
    similar_users=5
):
    similar_customers = (
        similarity.loc[customer_id]
        .drop(customer_id)
        .sort_values(ascending=False)
        .head(similar_users)
    )

    purchased_products = set(
        matrix.columns[matrix.loc[customer_id] > 0]
    )

    scores = {}

    for other_customer, similarity_score in similar_customers.items():

        other_products = matrix.loc[other_customer]

        for product_id, purchased in other_products.items():

            if purchased > 0 and product_id not in purchased_products:
                scores[product_id] = (
                    scores.get(product_id, 0) + similarity_score
                )

    recommendations = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [product for product, score in recommendations[:top_n]]


# Evaluation
hits = 0
total_test_cases = 0
precision_sum = 0
recall_sum = 0

for customer_id in purchase_matrix.index:

    purchased_products = list(
        purchase_matrix.columns[
            purchase_matrix.loc[customer_id] > 0
        ]
    )

    # Need at least two products for a meaningful test
    if len(purchased_products) < 2:
        continue

    # Temporarily hide one purchased product
    hidden_product = random.choice(purchased_products)

    test_matrix = purchase_matrix.copy()
    test_matrix.loc[customer_id, hidden_product] = 0

    # Generate recommendations from remaining purchases
    recommendations = recommend_for_customer(
        customer_id,
        test_matrix,
        similarity_df,
        top_n=3
    )

    # Evaluate whether hidden product was recommended
    hit = int(hidden_product in recommendations)

    hits += hit
    total_test_cases += 1

    precision_sum += hit / 3
    recall_sum += hit / 1

# Display metrics
if total_test_cases > 0:

    hit_rate = hits / total_test_cases
    precision_at_3 = precision_sum / total_test_cases
    recall_at_3 = recall_sum / total_test_cases

    print("=" * 60)
    print("RECOMMENDATION MODEL EVALUATION")
    print("=" * 60)

    print("\nTotal test cases:", total_test_cases)
    print("Successful recommendations:", hits)

    print("\nHit Rate@3:", round(hit_rate, 4))
    print("Precision@3:", round(precision_at_3, 4))
    print("Recall@3:", round(recall_at_3, 4))

    print("\nEvaluation completed successfully!")

else:
    print("No eligible customers found for evaluation.")
