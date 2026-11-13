
import pandas as pd
import random
from sklearn.metrics.pairwise import cosine_similarity

purchase_matrix = pd.read_csv(
    "data/processed/customer_product_matrix.csv",
    index_col="customer_id"
)

random.seed(42)

def recommend_products(customer_id, matrix, top_n=3):

    similarity_array = cosine_similarity(matrix)

    similarity_df = pd.DataFrame(
        similarity_array,
        index=matrix.index,
        columns=matrix.index
    )

    similar_customers = (
        similarity_df.loc[customer_id]
        .drop(customer_id)
        .sort_values(ascending=False)
        .head(5)
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

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [product for product, score in ranked[:top_n]]


hits = 0
total_cases = 0
precision_sum = 0

for customer_id in purchase_matrix.index:

    products = list(
        purchase_matrix.columns[
            purchase_matrix.loc[customer_id] > 0
        ]
    )

    if len(products) < 2:
        continue

    hidden_product = random.choice(products)

    test_matrix = purchase_matrix.copy()
    test_matrix.loc[customer_id, hidden_product] = 0

    recommendations = recommend_products(
        customer_id,
        test_matrix,
        top_n=3
    )

    hit = int(hidden_product in recommendations)

    hits += hit
    total_cases += 1
    precision_sum += hit / 3

if total_cases > 0:

    print("=" * 60)
    print("CORRECTED RECOMMENDATION EVALUATION")
    print("=" * 60)

    print("\nTotal test cases:", total_cases)
    print("Successful recommendations:", hits)

    print(
        "\nHit Rate@3:",
        round(hits / total_cases, 4)
    )

    print(
        "Precision@3:",
        round(precision_sum / total_cases, 4)
    )

    print("\nEvaluation completed!")

else:
    print("No eligible test cases found.")
