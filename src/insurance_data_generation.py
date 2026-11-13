
import pandas as pd
import numpy as np

np.random.seed(42)

number_of_claims = 1000

# Generate synthetic insurance claim records
df = pd.DataFrame({
    "claim_id": range(50001, 50001 + number_of_claims),

    "customer_age": np.random.randint(
        18, 75, number_of_claims
    ),

    "policy_type": np.random.choice(
        ["Basic", "Standard", "Premium"],
        number_of_claims,
        p=[0.35, 0.40, 0.25]
    ),

    "policy_tenure_years": np.random.randint(
        1, 16, number_of_claims
    ),

    "claim_amount": np.random.randint(
        500, 50001, number_of_claims
    ),

    "documents_complete": np.random.choice(
        [0, 1],
        number_of_claims,
        p=[0.20, 0.80]
    ),

    "previous_claims": np.random.poisson(
        lam=1.5,
        size=number_of_claims
    )
})

# Create a synthetic review-risk score.
# These weights are illustrative assumptions, not
# validated insurance rules.
risk_score = (
    1.5 * (df["documents_complete"] == 0).astype(int)
    + 0.8 * (df["claim_amount"] > 30000).astype(int)
    + 0.4 * (df["previous_claims"] >= 3).astype(int)
    - 0.3 * (df["policy_tenure_years"] >= 5).astype(int)
    + np.random.normal(0, 0.7, number_of_claims)
)

# Binary target: 1 = flag for review, 0 = routine
df["review_flag"] = (risk_score > 1.0).astype(int)

output_file = "data/raw/insurance_claims.csv"

df.to_csv(output_file, index=False)

print("Synthetic insurance dataset created!")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

print("\nReview Flag Distribution:")
print(df["review_flag"].value_counts())

print("\nReview Flag Percentage:")
print(
    (df["review_flag"].value_counts(normalize=True) * 100)
    .round(2)
)

print("\nFirst 5 Records:")
print(df.head())
