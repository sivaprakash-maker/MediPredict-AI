
import pandas as pd
from sklearn.model_selection import train_test_split

# Load synthetic insurance claims
df = pd.read_csv("data/raw/insurance_claims.csv")

print("Dataset shape:", df.shape)

# Check missing values and duplicates
print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

# Separate features and target
# claim_id is an identifier, not a predictive feature
target = "review_flag"

feature_columns = [
    "customer_age",
    "policy_type",
    "policy_tenure_years",
    "claim_amount",
    "documents_complete",
    "previous_claims"
]

X = df[feature_columns]
y = df[target]

# Convert policy_type into numeric columns
X = pd.get_dummies(
    X,
    columns=["policy_type"],
    dtype=int
)

# Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nFeatures after encoding:")
print(X.columns.tolist())

print("\nTraining feature shape:", X_train.shape)
print("Testing feature shape:", X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())

# Save the prepared datasets
X_train.to_csv(
    "data/processed/insurance_X_train.csv",
    index=False
)

X_test.to_csv(
    "data/processed/insurance_X_test.csv",
    index=False
)

y_train.to_frame().to_csv(
    "data/processed/insurance_y_train.csv",
    index=False
)

y_test.to_frame().to_csv(
    "data/processed/insurance_y_test.csv",
    index=False
)

print("\nPrepared insurance datasets saved successfully!")
