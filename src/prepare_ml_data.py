
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the processed dataset
df = pd.read_csv("data/processed/cleaned_medical_orders.csv")

print("Dataset loaded successfully!")
print(f"Original shape: {df.shape}")

# Target: the quantity we want to predict
target = "quantity"

# Select features available when an order is being estimated.
# Exclude order IDs, customer IDs, and values calculated
# using the actual quantity or final order outcome.
feature_columns = [
    "product_id",
    "category",
    "unit_price",
    "discount",
    "customer_age",
    "city",
    "payment_method",
    "inventory_before",
    "month",
    "day_of_week",
    "is_weekend"
]

X = df[feature_columns]
y = df[target]

print("\nInput feature shape:", X.shape)
print("Target shape:", y.shape)

print("\nTarget distribution:")
print(y.describe())

# Convert categorical features into numerical columns
X = pd.get_dummies(
    X,
    columns=[
        "product_id",
        "category",
        "city",
        "payment_method"
    ],
    dtype=int
)

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nML dataset preparation completed!")
print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)
print("Training target:", y_train.shape)
print("Testing target:", y_test.shape)
print("\nFeatures used by the model:")
print(X.columns.tolist())
