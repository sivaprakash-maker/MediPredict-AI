
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load processed dataset
df = pd.read_csv(
    "data/processed/cleaned_medical_orders.csv"
)

# Select features and target
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
y = df["quantity"]

# Convert categorical features to numeric columns
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

# Same train/test split as our previous experiment
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Train Random Forest
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    min_samples_leaf=5,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Display the most important features
importance_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 Important Features")
print("=========================")
print(
    importance_df.head(10).to_string(
        index=False
    )
)
