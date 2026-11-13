
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load processed data
df = pd.read_csv("data/processed/cleaned_medical_orders.csv")

# 2. Define input features and target
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

target = "quantity"

X = df[feature_columns]
y = df[target]

# 3. Convert categorical columns into numerical columns
X = pd.get_dummies(
    X,
    columns=["product_id", "category", "city", "payment_method"],
    dtype=int
)

# 4. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# 5. Create and train the model
model = LinearRegression()

model.fit(X_train, y_train)

print("Linear Regression model trained successfully!")

# 6. Predict quantities for the test dataset
y_pred = model.predict(X_test)

# 7. Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation Results")
print("------------------------")
print(f"MAE:  {mae:.4f}")
print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")

# 8. Compare with a simple baseline
baseline_prediction = [y_train.mean()] * len(y_test)

baseline_mae = mean_absolute_error(y_test, baseline_prediction)

print("\nBaseline Comparison")
print("-------------------")
print(f"Baseline MAE: {baseline_mae:.4f}")
print(f"Model MAE:    {mae:.4f}")

if mae < baseline_mae:
    print("The model improved on the baseline MAE.")
else:
    print("The model did not improve on the baseline MAE.")

# 9. Compare actual and predicted values
results = pd.DataFrame({
    "Actual_Quantity": y_test.to_numpy(),
    "Predicted_Quantity": y_pred.round(2)
})

print("\nSample Predictions:")
print(results.head(10).to_string(index=False))
