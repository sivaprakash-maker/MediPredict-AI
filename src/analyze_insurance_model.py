
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load prepared training and testing datasets
X_train = pd.read_csv("data/processed/insurance_X_train.csv")
X_test = pd.read_csv("data/processed/insurance_X_test.csv")

y_train = pd.read_csv("data/processed/insurance_y_train.csv")["review_flag"]
y_test = pd.read_csv("data/processed/insurance_y_test.csv")["review_flag"]

# Train the Random Forest model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, "models/insurance_random_forest.pkl")

print("Model saved successfully!")
print("Location: models/insurance_random_forest.pkl")

# Evaluate the saved model's current in-memory counterpart
predictions = model.predict(X_test)

print("\nTest Accuracy:", round(accuracy_score(y_test, predictions), 4))
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=["Routine (0)", "Review Flag (1)"],
        zero_division=0
    )
)

# Analyze feature importance
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
print(importance_df.head(10).to_string(index=False))

# Save feature importance results
importance_df.to_csv(
    "data/processed/insurance_feature_importance.csv",
    index=False
)

print("\nFeature importance saved successfully!")
