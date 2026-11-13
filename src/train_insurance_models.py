
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Load prepared datasets
X_train = pd.read_csv(
    "data/processed/insurance_X_train.csv"
)
X_test = pd.read_csv(
    "data/processed/insurance_X_test.csv"
)

y_train = pd.read_csv(
    "data/processed/insurance_y_train.csv"
)["review_flag"]

y_test = pd.read_csv(
    "data/processed/insurance_y_test.csv"
)["review_flag"]

# Define models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
}

results = []

# Train and evaluate each model
for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    results.append({
        "Model": name,
        "Accuracy": round(
            accuracy_score(y_test, predictions), 4
        ),
        "Precision": round(
            precision_score(
                y_test, predictions, zero_division=0
            ), 4
        ),
        "Recall": round(
            recall_score(
                y_test, predictions, zero_division=0
            ), 4
        ),
        "F1 Score": round(
            f1_score(
                y_test, predictions, zero_division=0
            ), 4
        )
    })

    print(f"\n{'=' * 45}")
    print(f"Model: {name}")
    print("=" * 45)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=["Routine (0)", "Review Flag (1)"],
        zero_division=0
    ))

# Majority-class baseline
baseline_predictions = [0] * len(y_test)

results.append({
    "Model": "Majority Baseline",
    "Accuracy": round(
        accuracy_score(y_test, baseline_predictions), 4
    ),
    "Precision": round(
        precision_score(
            y_test, baseline_predictions, zero_division=0
        ), 4
    ),
    "Recall": round(
        recall_score(
            y_test, baseline_predictions, zero_division=0
        ), 4
    ),
    "F1 Score": round(
        f1_score(
            y_test, baseline_predictions, zero_division=0
        ), 4
    )
})

# Compare all models
results_df = pd.DataFrame(results)

print("\n\nClassification Model Comparison")
print("================================")
print(results_df.to_string(index=False))

print("\nFor the review class, higher recall means fewer")
print("flagged claims are missed.")
print("Higher precision means fewer routine claims are")
print("incorrectly flagged.")
