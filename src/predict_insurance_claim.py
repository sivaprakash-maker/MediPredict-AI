
import pandas as pd
import joblib

# Load the saved model
model = joblib.load("models/insurance_random_forest.pkl")

# Example synthetic insurance claim
new_claim = pd.DataFrame([{
    "customer_age": 45,
    "policy_tenure_years": 3,
    "claim_amount": 42000,
    "documents_complete": 0,
    "previous_claims": 3,
    "policy_type_Basic": 0,
    "policy_type_Premium": 0,
    "policy_type_Standard": 1
}])

# Match the exact feature order used during training
new_claim = new_claim[model.feature_names_in_]

prediction = model.predict(new_claim)[0]
probabilities = model.predict_proba(new_claim)[0]

print("Predicted class:", prediction)
print("Probability of Routine:", round(probabilities[0], 4))
print("Probability of Review Flag:", round(probabilities[1], 4))

if prediction == 1:
    print("Result: Flag for human review")
else:
    print("Result: Routine classification")

print("\nNote: This is a synthetic demonstration, not a real claim decision.")
