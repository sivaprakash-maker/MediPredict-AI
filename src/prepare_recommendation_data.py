
import pandas as pd

# Load processed order dataset
df = pd.read_csv(
    "data/processed/cleaned_medical_orders.csv"
)

print("=" * 60)
print("PREPARING PRODUCT RECOMMENDATION DATA")
print("=" * 60)

# Keep only successfully completed orders
completed_orders = df[
    df["order_status"] == "Completed"
].copy()

print("\nCompleted orders:", len(completed_orders))

# Aggregate customer-product purchases
purchase_history = (
    completed_orders
    .groupby(["customer_id", "product_id"])
    .agg(
        purchase_count=("order_id", "nunique"),
        total_quantity=("quantity", "sum")
    )
    .reset_index()
)

# Create customer-product interaction matrix
purchase_matrix = purchase_history.pivot_table(
    index="customer_id",
    columns="product_id",
    values="purchase_count",
    fill_value=0
)

# Convert purchase counts into binary interactions
purchase_matrix = (purchase_matrix > 0).astype(int)

# Save both datasets
purchase_history.to_csv(
    "data/processed/customer_product_history.csv",
    index=False
)

purchase_matrix.to_csv(
    "data/processed/customer_product_matrix.csv"
)

print("\nPurchase history shape:", purchase_history.shape)
print("Purchase matrix shape:", purchase_matrix.shape)

print("\nCustomer-product matrix:")
print(purchase_matrix.head(10))

print("\nTotal successful customer-product interactions:",
      int(purchase_matrix.values.sum()))

print("\nFiles saved successfully!")
