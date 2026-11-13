
import pandas as pd

# Load the cleaned order dataset
input_path = "data/processed/cleaned_medical_orders.csv"
output_path = "data/processed/customer_segmentation_data.csv"

df = pd.read_csv(input_path)

# Validate required columns
required_columns = [
    "customer_id",
    "order_id",
    "total_amount",
    "quantity",
    "discount",
    "product_id",
    "order_status"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

# Aggregate order data into one row per customer
customer_df = df.groupby("customer_id").agg(
    order_count=("order_id", "nunique"),
    total_spend=("total_amount", "sum"),
    avg_order_value=("total_amount", "mean"),
    total_quantity=("quantity", "sum"),
    avg_discount=("discount", "mean"),
    unique_products=("product_id", "nunique"),
    cancelled_orders=(
        "order_status",
        lambda x: (x == "Cancelled").sum()
    ),
    returned_orders=(
        "order_status",
        lambda x: (x == "Returned").sum()
    )
).reset_index()

# Save customer-level data
customer_df.to_csv(output_path, index=False)

print("=" * 50)
print("CUSTOMER SEGMENTATION DATA PREPARED")
print("=" * 50)

print("\nCustomer-level dataset shape:")
print(customer_df.shape)

print("\nFirst 10 customers:")
print(customer_df.head(10).to_string(index=False))

print("\nSummary statistics:")
print(customer_df.drop(columns=["customer_id"]).describe().round(2))

print("\nMissing values:")
print(customer_df.isnull().sum())

print("\nUnique customers:", customer_df["customer_id"].nunique())
print("\nSaved to:", output_path)
