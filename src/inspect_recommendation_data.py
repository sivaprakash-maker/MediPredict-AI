
import pandas as pd

file_path = "data/processed/cleaned_medical_orders.csv"

df = pd.read_csv(file_path)

print("=" * 60)
print("PRODUCT RECOMMENDATION - DATA INSPECTION")
print("=" * 60)

print("\nDataset shape:", df.shape)

print("\nAvailable columns:")
print(df.columns.tolist())

print("\nOrder status distribution:")
print(df["order_status"].value_counts())

print("\nUnique customers:", df["customer_id"].nunique())
print("Unique products:", df["product_id"].nunique())

print("\nCustomer-product purchase history:")
print(
    df.groupby(["customer_id", "product_id"])
      .size()
      .reset_index(name="purchase_count")
      .head(10)
)

print("\nProduct purchase summary:")
print(
    df.groupby(["product_id", "product_name"])
      .agg(
          total_orders=("order_id", "nunique"),
          total_quantity=("quantity", "sum")
      )
      .reset_index()
      .sort_values("total_orders", ascending=False)
)

print("\nData inspection completed!")
