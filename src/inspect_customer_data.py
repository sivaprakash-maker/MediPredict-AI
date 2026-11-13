
import pandas as pd

file_path = "data/processed/cleaned_medical_orders.csv"

df = pd.read_csv(file_path)

print("=" * 50)
print("CUSTOMER SEGMENTATION - DATA INSPECTION")
print("=" * 50)

print("\nDataset shape:")
print(df.shape)

print("\nAvailable columns:")
print(df.columns.tolist())

print("\nFirst 5 records:")
print(df.head())

print("\nCustomer ID check:")
if "customer_id" in df.columns:
    print("Unique customers:", df["customer_id"].nunique())
    print("Missing customer IDs:", df["customer_id"].isna().sum())
else:
    print("customer_id column not found!")

print("\nMissing values in important columns:")
for col in [
    "customer_id",
    "quantity",
    "total_amount",
    "order_status"
]:
    if col in df.columns:
        print(f"{col}: {df[col].isna().sum()}")

print("\nData inspection completed.")
