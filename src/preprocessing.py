import pandas as pd


# ============================================================
# 1. LOAD RAW DATA
# ============================================================

input_file = "data/raw/medical_ecommerce_orders.csv"

df = pd.read_csv(input_file)

print("Raw dataset loaded successfully!")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")


# ============================================================
# 2. CHECK DATA TYPES
# ============================================================

print("\nData types before preprocessing:")
print(df.dtypes)


# ============================================================
# 3. CHECK MISSING VALUES
# ============================================================

print("\nMissing values:")
print(df.isnull().sum())


# ============================================================
# 4. CHECK DUPLICATES
# ============================================================

print("\nDuplicate rows:")
print(df.duplicated().sum())

# ============================================================
# 5. CONVERT ORDER DATE
# ============================================================

df["order_date"] = pd.to_datetime(df["order_date"])

print("\nData type after converting order_date:")
print(df["order_date"].dtype)

# ============================================================
# 6. VALIDATE NUMERICAL VALUES
# ============================================================

print("\nInvalid customer ages:")
print(df[(df["customer_age"] < 18) | (df["customer_age"] > 100)])

print("\nInvalid quantities:")
print(df[df["quantity"] <= 0])

print("\nInvalid discounts:")
print(df[(df["discount"] < 0) | (df["discount"] > 100)])

print("\nInvalid unit prices:")
print(df[df["unit_price"] <= 0])

print("\nInvalid inventory values:")
print(df[df["inventory_before"] < 0])

# ============================================================
# 7. CHECK DUPLICATE ORDER IDS
# ============================================================

duplicate_order_ids = df[df["order_id"].duplicated()]

print("\nDuplicate order IDs:")
print(duplicate_order_ids)

print(
    f"Number of duplicate order IDs: "
    f"{df['order_id'].duplicated().sum()}"
)

# ============================================================
# 8. OUTLIER DETECTION USING IQR
# ============================================================

Q1 = df["total_amount"].quantile(0.25)
Q3 = df["total_amount"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print("\nTotal Amount Outlier Analysis")
print(f"Q1: {Q1}")
print(f"Q3: {Q3}")
print(f"IQR: {IQR}")
print(f"Lower Bound: {lower_bound}")
print(f"Upper Bound: {upper_bound}")

outliers = df[
    (df["total_amount"] < lower_bound)
    | (df["total_amount"] > upper_bound)
]

print(f"Number of total_amount outliers: {len(outliers)}")

# ============================================================
# 9. CHECK CATEGORICAL VALUES
# ============================================================

categorical_columns = [
    "gender",
    "city",
    "payment_method",
    "order_status",
    "category"
]

for column in categorical_columns:
    print(f"\nUnique values in {column}:")
    print(df[column].unique())

print("\nOrder Status Distribution:")
print(df["order_status"].value_counts())

print("\nPayment Method Distribution:")
print(df["payment_method"].value_counts())

print("\nCity Distribution:")
print(df["city"].value_counts())

print("\nCategory Distribution:")
print(df["category"].value_counts())

df["year"] = df["order_date"].dt.year
df["month"] = df["order_date"].dt.month
df["day"] = df["order_date"].dt.day
df["day_of_week"] = df["order_date"].dt.dayofweek

df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

df["gross_amount"] = (
    df["unit_price"] * df["quantity"]
)

df["discount_amount"] = (
    df["gross_amount"] * df["discount"] / 100
)

df["calculated_total"] = (
    df["gross_amount"] - df["discount_amount"]
)


df["is_high_value_order"] = (
    df["total_amount"] > upper_bound
).astype(int)

df["amount_difference"] = (
    df["total_amount"] - df["calculated_total"]
)

print("\nMaximum amount calculation difference:")
print(df["amount_difference"].abs().max())

print("\nHigh-value order distribution:")
print(df["is_high_value_order"].value_counts())
output_file = "data/processed/cleaned_medical_orders.csv"

df.to_csv(output_file, index=False)

print("\nProcessed dataset saved successfully!")
print(f"Saved to: {output_file}")

print("\nSample total_amount outliers:")

print(
    outliers[
        [
            "order_id",
            "product_name",
            "category",
            "quantity",
            "discount",
            "unit_price",
            "total_amount",
            "order_status"
        ]
    ].sort_values(
        "total_amount",
        ascending=False
    ).head(20)
)

print(f"Number of total_amount outliers: {len(outliers)}")

# ============================================================
# 10. FEATURE ENGINEERING - DATE FEATURES
# ============================================================

df["year"] = df["order_date"].dt.year
df["month"] = df["order_date"].dt.month
df["day"] = df["order_date"].dt.day
df["day_of_week"] = df["order_date"].dt.dayofweek

df["is_weekend"] = (
    df["day_of_week"].isin([5, 6])
).astype(int)

# ============================================================
# 11. SAVE PROCESSED DATA
# ============================================================

output_file = "data/processed/cleaned_medical_orders.csv"

df.to_csv(output_file, index=False)

print("\nProcessed dataset saved successfully!")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")
print(f"Saved to: {output_file}")

print("\nFinal columns:")
print(df.columns.tolist())

print("\nSample processed data:")
print(
    df[
        [
            "order_date",
            "year",
            "month",
            "day_of_week",
            "is_weekend",
            "unit_price",
            "quantity",
            "gross_amount",
            "discount",
            "discount_amount",
            "total_amount",
            "is_high_value_order"
        ]
    ].head(10)
)