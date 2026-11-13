import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# 1. LOAD PROCESSED DATASET
# ============================================================

df = pd.read_csv("data/processed/cleaned_medical_orders.csv")

print("Processed dataset loaded successfully!")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")


# ============================================================
# 2. BASIC INFORMATION
# ============================================================

print("\nDataset Information:")
print(df.info())

print("\nBasic Statistics:")
print(df.describe())


# ============================================================
# 3. CUSTOMER AGE DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

sns.histplot(
    data=df,
    x="customer_age",
    bins=15,
    kde=True
)

plt.title("Customer Age Distribution")
plt.xlabel("Customer Age")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()

# ============================================================
# 4. ORDERS BY CATEGORY
# ============================================================

category_counts = df["category"].value_counts()

print("\nOrders by Category:")
print(category_counts)

plt.figure(figsize=(10, 6))

sns.barplot(
    x=category_counts.index,
    y=category_counts.values
)

plt.title("Orders by Product Category")
plt.xlabel("Product Category")
plt.ylabel("Number of Orders")

plt.xticks(rotation=30)

plt.tight_layout()
plt.show()


# ============================================================
# 5. REVENUE BY PRODUCT
# ============================================================

revenue_by_product = (
    df.groupby("product_name")["total_amount"]
    .sum()
    .sort_values(ascending=False)
)

print("\nRevenue by Product:")
print(revenue_by_product)

plt.figure(figsize=(12, 6))

sns.barplot(
    x=revenue_by_product.values,
    y=revenue_by_product.index
)

plt.title("Revenue by Product")
plt.xlabel("Total Revenue")
plt.ylabel("Product Name")

plt.tight_layout()
plt.show()


df.groupby("product_name")["total_amount"].sum().sort_values(ascending=False)
# ============================================================
# 6. MONTHLY SALES TREND
# ============================================================

# Convert order_date into datetime format
df["order_date"] = pd.to_datetime(df["order_date"])

# Extract year and month
df["year_month"] = df["order_date"].dt.to_period("M").astype(str)

# Calculate monthly order value
monthly_sales = (
    df.groupby("year_month")["total_amount"]
    .sum()
    .sort_index()
)

print("\nMonthly Order Value:")
print(monthly_sales)

# Plot the trend
plt.figure(figsize=(12, 6))

sns.lineplot(
    x=monthly_sales.index,
    y=monthly_sales.values,
    marker="o"
)

plt.title("Monthly Order Value Trend")
plt.xlabel("Month")
plt.ylabel("Total Order Value")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
df["order_date"].dt.to_period("M").astype(str)
df.groupby("year_month")["total_amount"].sum()


# ============================================================
# 7. ORDER STATUS DISTRIBUTION
# ============================================================

status_counts = df["order_status"].value_counts()

print("\nOrder Status Distribution:")
print(status_counts)

plt.figure(figsize=(8, 6))

sns.barplot(
    x=status_counts.index,
    y=status_counts.values
)

plt.title("Order Status Distribution")
plt.xlabel("Order Status")
plt.ylabel("Number of Orders")

plt.tight_layout()
plt.show()

# Calculate percentages
status_percentages = (
    df["order_status"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nOrder Status Percentages:")
print(status_percentages)

df["order_status"].value_counts()
df["order_status"].value_counts(normalize=True) * 100

# ============================================================
# 8. CORRELATION ANALYSIS
# ============================================================

# Select numerical columns relevant to the analysis
numerical_columns = [
    "customer_age",
    "quantity",
    "discount",
    "inventory_before",
    "unit_price",
    "total_amount",
    "gross_amount",
    "discount_amount"
]

correlation_matrix = df[numerical_columns].corr()

print("\nCorrelation Matrix:")
print(correlation_matrix.round(2))

# Visualize correlations
plt.figure(figsize=(10, 8))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    vmin=-1,
    vmax=1
)

plt.title("Correlation Heatmap")

plt.tight_layout()
plt.show()



# ============================================================
# 9. OUTLIER VISUALIZATION
# ============================================================

print("\nOutlier Visualization")

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df,
    x="total_amount"
)

plt.title("Boxplot of Total Order Amount")
plt.xlabel("Total Order Amount")

plt.tight_layout()
plt.show()

# Display the IQR boundaries
Q1 = df["total_amount"].quantile(0.25)
Q3 = df["total_amount"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[
    (df["total_amount"] < lower_bound)
    | (df["total_amount"] > upper_bound)
]

print(f"\nLower IQR boundary: {lower_bound:.2f}")
print(f"Upper IQR boundary: {upper_bound:.2f}")
print(f"Number of outliers: {len(outliers)}")

print("\nTop 10 highest-value orders:")
print(
    outliers[
        ["product_name", "quantity", "unit_price", "discount", "total_amount"]
    ]
    .sort_values("total_amount", ascending=False)
    .head(10)
)
