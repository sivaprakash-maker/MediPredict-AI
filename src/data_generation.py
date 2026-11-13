
import pandas as pd
import numpy as np

# Make random results reproducible
np.random.seed(42)

# Number of orders
number_of_orders = 1000

# Product master data
products = [
    ("P001", "Glucometer", "Diabetes Care", 1200),
    ("P002", "Glucose Test Strips", "Diabetes Care", 650),
    ("P003", "Blood Pressure Monitor", "Diagnostics", 1800),
    ("P004", "Digital Thermometer", "Diagnostics", 350),
    ("P005", "Vitamin D Tablets", "Supplements", 450),
    ("P006", "Calcium Tablets", "Supplements", 300),
    ("P007", "Hand Sanitizer", "Personal Care", 180),
    ("P008", "Surgical Mask", "Personal Care", 120),
    ("P009", "Pain Relief Gel", "Pain Relief", 250),
    ("P010", "First Aid Kit", "First Aid", 750)
]

product_df = pd.DataFrame(
    products,
    columns=["product_id", "product_name", "category", "unit_price"]
)

# Generate order dates and input features
order_dates = pd.date_range(
    start="2025-01-01",
    periods=number_of_orders,
    freq="D"
)

product_ids = np.random.choice(
    product_df["product_id"],
    number_of_orders
)

discounts = np.random.choice(
    [0, 5, 10, 15, 20],
    number_of_orders
)

inventory = np.random.randint(
    10, 200,
    number_of_orders
)

# Base demand assumptions for each product
base_demand = {
    "P001": 3,
    "P002": 6,
    "P003": 2,
    "P004": 5,
    "P005": 4,
    "P006": 5,
    "P007": 7,
    "P008": 8,
    "P009": 4,
    "P010": 3
}

# Create learnable demand patterns
base_quantity = np.array([
    base_demand[product_id]
    for product_id in product_ids
])

# Assumption: higher discounts can increase quantity
discount_effect = discounts * 0.12

# Assumption: demand changes slightly by month
months = order_dates.month.to_numpy()

seasonal_effect = np.where(
    np.isin(months, [10, 11, 12]),
    1.0,
    0.0
)

# Random variation to keep the data realistic
random_noise = np.random.normal(
    loc=0,
    scale=1.2,
    size=number_of_orders
)

# Calculate final integer quantity
raw_quantity = (
    base_quantity
    + discount_effect
    + seasonal_effect
    + random_noise
)

quantity = np.rint(raw_quantity).astype(int)

# Keep quantity within valid business limits
quantity = np.maximum(quantity, 1)
quantity = np.minimum(quantity, 9)
quantity = np.minimum(quantity, inventory)

# Generate remaining order details
data = {
    "order_id": range(10001, 10001 + number_of_orders),
    "order_date": order_dates,

    "customer_id": np.random.randint(
        1001, 1201, number_of_orders
    ),

    "customer_age": np.random.randint(
        18, 75, number_of_orders
    ),

    "gender": np.random.choice(
        ["Male", "Female"],
        number_of_orders
    ),

    "city": np.random.choice(
        ["Chennai", "Coimbatore", "Tiruppur",
         "Madurai", "Salem", "Trichy"],
        number_of_orders
    ),

    "product_id": product_ids,
    "quantity": quantity,
    "discount": discounts,

    "payment_method": np.random.choice(
        ["UPI", "Credit Card", "Debit Card", "COD"],
        number_of_orders
    ),

    "inventory_before": inventory,

    "order_status": np.random.choice(
        ["Completed", "Cancelled", "Returned"],
        number_of_orders,
        p=[0.85, 0.10, 0.05]
    )
}

orders_df = pd.DataFrame(data)

# Add product information
orders_df = orders_df.merge(
    product_df,
    on="product_id",
    how="left"
)

# Calculate total order amount
orders_df["total_amount"] = (
    orders_df["unit_price"]
    * orders_df["quantity"]
    * (1 - orders_df["discount"] / 100)
)

# Save dataset
output_file = "data/raw/medical_ecommerce_orders.csv"

orders_df.to_csv(
    output_file,
    index=False
)

print("Dataset created successfully!")
print(f"Rows: {orders_df.shape[0]}")
print(f"Columns: {orders_df.shape[1]}")
print(f"Saved to: {output_file}")

print("\nQuantity Summary:")
print(orders_df["quantity"].describe())

print("\nAverage Quantity by Product:")
print(
    orders_df.groupby("product_name")["quantity"]
    .mean()
    .sort_values(ascending=False)
    .round(2)
)

print("\nAverage Quantity by Discount:")
print(
    orders_df.groupby("discount")["quantity"]
    .mean()
    .round(2)
)
