from faker import Faker
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Initialize Faker
fake = Faker()
Faker.seed(42)  # For reproducibility

def generate_customers(n=1000):
    customers = []
    regions = ['North', 'South', 'East', 'West']
    
    for _ in range(n):
        region = np.random.choice(regions)
        customers.append({
            'customer_id': _ + 1,
            'name': fake.name(),
            'email': fake.email(),
            'phone': fake.phone_number(),
            'address': fake.street_address(),
            'city': fake.city(),
            'region': region,
            'signup_date': fake.date_between(start_date='-2y', end_date='today'),
            'loyalty_tier': np.random.choice(['Bronze', 'Silver', 'Gold', 'Platinum'], p=[0.4, 0.3, 0.2, 0.1])
        })
    
    return pd.DataFrame(customers)

def generate_products(n=200):
    categories = [
        ('Electronics', 100, 2000),
        ('Clothing', 20, 200),
        ('Books', 10, 100),
        ('Home & Garden', 30, 500),
        ('Sports', 25, 400),
        ('Beauty', 15, 150)
    ]
    
    products = []
    for _ in range(n):
        idx = np.random.randint(0, len(categories))
        category, min_price, max_price = categories[idx]
        products.append({
            'product_id': _ + 1,
            'name': fake.catch_phrase(),
            'category': category,
            'subcategory': fake.word(),
            'price': round(np.random.uniform(min_price, max_price), 2),
            'cost': 0,  # Will be calculated as percentage of price
            'in_stock': np.random.choice([True, False], p=[0.8, 0.2]),
            'min_stock': np.random.randint(10, 50),
            'max_stock': np.random.randint(100, 200),
            'supplier': fake.company()
        })
    
    df = pd.DataFrame(products)
    # Calculate cost as 60-80% of price
    df['cost'] = round(df['price'] * np.random.uniform(0.6, 0.8, len(df)), 2)
    return df

def generate_sales(customers_df, products_df, n=10000):
    sales = []
    date_range = pd.date_range(start='2023-01-01', end='2024-02-22')
    
    for _ in range(n):
        customer = customers_df.iloc[np.random.randint(0, len(customers_df))]
        product = products_df.iloc[np.random.randint(0, len(products_df))]
        order_date = np.random.choice(date_range)
        
        # Generate between 1 and 5 items per order
        quantity = np.random.randint(1, 6)
        
        # Apply random discount between 0-20%
        discount = round(np.random.uniform(0, 0.2), 2)
        
        # Calculate final amount
        unit_price = product['price']
        gross_amount = unit_price * quantity
        discount_amount = gross_amount * discount
        net_amount = gross_amount - discount_amount
        
        sales.append({
            'order_id': _ + 1,
            'order_date': order_date,
            'customer_id': customer['customer_id'],
            'product_id': product['product_id'],
            'quantity': quantity,
            'unit_price': unit_price,
            'gross_amount': round(gross_amount, 2),
            'discount': round(discount_amount, 2),
            'net_amount': round(net_amount, 2),
            'region': customer['region'],
            'payment_method': np.random.choice(['Credit Card', 'Debit Card', 'PayPal', 'Bank Transfer'], p=[0.4, 0.3, 0.2, 0.1])
        })
    
    return pd.DataFrame(sales)

def main():
    # Create output directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Generate datasets
    print("Generating customers data...")
    customers_df = generate_customers()
    
    print("Generating products data...")
    products_df = generate_products()
    
    print("Generating sales data...")
    sales_df = generate_sales(customers_df, products_df)
    
    # Save to parquet files
    customers_df.to_parquet('data/customers.parquet')
    products_df.to_parquet('data/products.parquet')
    sales_df.to_parquet('data/sales.parquet')
    
    print("Data generation complete!")
    print(f"Generated {len(customers_df)} customers")
    print(f"Generated {len(products_df)} products")
    print(f"Generated {len(sales_df)} sales records")

if __name__ == "__main__":
    main() 