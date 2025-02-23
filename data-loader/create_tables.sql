-- Create schema first
CREATE SCHEMA IF NOT EXISTS iceberg.iceberg WITH (location = 's3://iceberg')

-- Create customers table
CREATE TABLE IF NOT EXISTS iceberg.iceberg.customers (
    customer_id INT,
    name VARCHAR,
    email VARCHAR,
    phone VARCHAR,
    address VARCHAR,
    city VARCHAR,
    region VARCHAR,
    signup_date DATE,
    loyalty_tier VARCHAR
) WITH (
    format = 'PARQUET',
    partitioning = ARRAY['region'],
    location = 's3://iceberg/customers'
)

-- Create products table
CREATE TABLE IF NOT EXISTS iceberg.iceberg.products (
    product_id INT,
    name VARCHAR,
    category VARCHAR,
    subcategory VARCHAR,
    price DECIMAL(10,2),
    cost DECIMAL(10,2),
    in_stock BOOLEAN,
    min_stock INT,
    max_stock INT,
    supplier VARCHAR
) WITH (
    format = 'PARQUET',
    partitioning = ARRAY['category'],
    location = 's3://iceberg/products'
)

-- Create sales table
CREATE TABLE IF NOT EXISTS iceberg.iceberg.sales (
    order_id INT,
    order_date DATE,
    customer_id INT,
    product_id INT,
    quantity INT,
    unit_price DECIMAL(10,2),
    gross_amount DECIMAL(10,2),
    discount DECIMAL(10,2),
    net_amount DECIMAL(10,2),
    region VARCHAR,
    payment_method VARCHAR
) WITH (
    format = 'PARQUET',
    partitioning = ARRAY['region', 'payment_method'],
    location = 's3://iceberg/sales'
) 