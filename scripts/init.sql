CREATE SCHEMA IF NOT EXISTS iceberg.iceberg WITH (
    location = 's3://iceberg/'
);

CREATE TABLE IF NOT EXISTS iceberg.iceberg.customers (
    customer_id INT,
    name VARCHAR(100),
    email VARCHAR(100),
    phone VARCHAR(20),
    address VARCHAR(100),
    city VARCHAR(50),
    region VARCHAR(50),
    signup_date DATE,
    loyalty_tier VARCHAR(20)
) WITH (
    format = 'PARQUET',
    location = 's3://iceberg/customers'
);

CREATE TABLE IF NOT EXISTS iceberg.iceberg.products (
    product_id INT,
    name VARCHAR(100),
    category VARCHAR(50),
    subcategory VARCHAR(50),
    price DECIMAL(6,2),
    cost DECIMAL(6,2),
    in_stock BOOLEAN,
    min_stock INT,
    max_stock INT,
    supplier VARCHAR(100)
) WITH (
    format = 'PARQUET',
    location = 's3://iceberg/products'
);

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
    region VARCHAR(50),
    payment_method VARCHAR(50)
) WITH (
    format = 'PARQUET',
    partitioning = ARRAY['region', 'payment_method'],
    location = 's3://iceberg/sales'
);

-- Insert sample customers
INSERT INTO iceberg.iceberg.customers
SELECT 
    seq AS customer_id,
    'Customer ' || CAST(seq AS VARCHAR) AS name,
    'customer' || CAST(seq AS VARCHAR) || '@example.com' AS email,
    '+1-555-' || LPAD(CAST(seq AS VARCHAR), 4, '0') AS phone,
    CAST(seq * 123 AS VARCHAR) || ' Main St' AS address,
    CASE CAST(RANDOM() * 4 AS INTEGER)
        WHEN 0 THEN 'New York'
        WHEN 1 THEN 'Los Angeles'
        WHEN 2 THEN 'Chicago'
        WHEN 3 THEN 'Houston'
    END AS city,
    CASE CAST(RANDOM() * 4 AS INTEGER)
        WHEN 0 THEN 'North'
        WHEN 1 THEN 'South'
        WHEN 2 THEN 'East'
        WHEN 3 THEN 'West'
    END AS region,
    DATE_ADD('day', -CAST(RANDOM() * 1000 AS INTEGER), CURRENT_DATE) AS signup_date,
    CASE CAST(RANDOM() * 3 AS INTEGER)
        WHEN 0 THEN 'Bronze'
        WHEN 1 THEN 'Silver'
        WHEN 2 THEN 'Gold'
    END AS loyalty_tier
FROM UNNEST(SEQUENCE(1, 500)) AS t(seq);

-- Insert sample products
INSERT INTO iceberg.iceberg.products
SELECT
    seq AS product_id,
    'Product ' || CAST(seq AS VARCHAR) AS name,
    CASE CAST(RANDOM() * 3 AS INTEGER)
        WHEN 0 THEN 'Electronics'
        WHEN 1 THEN 'Clothing'
        WHEN 2 THEN 'Home'
    END AS category,
    CASE CAST(RANDOM() * 3 AS INTEGER)
        WHEN 0 THEN 'Accessories'
        WHEN 1 THEN 'Basics'
        WHEN 2 THEN 'Premium'
    END AS subcategory,
    CAST(RANDOM() * 1000 + 10 AS DECIMAL(6,2)) AS price,
    CAST(RANDOM() * 500 + 5 AS DECIMAL(6,2)) AS cost,
    CAST(RANDOM() > 0.5 AS BOOLEAN) AS in_stock,
    100 AS min_stock,
    1000 AS max_stock,
    'Supplier ' || CAST(CAST(RANDOM() * 10 AS INTEGER) AS VARCHAR) AS supplier
FROM UNNEST(SEQUENCE(1, 200)) AS t(seq);

-- Insert sample sales
INSERT INTO iceberg.iceberg.sales
SELECT
    CAST(RANDOM() * 10000 AS INTEGER) AS order_id,
    DATE_ADD('day', -CAST(RANDOM() * 365 AS INTEGER), CURRENT_DATE) AS order_date,
    c.customer_id,
    p.product_id,
    CAST(RANDOM() * 10 AS INTEGER) + 1 AS quantity,
    p.price AS unit_price,
    p.price * (CAST(RANDOM() * 10 AS INTEGER) + 1) AS gross_amount,
    CAST(RANDOM() * (p.price * (CAST(RANDOM() * 10 AS INTEGER) + 1) * 0.2) AS DECIMAL(10,2)) AS discount,
    (p.price * (CAST(RANDOM() * 10 AS INTEGER) + 1)) - CAST(RANDOM() * (p.price * (CAST(RANDOM() * 10 AS INTEGER) + 1) * 0.2) AS DECIMAL(10,2)) AS net_amount,
    c.region,
    CASE CAST(RANDOM() * 3 AS INTEGER)
        WHEN 0 THEN 'Credit Card'
        WHEN 1 THEN 'Debit Card'
        WHEN 2 THEN 'PayPal'
    END AS payment_method
FROM iceberg.iceberg.customers c
CROSS JOIN iceberg.iceberg.products p
CROSS JOIN UNNEST(SEQUENCE(1, 20)) AS t(seq)
WHERE CAST(RANDOM() * 500 AS INTEGER) + 1 = c.customer_id
AND CAST(RANDOM() * 200 AS INTEGER) + 1 = p.product_id; 