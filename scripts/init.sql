CREATE SCHEMA IF NOT EXISTS iceberg.iceberg WITH (
    location = 's3://iceberg/'
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
    region VARCHAR,
    payment_method VARCHAR
) WITH (
    format = 'PARQUET',
    partitioning = ARRAY['region', 'payment_method'],
    location = 's3://iceberg/sales'
);

INSERT INTO iceberg.iceberg.sales (order_id, order_date, customer_id, product_id, quantity, unit_price, gross_amount, discount, net_amount, region, payment_method)
SELECT
    CAST(RANDOM() * 10000 AS INTEGER) as order_id,
    DATE_ADD('day', -CAST(RANDOM() * 365 AS INTEGER), CURRENT_DATE) as order_date,
    CAST(RANDOM() * 500 AS INTEGER) as customer_id,
    CAST(RANDOM() * 200 AS INTEGER) as product_id,
    CAST(RANDOM() * 10 AS INTEGER) + 1 as quantity,
    CAST(RANDOM() * 1000 AS DECIMAL(10,2)) as unit_price,
    CAST(RANDOM() * 1000 AS DECIMAL(10,2)) as gross_amount,
    CAST(RANDOM() * 100 AS DECIMAL(10,2)) as discount,
    CAST(RANDOM() * 900 AS DECIMAL(10,2)) as net_amount,
    CASE CAST(RANDOM() * 4 AS INTEGER)
        WHEN 0 THEN 'North'
        WHEN 1 THEN 'South'
        WHEN 2 THEN 'East'
        WHEN 3 THEN 'West'
    END as region,
    CASE CAST(RANDOM() * 3 AS INTEGER)
        WHEN 0 THEN 'Credit Card'
        WHEN 1 THEN 'Debit Card'
        WHEN 2 THEN 'PayPal'
    END as payment_method
FROM UNNEST(SEQUENCE(1, 10000)) AS t; 