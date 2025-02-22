CREATE SCHEMA IF NOT EXISTS nessie.iceberg WITH (
    location = 's3a://iceberg/'
);

CREATE TABLE nessie.iceberg.sales (
    order_date DATE,
    product_id INT,
    customer_id INT,
    amount DECIMAL(10,2),
    region VARCHAR
) WITH (
    format = 'PARQUET',
    partitioning = ARRAY['region']
);

INSERT INTO nessie.iceberg.sales
SELECT
    DATE_ADD('day', -CAST(RANDOM() * 365 AS INTEGER), CURRENT_DATE),
    CAST(RANDOM() * 1000 AS INTEGER),
    CAST(RANDOM() * 500 AS INTEGER),
    CAST(RANDOM() * 1000 AS DECIMAL(10,2)),
    CASE CAST(RANDOM() * 4 AS INTEGER)
        WHEN 0 THEN 'North'
        WHEN 1 THEN 'South'
        WHEN 2 THEN 'East'
        WHEN 3 THEN 'West'
    END
FROM UNNEST(SEQUENCE(1, 10000)) AS t; 