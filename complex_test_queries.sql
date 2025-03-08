-- Test Query 1: Basic aggregation with grouping
-- "Calculate the average purchase amount per customer, ordered by highest average"
SELECT 
    c.customer_id,
    c.name,
    AVG(s.net_amount) as avg_purchase_amount
FROM 
    iceberg.iceberg.customers c
JOIN 
    iceberg.iceberg.sales s ON c.customer_id = s.customer_id
GROUP BY 
    c.customer_id, c.name
ORDER BY 
    avg_purchase_amount DESC;

-- Test Query 2: Complex joins with filtering and aggregation
-- "Show me the top 5 most profitable product categories with their total sales and profit margin"
SELECT 
    p.category,
    COUNT(s.order_id) as total_orders,
    SUM(s.quantity) as total_quantity_sold,
    SUM(s.net_amount) as total_revenue,
    SUM(s.net_amount - (p.cost * s.quantity)) as total_profit,
    (SUM(s.net_amount - (p.cost * s.quantity)) / SUM(s.net_amount)) * 100 as profit_margin_percentage
FROM 
    iceberg.iceberg.sales s
JOIN 
    iceberg.iceberg.products p ON s.product_id = p.product_id
GROUP BY 
    p.category
ORDER BY 
    total_profit DESC
LIMIT 5;

-- Test Query 3: Window functions with conditional filtering
-- "Find customers who spend more than the average in their region"
WITH region_averages AS (
    SELECT 
        c.region,
        AVG(s.net_amount) as region_avg_spend
    FROM 
        iceberg.iceberg.customers c
    JOIN 
        iceberg.iceberg.sales s ON c.customer_id = s.customer_id
    GROUP BY 
        c.region
),
customer_spending AS (
    SELECT 
        c.customer_id,
        c.name,
        c.region,
        SUM(s.net_amount) as total_spend
    FROM 
        iceberg.iceberg.customers c
    JOIN 
        iceberg.iceberg.sales s ON c.customer_id = s.customer_id
    GROUP BY 
        c.customer_id, c.name, c.region
)
SELECT 
    cs.customer_id,
    cs.name,
    cs.region,
    cs.total_spend,
    ra.region_avg_spend,
    (cs.total_spend - ra.region_avg_spend) as spend_vs_region_avg
FROM 
    customer_spending cs
JOIN 
    region_averages ra ON cs.region = ra.region
WHERE 
    cs.total_spend > ra.region_avg_spend
ORDER BY 
    spend_vs_region_avg DESC;

-- Test Query 4: Temporal analysis with date functions
-- "Show monthly sales trends by payment method over the last year"
SELECT 
    DATE_TRUNC('month', s.order_date) as month,
    s.payment_method,
    COUNT(s.order_id) as order_count,
    SUM(s.net_amount) as total_sales,
    AVG(s.net_amount) as avg_order_value
FROM 
    iceberg.iceberg.sales s
WHERE 
    s.order_date >= DATE_TRUNC('year', CURRENT_DATE) - INTERVAL '1' YEAR
GROUP BY 
    DATE_TRUNC('month', s.order_date), s.payment_method
ORDER BY 
    month, payment_method;

-- Test Query 5: Subqueries with multiple aggregations
-- "Identify products that have higher than average discount rates and their sales performance"
WITH product_discounts AS (
    SELECT 
        s.product_id,
        AVG(s.discount / s.gross_amount) * 100 as avg_discount_percentage
    FROM 
        iceberg.iceberg.sales s
    GROUP BY 
        s.product_id
),
avg_discount AS (
    SELECT 
        AVG(avg_discount_percentage) as overall_avg_discount
    FROM 
        product_discounts
)
SELECT 
    p.product_id,
    p.name,
    p.category,
    pd.avg_discount_percentage,
    ad.overall_avg_discount,
    COUNT(s.order_id) as total_orders,
    SUM(s.quantity) as total_quantity_sold,
    SUM(s.net_amount) as total_revenue
FROM 
    iceberg.iceberg.products p
JOIN 
    product_discounts pd ON p.product_id = pd.product_id
CROSS JOIN 
    avg_discount ad
JOIN 
    iceberg.iceberg.sales s ON p.product_id = s.product_id
WHERE 
    pd.avg_discount_percentage > ad.overall_avg_discount
GROUP BY 
    p.product_id, p.name, p.category, pd.avg_discount_percentage, ad.overall_avg_discount
ORDER BY 
    pd.avg_discount_percentage DESC;

-- Test Query 6: Complex conditional logic
-- "Categorize customers by spending habits and loyalty tier"
WITH customer_metrics AS (
    SELECT 
        c.customer_id,
        c.name,
        c.loyalty_tier,
        COUNT(DISTINCT s.order_id) as order_count,
        SUM(s.net_amount) as total_spend,
        MAX(s.order_date) as last_order_date,
        DATEDIFF('day', MAX(s.order_date), CURRENT_DATE) as days_since_last_order
    FROM 
        iceberg.iceberg.customers c
    LEFT JOIN 
        iceberg.iceberg.sales s ON c.customer_id = s.customer_id
    GROUP BY 
        c.customer_id, c.name, c.loyalty_tier
)
SELECT 
    cm.customer_id,
    cm.name,
    cm.loyalty_tier,
    cm.order_count,
    cm.total_spend,
    cm.days_since_last_order,
    CASE 
        WHEN cm.order_count = 0 THEN 'Never Purchased'
        WHEN cm.days_since_last_order > 365 THEN 'Inactive'
        WHEN cm.days_since_last_order > 180 THEN 'At Risk'
        WHEN cm.days_since_last_order > 90 THEN 'Cooling Down'
        ELSE 'Active'
    END as activity_status,
    CASE 
        WHEN cm.total_spend > 5000 THEN 'High Value'
        WHEN cm.total_spend > 1000 THEN 'Medium Value'
        WHEN cm.total_spend > 0 THEN 'Low Value'
        ELSE 'No Value'
    END as value_segment
FROM 
    customer_metrics cm
ORDER BY 
    cm.total_spend DESC; 