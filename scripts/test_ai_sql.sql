-- First, let's use ai_gen to generate a SQL query
WITH generated_query AS (
  SELECT "ai-functions".ai.ai_gen('Write a SQL query to find the top 5 customers by total sales amount in the iceberg.iceberg.sales and iceberg.iceberg.customers tables. The customer_id in sales table links to customer_id in customers table. Return only the SQL, no explanations.') AS sql_query
)
-- Now let's print the generated query for reference
SELECT sql_query FROM generated_query; 