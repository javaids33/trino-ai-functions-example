-- Query to demonstrate the ai_mask function for masking sensitive customer information
SELECT 
  c.customer_id,
  c.name,
  c.email,
  c.phone,
  "ai-functions".ai.ai_mask(
    CONCAT('Customer ', c.name, ' with email ', c.email, ' and phone ', c.phone), 
    ARRAY['email', 'phone number']
  ) AS masked_info
FROM 
  iceberg.iceberg.customers c
LIMIT 5; 