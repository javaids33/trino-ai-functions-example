-- Create a realistic e-commerce database with interrelated tables

-- USERS TABLE - Central entity for customer information
CREATE TABLE faker.ecommerce.users (
    user_id INTEGER COMMENT 'Unique identifier for each user',
    first_name VARCHAR COMMENT 'User first name',
    last_name VARCHAR COMMENT 'User last name',
    email VARCHAR COMMENT 'Unique email address for account access and communications',
    phone_number VARCHAR COMMENT 'Contact phone number',
    date_of_birth DATE COMMENT 'User birth date for age verification and birthday promotions',
    registration_date TIMESTAMP COMMENT 'When the user first created their account',
    last_login_timestamp TIMESTAMP COMMENT 'Most recent login time',
    is_verified BOOLEAN COMMENT 'Whether user has verified their email address',
    account_status VARCHAR COMMENT 'Current account status (active, suspended, closed)',
    preferred_language VARCHAR COMMENT 'User preferred language for communications',
    marketing_consent BOOLEAN COMMENT 'Whether user has opted in to marketing communications'
) COMMENT 'Core user account information and preferences';

-- ADDRESSES TABLE - User shipping and billing addresses
CREATE TABLE faker.ecommerce.addresses (
    address_id INTEGER COMMENT 'Unique identifier for each address',
    user_id INTEGER COMMENT 'Reference to the user who owns this address',
    address_type VARCHAR COMMENT 'Type of address (shipping, billing, or both)',
    street_address VARCHAR COMMENT 'Street number and name',
    apartment_unit VARCHAR COMMENT 'Apartment, suite, or unit number if applicable',
    city VARCHAR COMMENT 'City name',
    state_province VARCHAR COMMENT 'State or province name',
    postal_code VARCHAR COMMENT 'Postal or ZIP code',
    country VARCHAR COMMENT 'Country name',
    is_default BOOLEAN COMMENT 'Whether this is the user default address',
    created_at TIMESTAMP COMMENT 'When this address was added',
    last_updated TIMESTAMP COMMENT 'When this address was last updated'
) COMMENT 'User shipping and billing addresses';

-- PRODUCT CATEGORIES TABLE - Hierarchical product categories
CREATE TABLE faker.ecommerce.product_categories (
    category_id INTEGER COMMENT 'Unique identifier for each category',
    parent_category_id INTEGER COMMENT 'Reference to parent category, NULL for top-level categories',
    category_name VARCHAR COMMENT 'Display name of the category',
    category_description VARCHAR COMMENT 'Detailed description of what the category contains',
    image_url VARCHAR COMMENT 'URL to the category image',
    is_active BOOLEAN COMMENT 'Whether this category is currently active',
    display_order INTEGER COMMENT 'Order for displaying categories in the UI',
    created_at TIMESTAMP COMMENT 'When this category was created',
    last_updated TIMESTAMP COMMENT 'When this category was last updated'
) COMMENT 'Hierarchical product categorization system';

-- PRODUCTS TABLE - Main product information
CREATE TABLE faker.ecommerce.products (
    product_id INTEGER COMMENT 'Unique identifier for each product',
    category_id INTEGER COMMENT 'Reference to the product category',
    product_name VARCHAR COMMENT 'Display name of the product',
    product_description VARCHAR COMMENT 'Detailed product description',
    brand VARCHAR COMMENT 'Product brand name',
    sku VARCHAR COMMENT 'Stock keeping unit - unique product identifier',
    price DECIMAL(10,2) COMMENT 'Current retail price',
    cost DECIMAL(10,2) COMMENT 'Cost to the company',
    weight DECIMAL(8,2) COMMENT 'Product weight in grams',
    dimensions VARCHAR COMMENT 'Product dimensions in format LxWxH cm',
    is_active BOOLEAN COMMENT 'Whether product is currently available for sale',
    created_at TIMESTAMP COMMENT 'When product was added to catalog',
    last_updated TIMESTAMP COMMENT 'When product was last updated'
) COMMENT 'Core product information and attributes';

-- PRODUCT ATTRIBUTES TABLE - Additional product details
CREATE TABLE faker.ecommerce.product_attributes (
    attribute_id INTEGER COMMENT 'Unique identifier for each attribute record',
    product_id INTEGER COMMENT 'Reference to the product',
    attribute_name VARCHAR COMMENT 'Name of the attribute (color, size, material, etc.)',
    attribute_value VARCHAR COMMENT 'Value of the attribute',
    is_filterable BOOLEAN COMMENT 'Whether this attribute can be used to filter products',
    is_searchable BOOLEAN COMMENT 'Whether this attribute is searchable',
    display_order INTEGER COMMENT 'Order for displaying attributes in the UI'
) COMMENT 'Extended product attributes and specifications';

-- INVENTORY TABLE - Product stock information
CREATE TABLE faker.ecommerce.inventory (
    inventory_id INTEGER COMMENT 'Unique identifier for inventory record',
    product_id INTEGER COMMENT 'Reference to product',
    warehouse_id INTEGER COMMENT 'Reference to warehouse location',
    quantity_available INTEGER COMMENT 'Current available quantity',
    quantity_reserved INTEGER COMMENT 'Quantity reserved for pending orders',
    reorder_level INTEGER COMMENT 'Quantity threshold to trigger reordering',
    restock_eta TIMESTAMP COMMENT 'Estimated arrival time for next restock',
    last_stock_check TIMESTAMP COMMENT 'When inventory was last physically verified',
    last_updated TIMESTAMP COMMENT 'When this record was last updated'
) COMMENT 'Product stock levels and inventory management';

-- WAREHOUSES TABLE - Storage locations
CREATE TABLE faker.ecommerce.warehouses (
    warehouse_id INTEGER COMMENT 'Unique identifier for each warehouse',
    warehouse_name VARCHAR COMMENT 'Name of the warehouse',
    address_line VARCHAR COMMENT 'Physical address of the warehouse',
    city VARCHAR COMMENT 'City where warehouse is located',
    state_province VARCHAR COMMENT 'State or province of warehouse location',
    postal_code VARCHAR COMMENT 'Postal code of warehouse location',
    country VARCHAR COMMENT 'Country where warehouse is located',
    is_active BOOLEAN COMMENT 'Whether this warehouse is currently operational',
    storage_capacity INTEGER COMMENT 'Total storage capacity in cubic meters',
    manager_name VARCHAR COMMENT 'Name of the warehouse manager',
    contact_email VARCHAR COMMENT 'Email for warehouse contact',
    contact_phone VARCHAR COMMENT 'Phone number for warehouse contact'
) COMMENT 'Physical warehouse and storage locations';

-- ORDERS TABLE - Customer order information
CREATE TABLE faker.ecommerce.orders (
    order_id INTEGER COMMENT 'Unique identifier for each order',
    user_id INTEGER COMMENT 'Reference to customer who placed the order',
    order_status VARCHAR COMMENT 'Current status (pending, processing, shipped, delivered, canceled)',
    order_date TIMESTAMP COMMENT 'When the order was placed',
    shipping_address_id INTEGER COMMENT 'Reference to shipping address',
    billing_address_id INTEGER COMMENT 'Reference to billing address',
    shipping_method VARCHAR COMMENT 'Selected shipping method',
    shipping_cost DECIMAL(10,2) COMMENT 'Cost of shipping',
    tax_amount DECIMAL(10,2) COMMENT 'Tax charged on order',
    discount_amount DECIMAL(10,2) COMMENT 'Discount applied to order',
    total_amount DECIMAL(10,2) COMMENT 'Total order amount after discounts and taxes',
    payment_method VARCHAR COMMENT 'Method of payment',
    notes VARCHAR COMMENT 'Additional order notes or instructions',
    estimated_delivery TIMESTAMP COMMENT 'Estimated delivery date'
) COMMENT 'Customer orders and associated details';

-- ORDER ITEMS TABLE - Individual items in each order
CREATE TABLE faker.ecommerce.order_items (
    order_item_id INTEGER COMMENT 'Unique identifier for each order item',
    order_id INTEGER COMMENT 'Reference to the parent order',
    product_id INTEGER COMMENT 'Reference to the purchased product',
    quantity INTEGER COMMENT 'Quantity of the product ordered',
    unit_price DECIMAL(10,2) COMMENT 'Price per unit at time of purchase',
    discount_amount DECIMAL(10,2) COMMENT 'Discount applied to this specific item',
    total_price DECIMAL(10,2) COMMENT 'Total price for this line item after discounts',
    is_gift BOOLEAN COMMENT 'Whether item is marked as a gift',
    gift_message VARCHAR COMMENT 'Optional gift message'
) COMMENT 'Individual line items within customer orders';

-- PAYMENTS TABLE - Order payment information
CREATE TABLE faker.ecommerce.payments (
    payment_id INTEGER COMMENT 'Unique identifier for each payment',
    order_id INTEGER COMMENT 'Reference to the associated order',
    payment_method VARCHAR COMMENT 'Method of payment (credit card, PayPal, etc.)',
    payment_status VARCHAR COMMENT 'Current status (pending, completed, failed, refunded)',
    amount DECIMAL(10,2) COMMENT 'Payment amount',
    transaction_id VARCHAR COMMENT 'Payment processor transaction ID',
    payment_date TIMESTAMP COMMENT 'When the payment was processed',
    card_last_four VARCHAR COMMENT 'Last four digits of payment card if applicable',
    card_type VARCHAR COMMENT 'Type of card used if applicable',
    billing_address_id INTEGER COMMENT 'Reference to billing address used for payment'
) COMMENT 'Order payment transactions and status';

-- REVIEWS TABLE - Product reviews and ratings
CREATE TABLE faker.ecommerce.reviews (
    review_id INTEGER COMMENT 'Unique identifier for each review',
    product_id INTEGER COMMENT 'Reference to the reviewed product',
    user_id INTEGER COMMENT 'Reference to the user who wrote the review',
    rating INTEGER COMMENT 'Rating score (typically 1-5)',
    review_title VARCHAR COMMENT 'Short headline for the review',
    review_text VARCHAR COMMENT 'Full review content',
    is_verified_purchase BOOLEAN COMMENT 'Whether reviewer actually purchased the product',
    is_recommended BOOLEAN COMMENT 'Whether reviewer recommends this product',
    helpful_votes INTEGER COMMENT 'Number of users who found this review helpful',
    submission_date TIMESTAMP COMMENT 'When the review was submitted',
    last_updated TIMESTAMP COMMENT 'When the review was last updated',
    status VARCHAR COMMENT 'Review status (pending, approved, rejected)'
) COMMENT 'Customer product reviews and ratings';

-- SUPPLIERS TABLE - Product vendor information
CREATE TABLE faker.ecommerce.suppliers (
    supplier_id INTEGER COMMENT 'Unique identifier for each supplier',
    supplier_name VARCHAR COMMENT 'Company name of the supplier',
    contact_name VARCHAR COMMENT 'Name of primary contact person',
    contact_email VARCHAR COMMENT 'Email of primary contact',
    contact_phone VARCHAR COMMENT 'Phone number of primary contact',
    address VARCHAR COMMENT 'Physical address of supplier',
    city VARCHAR COMMENT 'City of supplier location',
    state_province VARCHAR COMMENT 'State or province of supplier',
    postal_code VARCHAR COMMENT 'Postal code of supplier',
    country VARCHAR COMMENT 'Country of supplier',
    tax_id VARCHAR COMMENT 'Tax or business identification number',
    payment_terms VARCHAR COMMENT 'Standard payment terms',
    notes VARCHAR COMMENT 'Additional notes about supplier'
) COMMENT 'Product vendors and supplier details';

-- PRODUCT_SUPPLIERS TABLE - Maps products to their suppliers
CREATE TABLE faker.ecommerce.product_suppliers (
    product_supplier_id INTEGER COMMENT 'Unique identifier for product-supplier relationship',
    product_id INTEGER COMMENT 'Reference to the product',
    supplier_id INTEGER COMMENT 'Reference to the supplier',
    supplier_product_code VARCHAR COMMENT 'Supplier internal code for the product',
    unit_cost DECIMAL(10,2) COMMENT 'Cost per unit from this supplier',
    minimum_order_quantity INTEGER COMMENT 'Minimum quantity supplier requires per order',
    lead_time_days INTEGER COMMENT 'Typical days from order to delivery',
    is_primary_supplier BOOLEAN COMMENT 'Whether this is the primary supplier for the product',
    last_order_date TIMESTAMP COMMENT 'Date of most recent order from this supplier'
) COMMENT 'Mapping between products and their suppliers with supply chain details';

-- MARKETING_CAMPAIGNS TABLE - Marketing campaign information
CREATE TABLE faker.ecommerce.marketing_campaigns (
    campaign_id INTEGER COMMENT 'Unique identifier for each marketing campaign',
    campaign_name VARCHAR COMMENT 'Name of the campaign',
    campaign_type VARCHAR COMMENT 'Type of campaign (email, social, display, etc.)',
    description VARCHAR COMMENT 'Detailed description of the campaign',
    start_date TIMESTAMP COMMENT 'When the campaign starts',
    end_date TIMESTAMP COMMENT 'When the campaign ends',
    budget DECIMAL(12,2) COMMENT 'Allocated budget for the campaign',
    target_audience VARCHAR COMMENT 'Description of the target audience',
    expected_reach INTEGER COMMENT 'Expected number of impressions',
    expected_conversion DECIMAL(5,2) COMMENT 'Expected conversion rate percentage',
    actual_cost DECIMAL(12,2) COMMENT 'Actual spent amount on the campaign',
    created_by VARCHAR COMMENT 'User who created the campaign',
    status VARCHAR COMMENT 'Current status of the campaign'
) COMMENT 'Marketing campaign planning and performance tracking';

-- USER_ACTIVITY_LOG TABLE - User behavior tracking
CREATE TABLE faker.ecommerce.user_activity_log (
    activity_id INTEGER COMMENT 'Unique identifier for each activity record',
    user_id INTEGER COMMENT 'Reference to the user, NULL for anonymous users',
    session_id VARCHAR COMMENT 'Browser session identifier',
    activity_type VARCHAR COMMENT 'Type of activity (page_view, product_view, add_to_cart, etc.)',
    activity_timestamp TIMESTAMP COMMENT 'When the activity occurred',
    ip_address VARCHAR COMMENT 'User IP address',
    user_agent VARCHAR COMMENT 'Browser user agent string',
    device_type VARCHAR COMMENT 'Type of device (desktop, mobile, tablet)',
    referrer_url VARCHAR COMMENT 'Referring URL if available',
    page_url VARCHAR COMMENT 'URL where activity occurred',
    product_id INTEGER COMMENT 'Reference to product if applicable',
    search_query VARCHAR COMMENT 'Search terms if activity was a search',
    duration_seconds INTEGER COMMENT 'Duration of activity in seconds if applicable'
) COMMENT 'User behavior tracking and analytics data';
