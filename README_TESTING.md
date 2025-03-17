# AI Translate Testing Framework

This directory contains a comprehensive testing framework for the AI translate functionality in Trino-AI. The framework is designed to test the translation of natural language queries to SQL, as well as the execution of the translated SQL queries against Trino.

## Test Scripts

The testing framework consists of the following scripts:

1. **test_trino_ai_translate.py**: Tests basic AI translate functionality directly from Trino.
2. **test_complex_queries.py**: Tests the AI translate API with complex natural language queries.
3. **test_complex_trino_queries.py**: Tests the AI translate function with complex queries directly from Trino.
4. **test_execute_translated_queries.py**: Tests the execution of translated SQL queries against Trino.
5. **run_all_tests.py**: Runs all test scripts in sequence and generates a comprehensive report.

## Test Queries

The test scripts include a variety of natural language queries, ranging from simple to complex:

### Basic Queries
- Show me all customers
- List the top 10 most expensive products
- Show me customers who purchased products over $100
- Find total sales by product category
- Calculate average purchase amount per customer
- Show me monthly sales trends

### Complex Queries
- Calculate the average purchase amount for each customer and rank them by their spending
- Show me the top 5 most profitable product categories based on total revenue minus cost
- Find customers who spend more than the average amount in their region and show their purchase history
- Show monthly sales trends by payment method over the last year with percentage change month over month
- Identify products with higher than average discount rates and analyze their sales performance compared to full-price items
- Categorize customers into spending tiers (high, medium, low) based on their purchase history and show their loyalty status

## Running the Tests

To run all tests, execute the following command:

```bash
python run_all_tests.py
```

This will run all test scripts in sequence and generate a comprehensive report in the `test_results` directory.

To run individual tests, execute the corresponding script:

```bash
python test_trino_ai_translate.py
python test_complex_queries.py
python test_complex_trino_queries.py
python test_execute_translated_queries.py
```

## Test Results

The test results are stored in the `test_results` directory. The following files are generated:

- **all_tests.log**: Log file containing the output of all tests.
- **trino_ai_translate_test_results.json**: Results of basic AI translate tests.
- **ai_translate_test_results.json**: Results of complex queries API tests.
- **complex_trino_queries_results.json**: Results of complex queries Trino tests.
- **execution_results.json**: Results of executing translated SQL queries.

Additionally, individual test results are stored in the following directories:

- **test_results/**: Contains individual query results.
- **test_results/execution/**: Contains individual execution results.

## Interpreting the Results

Each test result includes the following information:

- **query_number**: The number of the query.
- **natural_language_query**: The natural language query that was tested.
- **success**: Whether the translation was successful.
- **execution_time**: The time it took to execute the query.
- **sql**: The translated SQL query (if successful).
- **error**: The error message (if unsuccessful).

For execution results, the following additional information is included:

- **execution_success**: Whether the execution was successful.
- **output**: The output of the query (if successful).
- **error**: The error message (if unsuccessful).

## Troubleshooting

If you encounter any issues with the tests, check the following:

1. Make sure the Trino-AI service is running.
2. Make sure the AI translate function is properly defined in Trino.
3. Check the log files for error messages.
4. Verify that the Docker containers are running correctly.

## Adding New Tests

To add new tests, modify the `TEST_QUERIES` or `COMPLEX_QUERIES` lists in the corresponding test scripts. You can also create new test scripts following the same pattern as the existing ones.

## Conclusion

This testing framework provides a comprehensive way to test the AI translate functionality in Trino-AI. By running these tests regularly, you can ensure that the translation service is working correctly and identify any issues that need to be addressed. 