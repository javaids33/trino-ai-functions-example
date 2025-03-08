import requests
import json
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# List of complex natural language queries to test
COMPLEX_QUERIES = [
    "Calculate the average purchase amount per customer, ordered by highest average",
    "Show me the top 5 most profitable product categories with their total sales and profit margin",
    "Find customers who spend more than the average in their region",
    "Show monthly sales trends by payment method over the last year",
    "Identify products that have higher than average discount rates and their sales performance",
    "Categorize customers by spending habits and loyalty tier"
]

def test_ai_translate_complex_queries():
    """Test the AI translate functionality with complex natural language queries."""
    results = []
    
    for i, query in enumerate(COMPLEX_QUERIES, 1):
        try:
            logger.info(f"\n\n===== Testing Query {i}: {query} =====")
            
            # Make a direct API call to the trino-ai service
            url = "http://localhost:5001/api/ai_translate"
            payload = {
                "query": query,
                "target_format": "sql"
            }
            
            logger.info(f"Sending request to {url}")
            
            start_time = time.time()
            response = requests.post(url, json=payload)
            execution_time = time.time() - start_time
            
            logger.info(f"Response status code: {response.status_code}")
            logger.info(f"Response time: {execution_time:.2f} seconds")
            
            if response.status_code == 200:
                result = response.json()
                
                if "sql" in result:
                    sql = result.get("sql", "")
                    logger.info(f"Generated SQL:\n{sql}\n")
                    
                    # Store the result
                    results.append({
                        "query_number": i,
                        "natural_language_query": query,
                        "generated_sql": sql,
                        "execution_time": execution_time,
                        "status": "success"
                    })
                else:
                    logger.error(f"No SQL query in response: {result}")
                    results.append({
                        "query_number": i,
                        "natural_language_query": query,
                        "error": "No SQL query in response",
                        "status": "error"
                    })
            else:
                logger.error(f"Error response: {response.text}")
                results.append({
                    "query_number": i,
                    "natural_language_query": query,
                    "error": f"HTTP {response.status_code}: {response.text}",
                    "status": "error"
                })
        
        except Exception as e:
            logger.error(f"Error testing query {i}: {str(e)}")
            results.append({
                "query_number": i,
                "natural_language_query": query,
                "error": str(e),
                "status": "error"
            })
    
    # Print summary
    logger.info("\n\n===== Test Summary =====")
    success_count = sum(1 for r in results if r["status"] == "success")
    logger.info(f"Total queries: {len(COMPLEX_QUERIES)}")
    logger.info(f"Successful translations: {success_count}")
    logger.info(f"Failed translations: {len(COMPLEX_QUERIES) - success_count}")
    
    if success_count > 0:
        avg_time = sum(r["execution_time"] for r in results if r["status"] == "success") / success_count
        logger.info(f"Average execution time: {avg_time:.2f} seconds")
    
    # Save results to file
    with open("ai_translate_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to ai_translate_test_results.json")

if __name__ == "__main__":
    test_ai_translate_complex_queries() 