import subprocess
import logging
import json
import time
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# List of natural language queries to test
TEST_QUERIES = [
    "Show me all customers",
    "List the top 10 most expensive products",
    "Show me customers who purchased products over $100",
    "Find total sales by product category",
    "Calculate average purchase amount per customer",
    "Show me monthly sales trends"
]

def run_trino_query(query):
    """Run a query using the Trino CLI and return the output."""
    try:
        # Escape single quotes in the query
        escaped_query = query.replace("'", "\\'")
        
        # Construct the Trino CLI command
        cmd = f'docker-compose exec trino trino --execute "SELECT \\"ai-functions\\".ai.ai_translate(\'{escaped_query}\', \'sql\')"'
        
        logger.info(f"Running command: {cmd}")
        
        # Execute the command
        start_time = time.time()
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        execution_time = time.time() - start_time
        
        # Check if the command was successful
        if result.returncode == 0:
            output = result.stdout.strip()
            logger.info(f"Command output: {output}")
            
            # Extract the SQL query from the output
            # The output might be wrapped in quotes, so we need to handle that
            if output.startswith('"') and output.endswith('"'):
                # Remove the surrounding quotes and unescape any internal quotes
                sql = output[1:-1].replace('\\"', '"')
            else:
                sql = output
                
            return {
                "success": True,
                "sql": sql,
                "execution_time": execution_time
            }
        else:
            logger.error(f"Command failed with return code {result.returncode}")
            logger.error(f"Error output: {result.stderr}")
            
            return {
                "success": False,
                "error": result.stderr,
                "execution_time": execution_time
            }
    
    except Exception as e:
        logger.error(f"Error running Trino query: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "execution_time": 0
        }

def test_trino_ai_translate():
    """Test the AI translate function directly from Trino."""
    results = []
    
    for i, query in enumerate(TEST_QUERIES, 1):
        logger.info(f"\n\n===== Testing Query {i}: {query} =====")
        
        result = run_trino_query(query)
        
        # Store the result
        results.append({
            "query_number": i,
            "natural_language_query": query,
            "success": result["success"],
            "execution_time": result["execution_time"],
            **({"sql": result["sql"]} if result["success"] else {"error": result["error"]})
        })
    
    # Print summary
    logger.info("\n\n===== Test Summary =====")
    success_count = sum(1 for r in results if r["success"])
    logger.info(f"Total queries: {len(TEST_QUERIES)}")
    logger.info(f"Successful translations: {success_count}")
    logger.info(f"Failed translations: {len(TEST_QUERIES) - success_count}")
    
    if success_count > 0:
        avg_time = sum(r["execution_time"] for r in results if r["success"]) / success_count
        logger.info(f"Average execution time: {avg_time:.2f} seconds")
    
    # Save results to file
    with open("trino_ai_translate_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to trino_ai_translate_test_results.json")

if __name__ == "__main__":
    test_trino_ai_translate() 