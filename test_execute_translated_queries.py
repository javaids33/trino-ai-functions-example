import subprocess
import logging
import json
import time
import os
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_translated_queries():
    """Load the translated queries from the results file."""
    try:
        results_file = "test_results/complex_trino_queries_results.json"
        
        # Check if the file exists
        if not os.path.exists(results_file):
            logger.error(f"Results file {results_file} not found. Run test_complex_trino_queries.py first.")
            return []
        
        with open(results_file, "r") as f:
            results = json.load(f)
        
        # Filter for successful translations
        successful_translations = [r for r in results if r.get("success", False)]
        
        if not successful_translations:
            logger.warning("No successful translations found in the results file.")
        
        return successful_translations
    
    except Exception as e:
        logger.error(f"Error loading translated queries: {str(e)}")
        return []

def execute_sql_query(sql):
    """Execute a SQL query using the Trino CLI and return the output."""
    try:
        # Escape single quotes in the query
        escaped_sql = sql.replace("'", "\\'")
        
        # Construct the Trino CLI command
        cmd = f'docker-compose exec trino trino --execute "{escaped_sql}"'
        
        logger.info(f"Running SQL query: {sql}")
        logger.info(f"Command: {cmd}")
        
        # Execute the command
        start_time = time.time()
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        execution_time = time.time() - start_time
        
        # Check if the command was successful
        if result.returncode == 0:
            output = result.stdout.strip()
            
            # Truncate output if it's too long
            if len(output) > 1000:
                truncated_output = output[:1000] + "... [truncated]"
                logger.info(f"Query output (truncated): {truncated_output}")
            else:
                logger.info(f"Query output: {output}")
            
            return {
                "success": True,
                "output": output,
                "execution_time": execution_time
            }
        else:
            logger.error(f"Query execution failed with return code {result.returncode}")
            logger.error(f"Error output: {result.stderr}")
            
            return {
                "success": False,
                "error": result.stderr,
                "execution_time": execution_time
            }
    
    except Exception as e:
        logger.error(f"Error executing SQL query: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "execution_time": 0
        }

def test_execute_translated_queries():
    """Test the execution of translated SQL queries against Trino."""
    # Load the translated queries
    translated_queries = load_translated_queries()
    
    if not translated_queries:
        logger.error("No translated queries to execute. Exiting.")
        return
    
    # Create results directory if it doesn't exist
    os.makedirs("test_results/execution", exist_ok=True)
    
    execution_results = []
    
    for i, query_result in enumerate(translated_queries, 1):
        query_number = query_result.get("query_number", i)
        nl_query = query_result.get("natural_language_query", "Unknown")
        sql = query_result.get("sql", "")
        
        logger.info(f"\n\n===== Executing Query {query_number}: {nl_query} =====")
        
        # Execute the SQL query
        execution_result = execute_sql_query(sql)
        
        # Store the result
        result = {
            "query_number": query_number,
            "natural_language_query": nl_query,
            "sql": sql,
            "execution_success": execution_result["success"],
            "execution_time": execution_result["execution_time"],
            **({"output": execution_result["output"]} if execution_result["success"] else {"error": execution_result["error"]})
        }
        
        execution_results.append(result)
        
        # Save individual execution result to file
        with open(f"test_results/execution/query_{query_number}_execution.json", "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Execution result for query {query_number} saved to test_results/execution/query_{query_number}_execution.json")
    
    # Print summary
    logger.info("\n\n===== Execution Test Summary =====")
    success_count = sum(1 for r in execution_results if r.get("execution_success", False))
    logger.info(f"Total queries executed: {len(execution_results)}")
    logger.info(f"Successful executions: {success_count}")
    logger.info(f"Failed executions: {len(execution_results) - success_count}")
    
    if success_count > 0:
        avg_time = sum(r["execution_time"] for r in execution_results if r.get("execution_success", False)) / success_count
        logger.info(f"Average execution time: {avg_time:.2f} seconds")
    
    # Save all execution results to file
    with open("test_results/execution_results.json", "w") as f:
        json.dump(execution_results, f, indent=2)
    logger.info("All execution results saved to test_results/execution_results.json")

if __name__ == "__main__":
    test_execute_translated_queries() 