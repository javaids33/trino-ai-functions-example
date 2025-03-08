import subprocess
import logging
import time
import os
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test_results/all_tests.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def run_command(command, description):
    """Run a command and log the output."""
    logger.info(f"\n\n{'=' * 50}")
    logger.info(f"Running: {description}")
    logger.info(f"Command: {command}")
    logger.info(f"{'=' * 50}\n")
    
    start_time = time.time()
    
    try:
        # Run the command and capture output
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True
        )
        
        execution_time = time.time() - start_time
        
        # Log the output
        if result.stdout:
            logger.info(f"Output:\n{result.stdout}")
        
        # Log any errors
        if result.stderr:
            logger.error(f"Error output:\n{result.stderr}")
        
        # Log the result
        if result.returncode == 0:
            logger.info(f"✅ {description} completed successfully in {execution_time:.2f} seconds")
            return True
        else:
            logger.error(f"❌ {description} failed with return code {result.returncode} after {execution_time:.2f} seconds")
            return False
    
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"❌ {description} failed with exception: {str(e)} after {execution_time:.2f} seconds")
        return False

def run_all_tests():
    """Run all test scripts in sequence."""
    # Create results directory if it doesn't exist
    os.makedirs("test_results", exist_ok=True)
    
    logger.info("\n\n" + "=" * 80)
    logger.info("STARTING COMPREHENSIVE AI TRANSLATE TESTING")
    logger.info("=" * 80 + "\n")
    
    # List of tests to run
    tests = [
        {
            "command": "python test_trino_ai_translate.py",
            "description": "Basic AI Translate Tests"
        },
        {
            "command": "python test_complex_queries.py",
            "description": "Complex Queries API Tests"
        },
        {
            "command": "python test_complex_trino_queries.py",
            "description": "Complex Queries Trino Tests"
        },
        {
            "command": "python test_execute_translated_queries.py",
            "description": "Execute Translated Queries Tests"
        }
    ]
    
    # Run each test
    results = []
    for test in tests:
        success = run_command(test["command"], test["description"])
        results.append({
            "test": test["description"],
            "success": success
        })
    
    # Print summary
    logger.info("\n\n" + "=" * 80)
    logger.info("TEST EXECUTION SUMMARY")
    logger.info("=" * 80)
    
    for result in results:
        status = "✅ PASSED" if result["success"] else "❌ FAILED"
        logger.info(f"{status}: {result['test']}")
    
    success_count = sum(1 for r in results if r["success"])
    logger.info(f"\nTotal tests: {len(tests)}")
    logger.info(f"Successful tests: {success_count}")
    logger.info(f"Failed tests: {len(tests) - success_count}")
    
    if success_count == len(tests):
        logger.info("\n✅ ALL TESTS PASSED")
    else:
        logger.info(f"\n❌ {len(tests) - success_count} TESTS FAILED")
    
    logger.info("\n" + "=" * 80)
    logger.info("END OF TEST EXECUTION")
    logger.info("=" * 80)

if __name__ == "__main__":
    run_all_tests() 