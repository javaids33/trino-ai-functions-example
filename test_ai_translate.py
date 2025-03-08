import requests
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_ai_translate():
    """Test the AI translate functionality by making a direct API call to the trino-ai service."""
    try:
        # Make a direct API call to the trino-ai service
        url = "http://localhost:5001/api/ai_translate"
        payload = {
            "query": "Show me all customers",
            "target_format": "sql"
        }
        
        logger.info(f"Sending request to {url} with payload: {payload}")
        
        response = requests.post(url, json=payload)
        
        logger.info(f"Response status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"Response: {json.dumps(result, indent=2)}")
            
            if "sql" in result:
                logger.info(f"SQL query: {result['sql']}")
            else:
                logger.error(f"No SQL query in response: {result}")
        else:
            logger.error(f"Error response: {response.text}")
    
    except Exception as e:
        logger.error(f"Error testing AI translate: {str(e)}")

if __name__ == "__main__":
    test_ai_translate() 