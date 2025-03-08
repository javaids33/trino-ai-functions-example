import requests
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_ai_translate_api():
    """Test the AI translate API directly."""
    url = "http://localhost:5001/api/ai_translate"
    
    # Test query
    query = "What is the average sales amount by region?"
    
    # Prepare the request payload
    payload = {
        "query": query,
        "execute": True
    }
    
    logger.info(f"Sending request to {url} with query: {query}")
    
    try:
        # Send the request
        response = requests.post(url, json=payload, timeout=60)
        
        # Check if the request was successful
        if response.status_code == 200:
            logger.info("Request successful!")
            
            # Parse the response
            result = response.json()
            
            # Print the result
            logger.info("Response:")
            logger.info(json.dumps(result, indent=2))
            
            return result
        else:
            logger.error(f"Request failed with status code {response.status_code}")
            logger.error(f"Response: {response.text}")
            return None
    except Exception as e:
        logger.error(f"Error sending request: {str(e)}")
        return None

if __name__ == "__main__":
    test_ai_translate_api() 