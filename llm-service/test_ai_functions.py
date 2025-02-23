import trino
from datetime import datetime

def test_ai_functions():
    # Connect to Trino
    conn = trino.dbapi.connect(
        host='localhost',
        port=8080,
        user='admin',
        catalog='ai-functions',
        schema='ai'
    )
    
    cur = conn.cursor()
    
    # Test cases - let's start with simpler tests first
    test_cases = [
        ("Text Generation", "SELECT ai_gen('What is Trino?')"),
        ("Sentiment Analysis", "SELECT ai_analyze_sentiment('I love working with data!')"),
        ("Translation", "SELECT ai_translate('Hello world', 'es')"),
    ]
    
    # Run tests and log results
    with open(f'ai_functions_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log', 'w') as log:
        for test_name, query in test_cases:
            try:
                print(f"\nTesting: {test_name}")
                print(f"Query: {query}")
                log.write(f"\n=== {test_name} ===\n")
                log.write(f"Query: {query}\n")
                
                cur.execute(query)
                result = cur.fetchone()
                
                print(f"Result: {result[0]}")
                log.write(f"Result: {result[0]}\n")
                
            except Exception as e:
                error_msg = f"Error in {test_name}: {str(e)}"
                print(error_msg)
                log.write(f"{error_msg}\n")

if __name__ == "__main__":
    test_ai_functions() 