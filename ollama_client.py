from openai import OpenAI
import os
from typing import List, Dict, Any, Generator, Optional

class OllamaClient:
    def __init__(self, model: str = "llama3:8b"):
        self.client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="none"  # Required but ignored
        )
        self.default_model = model
    
    def generate_sql(self, context: str, question: str, model: Optional[str] = None) -> str:
        """Generate SQL query based on schema context and question"""
        completion = self.client.chat.completions.create(
            model=model or self.default_model,
            messages=[{
                "role": "system",
                "content": f"""You are a Trino SQL expert. Use this schema context:
                {context}
                
                Rules:
                1. Use only tables and columns from the provided schema
                2. Add LIMIT 1000 for large result sets
                3. Use Trino-specific functions when needed
                4. Ensure proper table qualifications
                5. Add appropriate JOINs based on schema relationships"""
            }, {
                "role": "user",
                "content": question
            }],
            temperature=0
        )
        return completion.choices[0].message.content

    def explain_results(self, sql: str, results: List[Any], model: Optional[str] = None) -> str:
        """Explain SQL query results in plain English"""
        completion = self.client.chat.completions.create(
            model=model or self.default_model,
            messages=[{
                "role": "system",
                "content": """Explain these SQL results in plain English. Include:
                1. Summary of what the query does
                2. Key insights from the results
                3. Any notable patterns or outliers"""
            }, {
                "role": "user",
                "content": f"SQL: {sql}\nResults: {results}"
            }],
            temperature=0.2
        )
        return completion.choices[0].message.content

    def stream_explanation(self, sql: str, results: List[Any], model: Optional[str] = None) -> Generator[str, None, None]:
        """Stream the explanation response"""
        completion = self.client.chat.completions.create(
            model=model or self.default_model,
            messages=[{
                "role": "system",
                "content": "Explain these SQL results in plain English:"
            }, {
                "role": "user",
                "content": f"SQL: {sql}\nResults: {results}"
            }],
            temperature=0.2,
            stream=True
        )
        for chunk in completion:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content 