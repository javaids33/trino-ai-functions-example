import logging
import json
import time
from typing import Dict, Any, List, Optional
import sys
import os

# Add the parent directory to the path so we can import from the parent module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.base_agent import Agent
from ollama_client import OllamaClient
from conversation_logger import conversation_logger

logger = logging.getLogger(__name__)

class DBAAgent(Agent):
    """
    Agent for analyzing natural language queries and determining database schema requirements.
    """
    
    def __init__(self, name: str = "dba_agent", description: str = "Analyzes natural language queries to determine database schema requirements", ollama_client: Optional[OllamaClient] = None):
        """
        Initialize the DBA agent
        
        Args:
            name: The name of the agent
            description: A description of what the agent does
            ollama_client: An optional OllamaClient instance
        """
        super().__init__(name, description)
        self.ollama_client = ollama_client
        self.logger.info("DBA Agent initialized")
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a natural language query to determine database schema requirements
        
        Args:
            inputs: A dictionary containing:
                - query: The natural language query
                - schema_context: The database schema context
                
        Returns:
            A dictionary containing:
                - tables: List of tables needed
                - columns: List of columns needed
                - joins: List of joins needed
                - filters: List of filters needed
                - aggregations: List of aggregations needed
        """
        query = inputs.get("query", "")
        schema_context = inputs.get("schema_context", "")
        
        if not query:
            self.logger.error("No query provided to DBA Agent")
            return {
                "error": "No query provided"
            }
        
        if not schema_context:
            self.logger.warning("No schema context provided to DBA Agent")
        
        self.logger.info(f"DBA Agent analyzing query: {query}")
        conversation_logger.log_trino_ai_processing("dba_analysis_start", {
            "query": query,
            "schema_context_length": len(schema_context)
        })
        
        try:
            # Analyze the query
            start_time = time.time()
            analysis = self._analyze_query(query, schema_context)
            analysis_time = time.time() - start_time
            
            self.logger.info(f"DBA analysis completed in {analysis_time:.2f}s")
            conversation_logger.log_trino_ai_processing("dba_analysis_complete", {
                "analysis_time": analysis_time,
                "tables_count": len(analysis.get("tables", [])),
                "columns_count": len(analysis.get("columns", [])),
                "joins_count": len(analysis.get("joins", []))
            })
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in DBA Agent: {str(e)}")
            conversation_logger.log_error("dba_agent", f"Error in DBA Agent: {str(e)}")
            
            return {
                "error": f"Error in DBA Agent: {str(e)}"
            }
    
    def _analyze_query(self, query: str, schema_context: str) -> Dict[str, Any]:
        """
        Analyze a natural language query to determine database schema requirements
        
        Args:
            query: The natural language query
            schema_context: The database schema context
            
        Returns:
            A dictionary containing the analysis results
        """
        if not self.ollama_client:
            raise ValueError("OllamaClient is required for DBA analysis")
        
        # Prepare the prompt for the LLM
        prompt = f"""
        You are a database administrator expert. Analyze the following natural language query and determine the database schema requirements.
        
        Natural language query: {query}
        
        Database schema:
        {schema_context}
        
        Provide a detailed analysis in JSON format with the following structure:
        {{
            "tables": [
                {{
                    "name": "table_name",
                    "alias": "optional_alias",
                    "reason": "why this table is needed"
                }}
            ],
            "columns": [
                {{
                    "table": "table_name",
                    "name": "column_name",
                    "purpose": "purpose of this column (select, filter, join, etc.)"
                }}
            ],
            "joins": [
                {{
                    "left_table": "table_name",
                    "left_column": "column_name",
                    "right_table": "table_name",
                    "right_column": "column_name",
                    "join_type": "INNER JOIN, LEFT JOIN, etc.",
                    "reason": "why this join is needed"
                }}
            ],
            "filters": [
                {{
                    "table": "table_name",
                    "column": "column_name",
                    "operator": "=, >, <, LIKE, etc.",
                    "value": "filter value",
                    "reason": "why this filter is needed"
                }}
            ],
            "aggregations": [
                {{
                    "type": "SUM, COUNT, AVG, etc.",
                    "table": "table_name",
                    "column": "column_name",
                    "alias": "optional_alias",
                    "reason": "why this aggregation is needed"
                }}
            ]
        }}
        
        Return ONLY the JSON without any explanations or markdown formatting.
        """
        
        # Call the LLM to analyze the query
        response = self.ollama_client.generate(prompt)
        
        # Extract and parse the JSON response
        try:
            # Find JSON in the response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                analysis = json.loads(json_str)
                
                # Ensure all expected keys are present
                for key in ["tables", "columns", "joins", "filters", "aggregations"]:
                    if key not in analysis:
                        analysis[key] = []
                
                return analysis
            else:
                self.logger.error("Failed to extract JSON from LLM response")
                return {
                    "tables": [],
                    "columns": [],
                    "joins": [],
                    "filters": [],
                    "aggregations": []
                }
                
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON from LLM response: {str(e)}")
            return {
                "error": f"Failed to parse analysis: {str(e)}",
                "tables": [],
                "columns": [],
                "joins": [],
                "filters": [],
                "aggregations": []
            }
    
    def get_parameters_schema(self) -> Dict[str, Any]:
        """
        Get the parameters schema for this agent
        
        Returns:
            The parameters schema for this agent
        """
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The natural language query to analyze"
                },
                "schema_context": {
                    "type": "string",
                    "description": "The database schema context"
                }
            },
            "required": ["query"]
        } 