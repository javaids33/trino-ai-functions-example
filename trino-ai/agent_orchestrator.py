import logging
import time
from typing import Dict, Any, List, Optional
import sys
import os

# Add the parent directory to the path so we can import from the parent module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ollama_client import OllamaClient
from agents.dba_agent import DBAAgent
from agents.sql_agent import SQLAgent
from agents.knowledge_agent import KnowledgeAgent
from tools.metadata_tools import GetSchemaContextTool, RefreshMetadataTool
from tools.sql_tools import ValidateSQLTool, ExecuteSQLTool
from colorama import Fore
from conversation_logger import conversation_logger
from trino_client import TrinoClient
from tool_registry import ToolRegistry

logger = logging.getLogger(__name__)

class AgentOrchestrator:
    """
    Orchestrates the execution of multiple agents to process natural language queries.
    """
    
    def __init__(self, ollama_client: Optional[OllamaClient] = None):
        """
        Initialize the Agent Orchestrator
        
        Args:
            ollama_client: An optional OllamaClient instance
        """
        self.ollama_client = ollama_client or OllamaClient()
        self.tool_registry = ToolRegistry()
        
        # Initialize agents
        self.agents = {
            "sql_agent": SQLAgent(ollama_client=self.ollama_client),
            "dba_agent": DBAAgent(name="dba_agent", description="Analyzes natural language queries to determine database schema requirements", ollama_client=self.ollama_client),
            "knowledge_agent": KnowledgeAgent(name="knowledge_agent", description="Retrieves and synthesizes knowledge from various sources", ollama_client=self.ollama_client)
        }
        
        logger.info("Agent orchestrator initialized")
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a natural language query
        
        Args:
            query: The natural language query
            
        Returns:
            The results of processing the query
        """
        if not query:
            logger.error("No query provided")
            return {
                "error": "No query provided"
            }
        
        logger.info(f"Processing query: {query}")
        conversation_logger.log_trino_ai_processing("query_processing_start", {
            "query": query
        })
        
        try:
            # Determine if this is a SQL query or a knowledge query
            query_type = self._determine_query_type(query)
            logger.info(f"Query type determined: {query_type}")
            
            if query_type == "sql":
                return self._process_sql_query(query)
            else:
                return self._process_knowledge_query(query)
                
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            conversation_logger.log_error("agent_orchestrator", f"Error processing query: {str(e)}")
            
            return {
                "error": f"Error processing query: {str(e)}"
            }
    
    def process_natural_language_query(self, query: str, model: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a natural language query with an optional model override
        
        Args:
            query: The natural language query
            model: Optional model to use for processing
            
        Returns:
            The results of processing the query
        """
        # Store the current model
        current_model = None
        if self.ollama_client and model:
            current_model = self.ollama_client.model
            self.ollama_client.model = model
            logger.info(f"Temporarily using model: {model}")
        
        try:
            # Process the query using the existing method
            result = self.process_query(query)
            return result
        finally:
            # Restore the original model if needed
            if self.ollama_client and current_model:
                self.ollama_client.model = current_model
                logger.info(f"Restored original model: {current_model}")
    
    def _determine_query_type(self, query: str) -> str:
        """
        Determine if a query is a SQL query or a knowledge query
        
        Args:
            query: The natural language query
            
        Returns:
            The query type: "sql" or "knowledge"
        """
        # This is a simplified implementation - in a real system, you would use a more sophisticated approach
        # For example, you might use a classifier trained on examples of SQL and knowledge queries
        
        # For now, we'll use a simple heuristic based on keywords
        sql_keywords = ["data", "database", "table", "query", "select", "show", "list", "count", "average", "sum", "report"]
        
        # Check if any SQL keywords are in the query
        if any(keyword.lower() in query.lower() for keyword in sql_keywords):
            return "sql"
        else:
            return "knowledge"
    
    def _process_sql_query(self, query: str) -> Dict[str, Any]:
        """
        Process a SQL query
        
        Args:
            query: The natural language query
            
        Returns:
            The results of processing the SQL query
        """
        logger.info(f"Processing SQL query: {query}")
        conversation_logger.log_trino_ai_processing("sql_query_processing_start", {
            "query": query
        })
        
        # Track reasoning steps
        agent_reasoning = []
        agent_reasoning.append(f"Identified query as a SQL query: '{query}'")
        
        try:
            # Get schema context
            schema_context = self._get_schema_context()
            agent_reasoning.append("Retrieved database schema context")
            
            # Use the DBA agent to analyze the query
            logger.info("Using DBA agent to analyze query")
            agent_reasoning.append("Using DBA agent to analyze query requirements")
            dba_start_time = time.time()
            dba_analysis = self.agents["dba_agent"].execute({
                "query": query,
                "schema_context": schema_context
            })
            dba_time = time.time() - dba_start_time
            logger.info(f"DBA analysis completed in {dba_time:.2f}s")
            
            # Check for errors in DBA analysis
            if "error" in dba_analysis:
                logger.error(f"Error in DBA analysis: {dba_analysis['error']}")
                agent_reasoning.append(f"Error in DBA analysis: {dba_analysis['error']}")
                return {
                    "error": f"Error in DBA analysis: {dba_analysis['error']}",
                    "agent_reasoning": agent_reasoning
                }
            
            # Add DBA analysis details to reasoning
            tables = dba_analysis.get("tables", [])
            joins = dba_analysis.get("joins", [])
            filters = dba_analysis.get("filters", [])
            
            if tables:
                table_names = [t.get("name") for t in tables]
                agent_reasoning.append(f"DBA identified relevant tables: {', '.join(table_names)}")
            
            if joins:
                join_desc = []
                for j in joins:
                    join_desc.append(f"{j.get('left_table')}.{j.get('left_column')} {j.get('join_type', 'JOIN')} {j.get('right_table')}.{j.get('right_column')}")
                agent_reasoning.append(f"DBA identified necessary joins: {'; '.join(join_desc)}")
            
            if filters:
                filter_desc = []
                for f in filters:
                    filter_desc.append(f"{f.get('table')}.{f.get('column')} {f.get('operator')} {f.get('value')}")
                agent_reasoning.append(f"DBA identified filters: {'; '.join(filter_desc)}")
            
            # Use the SQL agent to generate and execute SQL
            logger.info("Using SQL agent to generate and execute SQL")
            agent_reasoning.append("Using SQL agent to generate and execute SQL query")
            sql_start_time = time.time()
            sql_result = self.agents["sql_agent"].execute({
                "query": query,
                "execute": True,
                "max_rows": 100
            })
            sql_time = time.time() - sql_start_time
            logger.info(f"SQL generation and execution completed in {sql_time:.2f}s")
            
            # Check for errors in SQL generation
            if "error" in sql_result:
                logger.error(f"Error in SQL generation: {sql_result['error']}")
                agent_reasoning.append(f"Error in SQL generation: {sql_result['error']}")
                return {
                    "error": f"Error in SQL generation: {sql_result['error']}",
                    "agent_reasoning": agent_reasoning
                }
            
            # Add SQL generation details to reasoning
            sql_query = sql_result.get("sql_query", "")
            if sql_query:
                agent_reasoning.append(f"Generated SQL query: {sql_query}")
            
            execution_results = sql_result.get("execution_results", {})
            if execution_results:
                row_count = execution_results.get("row_count", 0)
                agent_reasoning.append(f"Executed SQL query successfully, retrieved {row_count} rows")
            
            # Log the successful query processing
            total_time = dba_time + sql_time
            logger.info(f"SQL query processing completed in {total_time:.2f}s")
            conversation_logger.log_trino_ai_processing("sql_query_processing_complete", {
                "dba_time": dba_time,
                "sql_time": sql_time,
                "total_time": total_time
            })
            
            # Log the agent reasoning
            conversation_logger.log_agent_reasoning("agent_orchestrator", [
                {"step": i+1, "description": step} for i, step in enumerate(agent_reasoning)
            ])
            
            return {
                "query": query,
                "query_type": "sql",
                "sql_query": sql_result.get("sql_query", ""),
                "execution_results": sql_result.get("execution_results", {}),
                "dba_analysis": dba_analysis,
                "schema_context": schema_context,
                "agent_reasoning": agent_reasoning,
                "explanation": sql_result.get("explanation", ""),
                "is_data_query": True
            }
            
        except Exception as e:
            logger.error(f"Error processing SQL query: {str(e)}")
            conversation_logger.log_error("agent_orchestrator", f"Error processing SQL query: {str(e)}")
            
            agent_reasoning.append(f"Error processing SQL query: {str(e)}")
            
            return {
                "error": f"Error processing SQL query: {str(e)}",
                "agent_reasoning": agent_reasoning
            }
    
    def _process_knowledge_query(self, query: str) -> Dict[str, Any]:
        """
        Process a knowledge query
        
        Args:
            query: The natural language query
            
        Returns:
            The results of processing the knowledge query
        """
        logger.info(f"Processing knowledge query: {query}")
        conversation_logger.log_trino_ai_processing("knowledge_query_processing_start", {
            "query": query
        })
        
        # Track reasoning steps
        agent_reasoning = []
        agent_reasoning.append(f"Identified query as a knowledge query: '{query}'")
        
        try:
            # Use the Knowledge agent to answer the query
            logger.info("Using Knowledge agent to answer query")
            agent_reasoning.append("Using Knowledge agent to retrieve and synthesize information")
            start_time = time.time()
            knowledge_result = self.agents["knowledge_agent"].execute({
                "query": query,
                "max_results": 5
            })
            knowledge_time = time.time() - start_time
            logger.info(f"Knowledge query processing completed in {knowledge_time:.2f}s")
            
            # Check for errors in knowledge retrieval
            if "error" in knowledge_result:
                logger.error(f"Error in knowledge retrieval: {knowledge_result['error']}")
                agent_reasoning.append(f"Error in knowledge retrieval: {knowledge_result['error']}")
                return {
                    "error": f"Error in knowledge retrieval: {knowledge_result['error']}",
                    "agent_reasoning": agent_reasoning
                }
            
            # Add knowledge retrieval details to reasoning
            knowledge = knowledge_result.get("knowledge", "")
            sources = knowledge_result.get("sources", [])
            
            if knowledge:
                agent_reasoning.append("Retrieved and synthesized knowledge from available sources")
            
            if sources:
                agent_reasoning.append(f"Used sources: {', '.join(sources)}")
            
            # Log the successful query processing
            conversation_logger.log_trino_ai_processing("knowledge_query_processing_complete", {
                "knowledge_time": knowledge_time
            })
            
            # Log the agent reasoning
            conversation_logger.log_agent_reasoning("agent_orchestrator", [
                {"step": i+1, "description": step} for i, step in enumerate(agent_reasoning)
            ])
            
            return {
                "query": query,
                "query_type": "knowledge",
                "knowledge": knowledge_result.get("knowledge", ""),
                "sources": knowledge_result.get("sources", []),
                "agent_reasoning": agent_reasoning,
                "response": knowledge_result.get("knowledge", ""),
                "explanation": "This is a knowledge query that was answered using available information sources.",
                "is_data_query": False
            }
            
        except Exception as e:
            logger.error(f"Error processing knowledge query: {str(e)}")
            conversation_logger.log_error("agent_orchestrator", f"Error processing knowledge query: {str(e)}")
            
            agent_reasoning.append(f"Error processing knowledge query: {str(e)}")
            
            return {
                "error": f"Error processing knowledge query: {str(e)}",
                "agent_reasoning": agent_reasoning
            }
    
    def _get_schema_context(self) -> str:
        """
        Get the database schema context
        
        Returns:
            The database schema context as a string
        """
        # This is a simplified implementation - in a real system, you would retrieve the schema from the database
        # For now, we'll use a hardcoded schema
        
        return """
        Table: customers
        Columns: customer_id (int), name (varchar), email (varchar), signup_date (date), status (varchar)
        
        Table: orders
        Columns: order_id (int), customer_id (int), order_date (date), total_amount (decimal), status (varchar)
        
        Table: order_items
        Columns: item_id (int), order_id (int), product_id (int), quantity (int), price (decimal)
        
        Table: products
        Columns: product_id (int), name (varchar), category (varchar), price (decimal), inventory (int)
        """ 