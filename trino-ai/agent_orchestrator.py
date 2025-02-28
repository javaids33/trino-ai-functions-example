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
from tools.metadata_tools import GetSchemaContextTool, RefreshMetadataTool
from tools.sql_tools import ValidateSQLTool, ExecuteSQLTool
from colorama import Fore
from conversation_logger import conversation_logger
from trino_client import TrinoClient

logger = logging.getLogger(__name__)

class AgentOrchestrator:
    """Orchestrates the multi-agent workflow for NLQ to SQL conversion"""
    
    def __init__(self, ollama_client: OllamaClient):
        self.ollama_client = ollama_client
        
        # Initialize tools
        logger.info(f"{Fore.CYAN}Initializing tools...{Fore.RESET}")
        
        # Create Trino client
        trino_client = TrinoClient(
            host=os.getenv("TRINO_HOST", "trino"),
            port=int(os.getenv("TRINO_PORT", "8080")),
            user=os.getenv("TRINO_USER", "admin"),
            catalog=os.getenv("TRINO_CATALOG", "iceberg"),
            schema=os.getenv("TRINO_SCHEMA", "iceberg")
        )
        
        self.tools = {
            "get_schema_context": GetSchemaContextTool(),
            "refresh_metadata": RefreshMetadataTool(),
            "validate_sql": ValidateSQLTool(trino_client=trino_client),
            "execute_sql": ExecuteSQLTool()
        }
        
        # Initialize agents
        logger.info(f"{Fore.CYAN}Initializing agents...{Fore.RESET}")
        self.agents = {
            "dba": DBAAgent(ollama_client=ollama_client, tools=self.tools),
            "sql": SQLAgent(ollama_client=ollama_client, tools=self.tools)
        }
        
        logger.info(f"{Fore.GREEN}Agent orchestrator initialized{Fore.RESET}")
        
    def process_natural_language_query(self, nlq: str, model: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a natural language query
        
        Args:
            nlq: The natural language query to process
            
        Returns:
            A dictionary containing the processed query
        """
        start_time = time.time()
        
        # Log conversation flow
        conversation_logger.log_trino_to_trino_ai("nlq", nlq)
        
        logger.info(f"{Fore.GREEN}Processing NLQ: {nlq}{Fore.RESET}")
        
        # Step 1: Determine if this is a data query or a general knowledge query
        is_data_query = self._determine_query_type(nlq)
        
        if not is_data_query:
            # Use the Knowledge Agent for general knowledge queries
            from agents.knowledge_agent import KnowledgeAgent
            knowledge_agent = KnowledgeAgent(ollama_client=self.ollama_client)
            
            # Log the decision to use the Knowledge Agent
            logger.info(f"{Fore.YELLOW}Using Knowledge Agent for general knowledge query{Fore.RESET}")
            conversation_logger.log_trino_ai_processing("orchestrator_knowledge_agent", {
                "decision": "Using Knowledge Agent for general knowledge query",
                "query": nlq
            })
            
            # Execute the Knowledge Agent
            knowledge_result = knowledge_agent.execute({"query": nlq})
            
            if "error" in knowledge_result:
                logger.error(f"{Fore.RED}Error in Knowledge Agent: {knowledge_result['error']}{Fore.RESET}")
                conversation_logger.log_error("orchestrator", f"Error in Knowledge Agent: {knowledge_result['error']}")
                return {
                    "error": knowledge_result["error"],
                    "stage": "knowledge_agent"
                }
            
            response = knowledge_result["answer"]
            
            elapsed_time = time.time() - start_time
            result = {
                "natural_language_query": nlq,
                "response": response,
                "is_data_query": False,
                "processing_time": f"{elapsed_time:.2f}s",
                "agent_reasoning": "This query was identified as a general knowledge question and processed by the Knowledge Agent.",
                "metadata": {
                    "query_type": "general_knowledge",
                    "processing_time": f"{elapsed_time:.2f}s",
                    "agent": "Knowledge Agent"
                }
            }
            
            logger.info(f"{Fore.GREEN}General knowledge query processed in {elapsed_time:.2f}s{Fore.RESET}")
            conversation_logger.log_trino_ai_to_trino("knowledge_response", {
                "natural_language_query": nlq,
                "response": response,
                "processing_time": f"{elapsed_time:.2f}s"
            })
            
            # Log conversation summary
            logger.info(conversation_logger.get_conversation_summary())
            
            return result
        
        # For data queries, continue with the existing flow
        logger.info(f"{Fore.YELLOW}Using SQL generation pipeline for data query{Fore.RESET}")
        conversation_logger.log_trino_ai_processing("orchestrator_data_query", {
            "decision": "Using SQL generation pipeline for data query",
            "query": nlq
        })
        
        # Step 2: Get schema context
        logger.info(f"{Fore.YELLOW}Step 2: Retrieving schema context{Fore.RESET}")
        conversation_logger.log_trino_ai_processing("orchestrator_step2", {"step": "Retrieving schema context"})
        context_tool = self.tools["get_schema_context"]
        context_result = context_tool.execute({"query": nlq})
        
        if "error" in context_result:
            logger.error(f"{Fore.RED}Error retrieving schema context: {context_result['error']}{Fore.RESET}")
            conversation_logger.log_error("orchestrator", f"Error retrieving schema context: {context_result['error']}")
            return {
                "error": context_result["error"],
                "stage": "context_retrieval"
            }
            
        schema_context = context_result["context"]
        logger.info(f"{Fore.GREEN}Retrieved schema context with {context_result['table_count']} tables{Fore.RESET}")
        conversation_logger.log_trino_ai_processing("orchestrator_context_retrieved", {
            "table_count": context_result["table_count"],
            "schema_info_status": context_result["schema_info_status"]
        })
        
        # Step 3: Run DBA analysis
        logger.info(f"{Fore.YELLOW}Step 3: Running DBA analysis{Fore.RESET}")
        conversation_logger.log_trino_ai_processing("orchestrator_step3", {"step": "Running DBA analysis"})
        dba_agent = self.agents["dba"]
        dba_result = dba_agent.execute({
            "query": nlq,
            "schema_context": schema_context
        })
        
        if "error" in dba_result:
            logger.error(f"{Fore.RED}Error in DBA analysis: {dba_result['error']}{Fore.RESET}")
            conversation_logger.log_error("orchestrator", f"Error in DBA analysis: {dba_result['error']}")
            return {
                "error": dba_result["error"],
                "stage": "dba_analysis",
                "schema_context": schema_context
            }
            
        logger.info(f"{Fore.GREEN}DBA analysis complete: identified {len(dba_result.get('tables', []))} tables{Fore.RESET}")
        conversation_logger.log_trino_ai_processing("orchestrator_dba_complete", {
            "tables_count": len(dba_result.get("tables", [])),
            "columns_count": len(dba_result.get("columns", [])),
            "joins_count": len(dba_result.get("joins", [])),
            "filters_count": len(dba_result.get("filters", [])),
            "aggregations_count": len(dba_result.get("aggregations", []))
        })
        
        # Step 4: Generate SQL
        logger.info(f"{Fore.YELLOW}Step 4: Generating SQL{Fore.RESET}")
        conversation_logger.log_trino_ai_processing("orchestrator_step4", {"step": "Generating SQL"})
        sql_agent = self.agents["sql"]
        sql_result = sql_agent.execute({
            "query": nlq,
            "schema_context": schema_context,
            "dba_analysis": dba_result
        })
        
        if "error" in sql_result:
            logger.error(f"{Fore.RED}Error generating SQL: {sql_result['error']}{Fore.RESET}")
            conversation_logger.log_error("orchestrator", f"Error generating SQL: {sql_result['error']}")
            return {
                "error": sql_result["error"],
                "stage": "sql_generation",
                "schema_context": schema_context,
                "dba_analysis": dba_result
            }
            
        sql = sql_result["sql"]
        is_valid = sql_result["is_valid"]
        error_message = sql_result["error_message"]
        
        logger.info(f"{Fore.GREEN}Initial SQL generated, valid: {is_valid}{Fore.RESET}")
        conversation_logger.log_trino_ai_processing("orchestrator_sql_generated", {
            "sql_length": len(sql),
            "is_valid": is_valid,
            "error_message": error_message
        })
        
        # Step 5: Refine SQL if needed
        refinement_steps = 0
        max_refinements = 2
        
        while not is_valid and refinement_steps < max_refinements:
            logger.info(f"{Fore.YELLOW}Step 5.{refinement_steps + 1}: Refining SQL, error: {error_message}{Fore.RESET}")
            conversation_logger.log_trino_ai_processing(f"orchestrator_step5_{refinement_steps + 1}", {
                "step": f"Refining SQL (attempt {refinement_steps + 1})",
                "error_message": error_message
            })
            
            refinement_steps += 1
            
            refinement_result = sql_agent.refine_sql({
                "query": nlq,
                "sql": sql,
                "error_message": error_message,
                "schema_context": schema_context,
                "dba_analysis": dba_result
            })
            
            sql = refinement_result["sql"]
            is_valid = refinement_result["is_valid"]
            error_message = refinement_result["error_message"]
            
            logger.info(f"{Fore.GREEN}Refined SQL (step {refinement_steps}), valid: {is_valid}{Fore.RESET}")
            conversation_logger.log_trino_ai_processing(f"orchestrator_sql_refined_{refinement_steps}", {
                "sql_length": len(sql),
                "is_valid": is_valid,
                "error_message": error_message
            })
            
        # Step 6: Prepare final response
        elapsed_time = time.time() - start_time
        
        # Collect agent reasoning information
        agent_reasoning = f"""
        Query identified as a data query requiring SQL generation.
        DBA Agent identified {len(dba_result.get('tables', []))} tables and {len(dba_result.get('columns', []))} columns.
        SQL Agent generated a query with {len(sql.split())} tokens.
        SQL validation: {'Successful' if is_valid else 'Failed with errors'}.
        Refinement steps: {refinement_steps}
        """
        
        # Format the result to work with the existing ai_translate function
        result = {
            "natural_language_query": nlq,
            "sql_query": sql,
            "explanation": sql_result.get("explanation", ""),
            "is_valid": is_valid if isinstance(is_valid, bool) else True,  # Ensure is_valid is always a boolean
            "refinement_steps": refinement_steps,
            "dba_analysis": dba_result,
            "schema_context": schema_context,
            "processing_time": elapsed_time,
            "is_data_query": True,
            "agent_reasoning": agent_reasoning,
            "metadata": {
                "query_type": "data_query",
                "tables_used": dba_result.get("tables", []),
                "columns_used": dba_result.get("columns", []),
                "processing_time": f"{elapsed_time:.2f}s",
                "refinement_steps": refinement_steps
            }
        }
        
        if not is_valid and error_message:
            result["error_message"] = error_message
            
        logger.info(f"{Fore.GREEN}NLQ processing complete in {elapsed_time:.2f}s, valid: {is_valid}{Fore.RESET}")
        conversation_logger.log_trino_ai_to_trino("ai_gen", {
            "natural_language_query": nlq,
            "sql_query": sql,
            "is_valid": is_valid,
            "processing_time": f"{elapsed_time:.2f}s"
        })
        
        # Log conversation summary
        logger.info(conversation_logger.get_conversation_summary())
        
        return result
        
    def _determine_query_type(self, nlq: str) -> bool:
        """
        Determine if a query is a data query that needs SQL generation
        or a general knowledge query
        
        Args:
            nlq: The natural language query to analyze
            
        Returns:
            True if this is a data query, False if it's a general knowledge query
        """
        # Log the determination step
        logger.info(f"{Fore.BLUE}Determining query type for: {nlq}{Fore.RESET}")
        conversation_logger.log_trino_ai_processing("query_type_determination", {
            "query": nlq
        })
        
        # Use a simple prompt to determine the query type
        prompt = f"""
        Analyze the following query and determine if it requires database access or if it's a general knowledge question:
        
        Query: {nlq}
        
        If the query requires looking up specific data, statistics, or records from a database, respond with "DATA_QUERY".
        If the query is asking for general knowledge, facts, or information that doesn't require current database access, respond with "KNOWLEDGE_QUERY".
        
        Respond with only "DATA_QUERY" or "KNOWLEDGE_QUERY", nothing else.
        """
        
        # Get the response from Ollama
        messages = [{"role": "user", "content": prompt}]
        response = self.ollama_client.generate_response(messages, agent_name="Query Classifier")
        
        # Clean up the response and determine the result
        response = response.strip().upper()
        
        # Log the decision
        is_data_query = "DATA_QUERY" in response
        logger.info(f"{Fore.BLUE}Query classified as: {'DATA_QUERY' if is_data_query else 'KNOWLEDGE_QUERY'}{Fore.RESET}")
        conversation_logger.log_trino_ai_processing("query_type_result", {
            "query": nlq,
            "classification": "DATA_QUERY" if is_data_query else "KNOWLEDGE_QUERY",
            "raw_response": response
        })
        
        return is_data_query 