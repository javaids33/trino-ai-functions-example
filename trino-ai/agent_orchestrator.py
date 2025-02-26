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
        """Process a natural language query through the full agent pipeline"""
        logger.info(f"{Fore.BLUE}Processing NLQ: {nlq}{Fore.RESET}")
        conversation_logger.log_trino_request("ai_gen", nlq)
        start_time = time.time()
        
        # Step 1: Get schema context
        logger.info(f"{Fore.YELLOW}Step 1: Retrieving schema context{Fore.RESET}")
        conversation_logger.log_trino_ai_processing("orchestrator_step1", {"step": "Retrieving schema context"})
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
        
        # Step 2: Run DBA analysis
        logger.info(f"{Fore.YELLOW}Step 2: Running DBA analysis{Fore.RESET}")
        conversation_logger.log_trino_ai_processing("orchestrator_step2", {"step": "Running DBA analysis"})
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
        
        # Step 3: Generate SQL
        logger.info(f"{Fore.YELLOW}Step 3: Generating SQL{Fore.RESET}")
        conversation_logger.log_trino_ai_processing("orchestrator_step3", {"step": "Generating SQL"})
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
        
        # Step 4: Refine SQL if needed
        refinement_steps = 0
        max_refinements = 2
        
        while not is_valid and refinement_steps < max_refinements:
            logger.info(f"{Fore.YELLOW}Step 4.{refinement_steps + 1}: Refining SQL, error: {error_message}{Fore.RESET}")
            conversation_logger.log_trino_ai_processing(f"orchestrator_step4_{refinement_steps + 1}", {
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
            
        # Step 5: Prepare final response
        elapsed_time = time.time() - start_time
        
        # Format the result to work with the existing ai_translate function
        result = {
            "natural_language_query": nlq,
            "sql_query": sql,
            "explanation": sql_result.get("explanation", ""),
            "is_valid": is_valid if isinstance(is_valid, bool) else True,  # Ensure is_valid is always a boolean
            "refinement_steps": refinement_steps,
            "dba_analysis": dba_result,
            "schema_context": schema_context,
            "processing_time": elapsed_time
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