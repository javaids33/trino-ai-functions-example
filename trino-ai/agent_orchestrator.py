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
from tools.intent_verification_tool import QueryIntentVerificationTool
from colorama import Fore
from conversation_logger import conversation_logger
from trino_client import TrinoClient
from workflow_context import WorkflowContext

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
            "execute_sql": ExecuteSQLTool(),
            "verify_intent": QueryIntentVerificationTool(ollama_client=ollama_client)
        }
        
        # Initialize agents
        logger.info(f"{Fore.CYAN}Initializing agents...{Fore.RESET}")
        self.agents = {
            "dba": DBAAgent(ollama_client=ollama_client, tools=self.tools),
            "sql": SQLAgent(ollama_client=ollama_client, tools=self.tools)
        }
        
        # Set maximum number of refinement attempts
        self.max_refinements = 2
        
        logger.info(f"{Fore.GREEN}Agent orchestrator initialized{Fore.RESET}")
        
    def process_natural_language_query(self, nlq: str, model: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a natural language query
        
        Args:
            nlq: The natural language query to process
            model: Optional model name to use for processing
            
        Returns:
            A dictionary containing the processed query
        """
        start_time = time.time()
        
        # Initialize workflow context
        workflow_context = WorkflowContext()
        workflow_context.set_query(nlq)
        
        # Set the current context in the conversation logger
        conversation_logger.set_current_context(workflow_context)
        
        # Log conversation flow
        conversation_logger.log_trino_to_trino_ai("nlq", {"query": nlq})
        
        logger.info(f"{Fore.GREEN}Processing NLQ: {nlq}{Fore.RESET}")
        
        # Step 1: Determine if this is a data query or a general knowledge query
        is_data_query = self._determine_query_type(nlq, workflow_context)
        workflow_context.set_data_query_status(is_data_query)
        
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
            
            # Add decision point to workflow context
            workflow_context.add_decision_point(
                "orchestrator", 
                "use_knowledge_agent", 
                "Query classified as a general knowledge question that doesn't require database access"
            )
            
            # Execute the Knowledge Agent with workflow context
            knowledge_result = knowledge_agent.execute({"query": nlq}, workflow_context)
            
            # Return the knowledge response
            response = {
                "query": nlq,
                "is_data_query": False,
                "response": knowledge_result.get("response", ""),
                "processing_time": time.time() - start_time
            }
            
            # Log the response
            conversation_logger.log_trino_ai_to_trino("knowledge_response", response)
            
            return response
        
        # Step 2: Get schema context for the query
        logger.info(f"{Fore.YELLOW}Step 2: Getting schema context{Fore.RESET}")
        conversation_logger.log_trino_ai_processing("orchestrator_step2", {"step": "Getting schema context"})
        
        schema_context_tool = self.tools["get_schema_context"]
        schema_context_result = schema_context_tool.execute({"query": nlq})
        schema_context = schema_context_result.get("schema_context", "")
        
        # Add schema context to workflow context
        workflow_context.set_schema_context(schema_context)
        workflow_context.add_metadata("schema_context", {
            "length": len(schema_context),
            "tables": schema_context_result.get("tables", [])
        })
        
        # Step 3: Use DBA Agent to analyze the query
        logger.info(f"{Fore.YELLOW}Step 3: Analyzing query with DBA Agent{Fore.RESET}")
        conversation_logger.log_trino_ai_processing("orchestrator_step3", {"step": "Analyzing query with DBA Agent"})
        
        dba_agent = self.agents["dba"]
        dba_result = dba_agent.execute({
            "query": nlq,
            "schema_context": schema_context
        }, workflow_context)
        
        if "error" in dba_result:
            logger.error(f"{Fore.RED}Error from DBA Agent: {dba_result['error']}{Fore.RESET}")
            conversation_logger.log_error("orchestrator", f"DBA Agent error: {dba_result['error']}")
            return {
                "query": nlq,
                "is_data_query": True,
                "error": f"Error analyzing query: {dba_result['error']}",
                "processing_time": time.time() - start_time
            }
        
        # Add DBA analysis to workflow context
        workflow_context.add_metadata("dba_analysis", dba_result)
        
        # Step 4: Use SQL Agent to generate SQL
        logger.info(f"{Fore.YELLOW}Step 4: Generating SQL with SQL Agent{Fore.RESET}")
        conversation_logger.log_trino_ai_processing("orchestrator_step4", {"step": "Generating SQL with SQL Agent"})
        
        sql_agent = self.agents["sql"]
        sql_result = sql_agent.execute({
            "query": nlq,
            "schema_context": schema_context,
            "dba_analysis": dba_result
        }, workflow_context)
        
        if "error" in sql_result:
            logger.error(f"{Fore.RED}Error from SQL Agent: {sql_result['error']}{Fore.RESET}")
            conversation_logger.log_error("orchestrator", f"SQL Agent error: {sql_result['error']}")
            return {
                "query": nlq,
                "is_data_query": True,
                "error": f"Error generating SQL: {sql_result['error']}",
                "processing_time": time.time() - start_time
            }
        
        sql = sql_result["sql"]
        is_valid = sql_result["is_valid"]
        error_message = sql_result["error_message"]
        
        # Step 5: Validate the SQL
        logger.info(f"{Fore.YELLOW}Step 5: Validating SQL{Fore.RESET}")
        conversation_logger.log_trino_ai_processing("orchestrator_step5", {"step": "Validating SQL"})
        
        # Track refinement attempts
        refinement_steps = 0
        max_refinements = self.max_refinements
        
        # If SQL is not valid, try to refine it
        while not is_valid and refinement_steps < max_refinements:
            logger.info(f"{Fore.YELLOW}Step 5.{refinement_steps + 1}: Refining SQL (attempt {refinement_steps + 1}/{max_refinements}){Fore.RESET}")
            conversation_logger.log_trino_ai_processing(f"orchestrator_step5_{refinement_steps + 1}", {
                "step": f"Refining SQL (attempt {refinement_steps + 1}/{max_refinements})",
                "error_message": error_message
            })
            
            workflow_context.add_decision_point(
                "orchestrator",
                "sql_refinement",
                f"SQL validation failed: {error_message}"
            )
            
            refinement_steps += 1
            
            refinement_result = sql_agent.refine_sql({
                "query": nlq,
                "sql": sql,
                "error_message": error_message,
                "schema_context": schema_context,
                "dba_analysis": dba_result
            }, workflow_context)
            
            sql = refinement_result["sql"]
            is_valid = refinement_result["is_valid"]
            error_message = refinement_result["error_message"]
            
            # Update SQL in workflow context
            workflow_context.set_sql(sql)
            workflow_context.add_metadata(f"sql_refinement_{refinement_steps}", {
                "is_valid": is_valid,
                "error_message": error_message if not is_valid else "",
                "refinement_attempt": refinement_steps
            })
        
        # Step 6: Verify query intent if SQL is valid
        if is_valid:
            logger.info(f"{Fore.YELLOW}Step 6: Verifying query intent{Fore.RESET}")
            conversation_logger.log_trino_ai_processing("orchestrator_step6", {"step": "Verifying query intent"})
            
            intent_verification_result = self.tools["verify_intent"].execute({
                "query": nlq,
                "sql": sql,
                "schema_context": schema_context,
                "dba_analysis": dba_result
            })
            
            matches_intent = intent_verification_result.get("matches_intent", False)
            missing_aspects = intent_verification_result.get("missing_aspects", [])
            suggestions = intent_verification_result.get("suggestions", [])
            
            # Add intent verification results to workflow context
            workflow_context.add_metadata("intent_verification", {
                "matches_intent": matches_intent,
                "missing_aspects": missing_aspects,
                "suggestions": suggestions
            })
            
            logger.info(f"{Fore.GREEN}Intent verification: matches_intent={matches_intent}{Fore.RESET}")
            conversation_logger.log_trino_ai_processing("orchestrator_intent_verified", {
                "matches_intent": matches_intent,
                "missing_aspects": missing_aspects,
                "suggestions": suggestions
            })
            
            # If the SQL doesn't match the intent, try to refine it one more time
            if not matches_intent and refinement_steps < max_refinements:
                logger.info(f"{Fore.YELLOW}Step 6.1: Refining SQL based on intent verification{Fore.RESET}")
                conversation_logger.log_trino_ai_processing("orchestrator_step6_1", {
                    "step": "Refining SQL based on intent verification",
                    "missing_aspects": missing_aspects
                })
                
                workflow_context.add_decision_point(
                    "orchestrator",
                    "sql_intent_refinement",
                    f"SQL does not fully match query intent. Missing: {', '.join(missing_aspects)}"
                )
                
                refinement_steps += 1
                
                # Prepare a special refinement input that includes intent verification feedback
                refinement_input = {
                    "query": nlq,
                    "sql": sql,
                    "error_message": f"SQL does not fully match query intent. Missing aspects: {', '.join(missing_aspects)}. Suggestions: {', '.join(suggestions)}",
                    "schema_context": schema_context,
                    "dba_analysis": dba_result,
                    "intent_feedback": intent_verification_result
                }
                
                refinement_result = sql_agent.refine_sql(refinement_input, workflow_context)
                
                sql = refinement_result["sql"]
                is_valid = refinement_result["is_valid"]
                error_message = refinement_result["error_message"]
                
                # Update SQL in workflow context
                workflow_context.set_sql(sql)
                workflow_context.add_metadata("sql_intent_refinement", {
                    "is_valid": is_valid,
                    "error_message": error_message if not is_valid else "",
                    "refinement_attempt": refinement_steps
                })
        
        # Step 7: Execute the SQL if valid
        if is_valid:
            logger.info(f"{Fore.YELLOW}Step 7: Executing SQL{Fore.RESET}")
            conversation_logger.log_trino_ai_processing("orchestrator_step7", {"step": "Executing SQL"})
            
            execute_tool = self.tools["execute_sql"]
            execution_result = execute_tool.execute({"sql": sql})
            
            if "error" in execution_result:
                logger.error(f"{Fore.RED}Error executing SQL: {execution_result['error']}{Fore.RESET}")
                conversation_logger.log_error("orchestrator", f"SQL execution error: {execution_result['error']}")
                
                # Add execution error to workflow context
                workflow_context.add_metadata("sql_execution", {
                    "success": False,
                    "error": execution_result["error"]
                })
                
                return {
                    "query": nlq,
                    "is_data_query": True,
                    "sql": sql,
                    "error": f"Error executing SQL: {execution_result['error']}",
                    "processing_time": time.time() - start_time
                }
            
            # Add execution result to workflow context
            workflow_context.add_metadata("sql_execution", {
                "success": True,
                "row_count": len(execution_result.get("results", [])),
                "column_count": len(execution_result.get("columns", []))
            })
            
            # Prepare the final response
            response = {
                "query": nlq,
                "is_data_query": True,
                "sql": sql,
                "results": execution_result.get("results", []),
                "columns": execution_result.get("columns", []),
                "processing_time": time.time() - start_time
            }
            
            # Log the response
            conversation_logger.log_trino_ai_to_trino("sql_response", {
                "query": nlq,
                "sql": sql,
                "result_count": len(execution_result.get("results", []))
            })
            
            # Log the NL2SQL conversion
            conversation_logger.log_nl2sql_conversion(
                query=nlq,
                sql=sql,
                metadata={
                    "is_valid": is_valid,
                    "processing_time": time.time() - start_time,
                    "refinement_steps": refinement_steps
                }
            )
            
            return response
        else:
            # Return error if SQL is still not valid after refinement attempts
            logger.error(f"{Fore.RED}Failed to generate valid SQL after {refinement_steps} refinement attempts{Fore.RESET}")
            conversation_logger.log_error("orchestrator", f"Failed to generate valid SQL: {error_message}")
            
            return {
                "query": nlq,
                "is_data_query": True,
                "invalid_sql": sql,
                "error": f"Failed to generate valid SQL: {error_message}",
                "processing_time": time.time() - start_time
            }
        
    def _determine_query_type(self, nlq: str, workflow_context: Optional[WorkflowContext] = None) -> bool:
        """
        Determine if a query is a data query that needs SQL generation
        or a general knowledge query
        
        Args:
            nlq: The natural language query to analyze
            workflow_context: Optional workflow context to update
            
        Returns:
            True if this is a data query, False if it's a general knowledge query
        """
        # Log the determination step
        logger.info(f"{Fore.BLUE}Determining query type for: {nlq}{Fore.RESET}")
        conversation_logger.log_trino_ai_processing("query_type_determination", {
            "query": nlq
        })
        
        # Add decision point to workflow context if provided
        if workflow_context:
            workflow_context.add_decision_point(
                "orchestrator",
                "query_type_determination",
                f"Analyzing query to determine if it requires database access: '{nlq}'"
            )
        
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
        response = self.ollama_client.generate_response(prompt=messages,system_prompt="Query Classifier")
        
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
        
        # Add classification result to workflow context if provided
        if workflow_context:
            workflow_context.add_metadata("query_classification", {
                "is_data_query": is_data_query,
                "raw_response": response
            })
            workflow_context.add_decision_point(
                "Query Classifier",
                "query_classification",
                f"Query classified as {'DATA_QUERY' if is_data_query else 'KNOWLEDGE_QUERY'}"
            )
        
        return is_data_query 