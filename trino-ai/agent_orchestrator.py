from colorama import Fore, Style, init
import logging
import json
import time
import uuid
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
import re
import traceback

# Local imports
from workflow_context import WorkflowContext
from conversation_logger import ConversationLogger
from ollama_client import OllamaClient
from trino_client import TrinoClient
from trino_executor import TrinoExecutor
from embeddings import EmbeddingService
from exemplars_manager import ExemplarsManager
from monitoring.monitoring_service import MonitoringService

# Agent imports - only import modules that exist
from agents.base_agent import Agent
from agents.sql_agent import SQLAgent
from agents.knowledge_agent import KnowledgeAgent
from agents.dba_agent import DBAAgent
from agents.translation_agent import TranslationAgent

# Initialize colorama
init(autoreset=True)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AgentOrchestrator:
    def __init__(self, ollama_client=None, tools=None, exemplars_path="exemplars"):
        """
        Initialize the agent orchestrator
        
        Args:
            ollama_client: An Ollama client for LLM calls
            tools: A dictionary of tools
            exemplars_path: Path to the exemplars directory
        """
        self.ollama_client = ollama_client
        self.tools = tools or {}
        self.workflow_context = None
        
        # Initialize exemplars manager
        self.exemplars_manager = ExemplarsManager(exemplars_path)
        
        # Fix: Pass the file_path argument to load_exemplars
        self.exemplars_manager.load_exemplars(f"{exemplars_path}/exemplars.json")
        
        # Fix: Remove the exemplars_manager argument
        self.dba_agent = DBAAgent(
            name="DBA Agent",
            description="Analyzes natural language queries to identify tables, columns, joins, filters, and aggregations",
            ollama_client=ollama_client
        )
        
        # Initialize monitoring service
        self.monitoring_service = MonitoringService()
        
        # Initialize agents - only use agents that exist
        self.knowledge_agent = KnowledgeAgent(ollama_client)
        self.sql_agent = SQLAgent(ollama_client, self.exemplars_manager)
        self.translation_agent = TranslationAgent(ollama_client)
        
        # Initialize conversation logger
        self.conversation_logger = ConversationLogger()
        
        # Track active queries
        self.active_queries = {}
        
    def process_nlq(self, nlq: str, model: str = "llama3.2", use_progressive_build: bool = False, use_multi_candidate: bool = False) -> Dict[str, Any]:
        """
        Process a natural language query and return the result.
        
        Args:
            nlq: The natural language query to process
            model: The model to use for processing
            use_progressive_build: Whether to use the progressive build approach
            use_multi_candidate: Whether to use the multi-candidate approach
            
        Returns:
            A dictionary containing the result of the query
        """
        # Generate a unique ID for this query
        query_id = str(uuid.uuid4())
        
        # Add to active queries
        self.active_queries[query_id] = {
            "id": query_id,
            "timestamp": datetime.now().isoformat(),
            "natural_language": nlq,
            "status": "processing",
            "model": model
        }
        
        try:
            # Initialize workflow context
            workflow_context = WorkflowContext()
            workflow_context.set_query(nlq)
            
            # Set the current context in the conversation logger
            conversation_logger = ConversationLogger()
            conversation_logger.set_current_context(workflow_context)
            
            # Log conversation flow
            conversation_logger.log_trino_to_trino_ai("nlq", {"query": nlq})
            
            logger.info(f"{Fore.GREEN}Processing NLQ: {nlq}{Fore.RESET}")
            logger.info(f"{Fore.GREEN}SQL Generation Strategy: {'Progressive Build' if use_progressive_build else 'Multi-Candidate' if use_multi_candidate else 'Standard'}{Fore.RESET}")
            
            # Step 1: Determine if this is a data query or a general knowledge query
            monitoring_service = MonitoringService()
            monitoring_service.log_agent_activity("classifier", {
                "action": "classify_query",
                "details": {"query": nlq},
                "query_id": query_id
            })
            
            is_data_query = self._determine_query_type(nlq, workflow_context)
            workflow_context.set_data_query_status(is_data_query)
            
            if not is_data_query:
                # Use the Knowledge Agent for general knowledge queries
                logger.info(f"{Fore.YELLOW}Using Knowledge Agent for general knowledge query{Fore.RESET}")
                monitoring_service.log_agent_activity("knowledge", {
                    "action": "process_knowledge_query",
                    "details": {"query": nlq},
                    "query_id": query_id
                })
                
                result = self.knowledge_agent.answer_question(nlq, workflow_context)
                workflow_context.set_final_answer(result)
                
                # Update active query status
                self.active_queries[query_id]["status"] = "completed"
                self.active_queries[query_id]["result"] = result
                
                # Log the final answer
                conversation_logger.log_trino_ai_to_trino("answer", {"answer": result})
                
                return {
                    "query_id": query_id,
                    "query": nlq,
                    "answer": result,
                    "is_data_query": False,
                    "sql": None,
                    "execution_result": None,
                    "execution_time": None
                }
            
            # For data queries, proceed with SQL generation and execution
            logger.info(f"{Fore.BLUE}Using SQL Agent for data query{Fore.RESET}")
            
            # Step 2: Generate SQL
            start_time = time.time()
            
            if use_progressive_build:
                # Use progressive build approach
                sql_result = self._generate_sql_progressive(nlq, workflow_context, query_id)
            elif use_multi_candidate:
                # Use multi-candidate approach
                sql_result = self._generate_sql_multi_candidate(nlq, workflow_context, query_id)
            else:
                # Use standard approach
                sql_result = self._generate_sql_standard(nlq, workflow_context, query_id)
            
            if "error" in sql_result:
                # Handle SQL generation error
                error_message = sql_result["error"]
                logger.error(f"{Fore.RED}SQL Generation Error: {error_message}{Fore.RESET}")
                
                # Update active query status
                self.active_queries[query_id]["status"] = "error"
                self.active_queries[query_id]["error"] = error_message
                
                # Log the error
                conversation_logger.log_trino_ai_to_trino("error", {"error": error_message})
                
                return {
                    "query_id": query_id,
                    "query": nlq,
                    "error": error_message,
                    "is_data_query": True
                }
            
            sql = sql_result["sql"]
            workflow_context.set_generated_sql(sql)
            
            # Step 3: Execute SQL
            logger.info(f"{Fore.CYAN}Executing SQL: {sql}{Fore.RESET}")
            monitoring_service.log_agent_activity("execution", {
                "action": "execute_sql",
                "details": {"sql": sql},
                "query_id": query_id
            })
            
            execution_result = self._execute_sql(sql, workflow_context, query_id)
            
            if "error" in execution_result:
                # Handle execution error
                error_message = execution_result["error"]
                logger.error(f"{Fore.RED}SQL Execution Error: {error_message}{Fore.RESET}")
                
                # Try to recover from the error
                recovery_result = self._handle_execution_error(nlq, sql, error_message, workflow_context, query_id)
                
                if "error" in recovery_result:
                    # Could not recover from the error
                    logger.error(f"{Fore.RED}Could not recover from execution error{Fore.RESET}")
                    
                    # Update active query status
                    self.active_queries[query_id]["status"] = "error"
                    self.active_queries[query_id]["error"] = error_message
                    
                    # Log the error
                    conversation_logger.log_trino_ai_to_trino("error", {"error": error_message})
                    
                    return {
                        "query_id": query_id,
                        "query": nlq,
                        "error": error_message,
                        "is_data_query": True,
                        "sql": sql
                    }
                
                # Successfully recovered from the error
                sql = recovery_result["sql"]
                execution_result = recovery_result["execution_result"]
                workflow_context.set_generated_sql(sql)
            
            # Step 4: Generate explanation
            logger.info(f"{Fore.GREEN}Generating explanation{Fore.RESET}")
            monitoring_service.log_agent_activity("explanation", {
                "action": "generate_explanation",
                "details": {"sql": sql, "result_size": len(execution_result["data"]) if "data" in execution_result else 0},
                "query_id": query_id
            })
            
            explanation = self._generate_explanation(nlq, sql, execution_result, workflow_context, query_id)
            workflow_context.set_explanation(explanation)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Update active query status
            self.active_queries[query_id]["status"] = "completed"
            self.active_queries[query_id]["sql"] = sql
            self.active_queries[query_id]["execution_result"] = execution_result
            self.active_queries[query_id]["explanation"] = explanation
            self.active_queries[query_id]["execution_time"] = execution_time
            
            # Log the final answer
            conversation_logger.log_trino_ai_to_trino("answer", {
                "sql": sql,
                "result": execution_result,
                "explanation": explanation
            })
            
            # Save this as an exemplar if it was successful
            self._save_as_exemplar(nlq, sql, execution_result, explanation, workflow_context)
            
            return {
                "query_id": query_id,
                "query": nlq,
                "is_data_query": True,
                "sql": sql,
                "execution_result": execution_result,
                "explanation": explanation,
                "execution_time": execution_time
            }
        except Exception as e:
            logger.error(f"{Fore.RED}Error processing query: {str(e)}{Fore.RESET}")
            logger.error(traceback.format_exc())
            
            # Update active query status
            self.active_queries[query_id]["status"] = "error"
            self.active_queries[query_id]["error"] = str(e)
            
            # Log the error
            conversation_logger = ConversationLogger()
            conversation_logger.log_trino_ai_to_trino("error", {"error": str(e)})
            
            return {
                "query_id": query_id,
                "query": nlq,
                "error": f"An unexpected error occurred: {str(e)}",
                "is_data_query": None
            } 

    def _refine_sql(self, sql: str, nlq: str, suggestions: List[str], workflow_context: WorkflowContext) -> str:
        """Refine SQL based on critique suggestions"""
        system_prompt = """
        You are a SQL refinement expert. You will be given an initial SQL query and suggestions for improvement.
        Refine the SQL query to address all the suggestions while maintaining the original intent.
        Return only the refined SQL query without explanation.
        """
        
        # Create suggestions text with proper line breaks (fixing the backslash in f-string error)
        suggestions_text = ""
        for suggestion in suggestions:
            suggestions_text += f"- {suggestion}\n"
        
        user_content = f"""
        Original question: {nlq}
        
        Initial SQL query:
        {sql}
        
        Suggestions for improvement:
        {suggestions_text}
        
        Provide the refined SQL query:
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        
        response = self.ollama_client.chat_completion(messages, agent_name="SQL Refiner")
        refined_sql = response.get("message", {}).get("content", sql)
        
        # Extract just the SQL query from the response if needed
        sql_pattern = r'```sql\n(.*?)\n```'
        match = re.search(sql_pattern, refined_sql, re.DOTALL)
        if match:
            refined_sql = match.group(1)
        
        return refined_sql 