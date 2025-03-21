import logging
import json
import re
from typing import Dict, Any, List, Optional
import sys
import os
import time

# Add the parent directory to the path so we can import from the parent module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.base_agent import Agent
from ollama_client import OllamaClient
from colorama import Fore
from conversation_logger import conversation_logger
from workflow_context import WorkflowContext
from exemplars_manager import ExemplarsManager

logger = logging.getLogger(__name__)

class SQLAgent(Agent):
    """Agent that generates and refines SQL queries"""
    
    def __init__(self, name: str = "SQL Agent", description: str = "Generates accurate, efficient, and valid SQL queries", 
                 ollama_client: Optional[OllamaClient] = None, tools: Optional[Dict[str, Any]] = None,
                 exemplars_manager: Optional[ExemplarsManager] = None):
        super().__init__(name, description, ollama_client)
        self.tools = tools or {}
        self.exemplars_manager = exemplars_manager
        logger.info(f"{Fore.CYAN}SQL Agent initialized with {len(self.tools)} tools and {'with' if exemplars_manager else 'without'} exemplars manager{Fore.RESET}")
    
    def execute(self, inputs: Dict[str, Any], workflow_context: Optional[WorkflowContext] = None) -> Dict[str, Any]:
        """
        Generate SQL from natural language query and DBA analysis
        
        Args:
            inputs: Dictionary containing:
                - query: The natural language query
                - schema_context: The schema context
                - dba_analysis: The DBA analysis results
            workflow_context: Optional workflow context for logging and tracking
                
        Returns:
            Dictionary containing the generated SQL and explanation
        """
        query = inputs.get("query", "")
        schema_context = inputs.get("schema_context", "")
        dba_analysis = inputs.get("dba_analysis", {})
        
        logger.info(f"{Fore.CYAN}SQL Agent generating SQL for query: {query}{Fore.RESET}")
        conversation_logger.log_trino_ai_processing("sql_agent_generation_start", {
            "query": query,
            "schema_context_length": len(schema_context),
            "dba_analysis_tables": dba_analysis.get("tables", []),
            "dba_analysis_joins": dba_analysis.get("joins", [])
        })
        
        # Get conversation context if available
        conversation_context = ""
        if workflow_context:
            conversation_context = self.get_conversation_context(workflow_context)
        conv_context_str = f"Conversation Context:\n{conversation_context}" if conversation_context else ""
        
        # Get relevant exemplars if exemplars manager is available
        exemplars_text = ""
        if self.exemplars_manager:
            logger.info(f"{Fore.CYAN}Retrieving relevant exemplars for query: {query}{Fore.RESET}")
            relevant_exemplars = self.exemplars_manager.get_relevant_exemplars(query, n=3)
            if relevant_exemplars:
                exemplars_text = self.exemplars_manager.format_exemplars_for_prompt(relevant_exemplars)
                logger.info(f"{Fore.GREEN}Found {len(relevant_exemplars)} relevant exemplars{Fore.RESET}")
                conversation_logger.log_trino_ai_processing("sql_agent_exemplars_found", {
                    "exemplar_count": len(relevant_exemplars),
                    "exemplar_queries": [ex.get("query", "") for ex in relevant_exemplars]
                })
            else:
                logger.info(f"{Fore.YELLOW}No relevant exemplars found{Fore.RESET}")
        
        # Prepare the prompt for the LLM
        system_prompt = self.get_system_prompt()
        
        # Create a detailed prompt with the DBA analysis
        tables_info = ""
        if isinstance(dba_analysis.get("tables"), list):
            for table in dba_analysis.get("tables", []):
                if isinstance(table, dict):
                    tables_info += f"- {table.get('name', 'Unknown')}"
                    if table.get("alias"):
                        tables_info += f" (alias: {table.get('alias')})"
                    if table.get("reason"):
                        tables_info += f": {table.get('reason')}"
                    tables_info += "\n"
                elif isinstance(table, str):
                    tables_info += f"- {table}\n"
        
        columns_info = ""
        if isinstance(dba_analysis.get("columns"), list):
            for column in dba_analysis.get("columns", []):
                if isinstance(column, dict):
                    columns_info += f"- {column.get('table', 'Unknown')}.{column.get('name', 'Unknown')}"
                    if column.get("purpose"):
                        columns_info += f": {column.get('purpose')}"
                    columns_info += "\n"
                elif isinstance(column, str):
                    columns_info += f"- {column}\n"
        
        joins_info = ""
        if isinstance(dba_analysis.get("joins"), list):
            for join in dba_analysis.get("joins", []):
                if isinstance(join, dict):
                    joins_info += f"- {join.get('left_table', 'Unknown')}.{join.get('left_column', 'Unknown')} {join.get('join_type', 'JOIN')} {join.get('right_table', 'Unknown')}.{join.get('right_column', 'Unknown')}"
                    if join.get("reason"):
                        joins_info += f": {join.get('reason')}"
                    joins_info += "\n"
                elif isinstance(join, str):
                    joins_info += f"- {join}\n"
        
        filters_info = ""
        if isinstance(dba_analysis.get("filters"), list):
            for filter_item in dba_analysis.get("filters", []):
                if isinstance(filter_item, dict):
                    filters_info += f"- {filter_item.get('table', 'Unknown')}.{filter_item.get('column', 'Unknown')} {filter_item.get('operator', '=')} {filter_item.get('value', '')}"
                    if filter_item.get("reason"):
                        filters_info += f": {filter_item.get('reason')}"
                    filters_info += "\n"
                elif isinstance(filter_item, str):
                    filters_info += f"- {filter_item}\n"
        
        aggregations_info = ""
        if isinstance(dba_analysis.get("aggregations"), list):
            for agg in dba_analysis.get("aggregations", []):
                if isinstance(agg, dict):
                    aggregations_info += f"- {agg.get('function', 'Unknown')}({agg.get('column', 'Unknown')})"
                    if agg.get("purpose"):
                        aggregations_info += f": {agg.get('purpose')}"
                    aggregations_info += "\n"
                elif isinstance(agg, str):
                    aggregations_info += f"- {agg}\n"
        
        user_prompt = f"""
        Natural Language Query: {query}
        
        Schema Context:
        {schema_context}
        
        {conv_context_str}
        
        {exemplars_text if exemplars_text else ""}
        
        DBA Analysis:
        
        Tables:
        {tables_info if tables_info else "No tables identified"}
        
        Columns:
        {columns_info if columns_info else "No columns identified"}
        
        Joins:
        {joins_info if joins_info else "No joins identified"}
        
        Filters:
        {filters_info if filters_info else "No filters identified"}
        
        Aggregations:
        {aggregations_info if aggregations_info else "No aggregations identified"}
        
        Generate a valid Trino SQL query that answers the natural language query based on the DBA analysis.
        Return only the SQL query without any explanations or markdown formatting.
        """
        
        # Get the SQL from the LLM
        logger.info(f"{Fore.YELLOW}Sending query to LLM for SQL generation{Fore.RESET}")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = self.ollama_client.chat_completion(messages, agent_name="sql_agent")
            
            if "error" in response:
                logger.error(f"{Fore.RED}Error from LLM: {response['error']}{Fore.RESET}")
                conversation_logger.log_error("sql_agent", f"LLM error: {response['error']}")
                return {"error": response["error"]}
            
            response_text = response.get("message", {}).get("content", "")
            logger.info(f"{Fore.GREEN}Received SQL from LLM ({len(response_text)} chars){Fore.RESET}")
            conversation_logger.log_ollama_to_trino_ai("sql_agent", response_text[:500] + "..." if len(response_text) > 500 else response_text)
            
            # Extract SQL from the response
            sql = self._extract_sql(response_text)
            
            if not sql:
                logger.warning(f"{Fore.YELLOW}No SQL found in response, using full response{Fore.RESET}")
                sql = response_text
            
            # Validate the SQL
            is_valid = True
            error_message = ""
            
            if "validate_sql" in self.tools:
                logger.info(f"{Fore.YELLOW}Validating SQL{Fore.RESET}")
                validate_tool = self.tools["validate_sql"]
                validation_result = validate_tool.execute({"sql": sql})
                
                is_valid = validation_result.get("is_valid", False)
                error_message = validation_result.get("error_message", "")
                
                if is_valid:
                    logger.info(f"{Fore.GREEN}SQL validation successful{Fore.RESET}")
                    conversation_logger.log_trino_ai_processing("sql_agent_validation_success", {
                        "sql": sql
                    })
            
                    # Store successful query as exemplar if exemplars manager is available
                    if self.exemplars_manager and workflow_context:
                        # Only store if this is a final, valid SQL that was executed successfully
                        if workflow_context.get_metadata("sql_execution", {}).get("success", False):
                            logger.info(f"{Fore.CYAN}Storing successful query as exemplar{Fore.RESET}")
                            self.exemplars_manager.add_exemplar(
                                query=query,
                                sql=sql,
                                metadata={
                                    "tables": [t.get("name") if isinstance(t, dict) else t for t in dba_analysis.get("tables", [])],
                                    "timestamp": time.time()
                                }
                            )
                else:
                    logger.warning(f"{Fore.YELLOW}SQL validation failed: {error_message}{Fore.RESET}")
                    conversation_logger.log_trino_ai_processing("sql_agent_validation_failed", {
                        "sql": sql,
                        "error_message": error_message
                    })
                    
                    # Try to refine the SQL
                    refinement_result = self.refine_sql({
                        "query": query,
                        "sql": sql,
                        "error_message": error_message,
                        "schema_context": schema_context,
                        "dba_analysis": dba_analysis
                    }, workflow_context)
                    
                    sql = refinement_result.get("sql", sql)
                    is_valid = refinement_result.get("is_valid", is_valid)
                    error_message = refinement_result.get("error_message", error_message)
            
            # Update workflow context if provided
            if workflow_context:
                workflow_context.set_sql(sql)
                workflow_context.add_metadata("sql_generation", {
                    "is_valid": is_valid,
                    "error_message": error_message if not is_valid else ""
                })
            
            return {
                "sql": sql,
                "is_valid": is_valid,
                "error_message": error_message if not is_valid else ""
            }
            
        except Exception as e:
            logger.error(f"{Fore.RED}Error during SQL generation: {str(e)}{Fore.RESET}")
            conversation_logger.log_error("sql_agent", f"Error during SQL generation: {str(e)}")
            return {"error": str(e)}
    
    def _extract_sql(self, text: str) -> str:
        """
        Extract SQL from text, removing markdown code blocks if present
        
        Args:
            text: The text containing SQL
            
        Returns:
            The extracted SQL
        """
        # Try to find SQL in code blocks
        sql_match = re.search(r'```(?:sql)?\s*([\s\S]*?)\s*```', text)
        
        if sql_match:
            sql = sql_match.group(1).strip()
            logger.info(f"{Fore.GREEN}Extracted SQL from code block{Fore.RESET}")
        else:
            # Just use the whole text
            sql = text.strip()
            logger.info(f"{Fore.YELLOW}No code block found, using entire text as SQL{Fore.RESET}")
        
        return sql
    
    def refine_sql(self, inputs: Dict[str, Any], workflow_context: Optional[WorkflowContext] = None) -> Dict[str, Any]:
        """
        Refine a SQL query based on error message or intent feedback
        
        Args:
            inputs: Dictionary containing:
                - query: The natural language query
                - sql: The SQL to refine
                - error_message: Error message from validation
                - schema_context: The schema context
                - dba_analysis: The DBA analysis results 
                - intent_feedback: Optional feedback from intent verification
            workflow_context: Optional workflow context for logging and tracking
            
        Returns:
            Dictionary containing the refined SQL and validation status
        """
        query = inputs.get("query", "")
        sql = inputs.get("sql", "")
        error_message = inputs.get("error_message", "")
        schema_context = inputs.get("schema_context", "")
        dba_analysis = inputs.get("dba_analysis", {})
        intent_feedback = inputs.get("intent_feedback", {})
        
        logger.info(f"{Fore.CYAN}SQL Agent refining SQL for query: {query}{Fore.RESET}")
        conversation_logger.log_trino_ai_processing("sql_agent_refinement_start", {
            "query": query,
            "sql": sql[:200] + "..." if len(sql) > 200 else sql,
            "error_message": error_message,
            "has_intent_feedback": bool(intent_feedback)
        })
        
        # Get conversation context if available
        conversation_context = ""
        if workflow_context:
            conversation_context = self.get_conversation_context(workflow_context)
        conv_context_str = f"Conversation Context:\n{conversation_context}" if conversation_context else ""
        
        # Get relevant exemplars if exemplars manager is available
        exemplars_text = ""
        if self.exemplars_manager:
            # For refinement, we want to find exemplars that are similar to the error or intent issue
            search_query = query
            if intent_feedback:
                # If we have intent feedback, include that in the search to find similar examples
                missing_aspects = intent_feedback.get("missing_aspects", [])
                search_query = f"{query} {' '.join(missing_aspects)}"
            elif error_message:
                # If we have an error message, include that in the search
                search_query = f"{query} {error_message}"
                
            logger.info(f"{Fore.CYAN}Retrieving relevant exemplars for refinement: {search_query}{Fore.RESET}")
            relevant_exemplars = self.exemplars_manager.get_relevant_exemplars(search_query, n=2)
            if relevant_exemplars:
                exemplars_text = self.exemplars_manager.format_exemplars_for_prompt(relevant_exemplars)
                logger.info(f"{Fore.GREEN}Found {len(relevant_exemplars)} relevant exemplars for refinement{Fore.RESET}")
                conversation_logger.log_trino_ai_processing("sql_agent_refinement_exemplars_found", {
                    "exemplar_count": len(relevant_exemplars),
                    "exemplar_queries": [ex.get("query", "") for ex in relevant_exemplars]
                })
            else:
                logger.info(f"{Fore.YELLOW}No relevant exemplars found for refinement{Fore.RESET}")
        
        # Determine if this is refinement for SQL syntax or for intent mismatch
        is_intent_refinement = "intent_feedback" in inputs and intent_feedback
        
        # Choose appropriate system prompt
        system_prompt = self.get_intent_refinement_prompt() if is_intent_refinement else self.get_error_refinement_prompt()
        
        # Prepare feedback information
        feedback_info = ""
        if is_intent_refinement:
            missing_aspects = intent_feedback.get("missing_aspects", [])
            suggestions = intent_feedback.get("suggestions", [])
            
            feedback_info = f"""
            Intent Analysis Feedback:
            - Missing aspects: {', '.join(missing_aspects)}
            - Suggestions: {', '.join(suggestions)}
            """
        
        # Format DBA analysis for the prompt
        dba_analysis_formatted = self._format_dba_analysis(dba_analysis)
        
        # Create the user prompt
        user_prompt = f"""
        Natural Language Query: {query}
        
        Current SQL:
        {sql}
        
        {'Error Message:' if not is_intent_refinement else 'Intent Mismatch:'}
        {error_message}
        
        {feedback_info if is_intent_refinement else ''}
        
        {exemplars_text if exemplars_text else ""}
        
        Schema Context:
        {schema_context}
        
        DBA Analysis:
        {dba_analysis_formatted}
        
        {conv_context_str}
        
        Please refine the SQL query to {'fully address the original query intent' if is_intent_refinement else 'fix the error'}.
        Return only the refined SQL without any explanations or markdown formatting.
        """
        
        # Get the refined SQL from the LLM
        logger.info(f"{Fore.YELLOW}Sending refinement request to LLM{Fore.RESET}")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = self.ollama_client.chat_completion(messages, agent_name="sql_agent_refine")
            
            if "error" in response:
                logger.error(f"{Fore.RED}Error from LLM during refinement: {response['error']}{Fore.RESET}")
                conversation_logger.log_error("sql_agent", f"LLM refinement error: {response['error']}")
                return {
                    "sql": sql,
                    "is_valid": False,
                    "error_message": f"Refinement failed: {response['error']}"
                }
            
            response_text = response.get("message", {}).get("content", "")
            logger.info(f"{Fore.GREEN}Received refined SQL from LLM ({len(response_text)} chars){Fore.RESET}")
            conversation_logger.log_ollama_to_trino_ai("sql_agent_refine", response_text[:500] + "..." if len(response_text) > 500 else response_text)
            
            # Extract SQL from the response
            refined_sql = self._extract_sql(response_text)
            
            if not refined_sql:
                logger.warning(f"{Fore.YELLOW}No SQL found in refinement response, using full response{Fore.RESET}")
                refined_sql = response_text
            
            # Validate the refined SQL
            is_valid = True
            new_error_message = ""
            
            if "validate_sql" in self.tools:
                logger.info(f"{Fore.YELLOW}Validating refined SQL{Fore.RESET}")
                validation_result = self.tools["validate_sql"].execute({"sql": refined_sql})
                
                is_valid = validation_result.get("is_valid", False)
                new_error_message = validation_result.get("error_message", "")
                
                if is_valid:
                    logger.info(f"{Fore.GREEN}Refined SQL validation successful{Fore.RESET}")
                    conversation_logger.log_trino_ai_processing("sql_agent_refinement_success", {
                        "sql": refined_sql[:200] + "..." if len(refined_sql) > 200 else refined_sql
                    })
                    
                    # Store successful refinement as exemplar if exemplars manager is available
                    if self.exemplars_manager and workflow_context:
                        # Only store if this is a successful refinement
                        logger.info(f"{Fore.CYAN}Storing successful refinement as exemplar{Fore.RESET}")
                        refinement_type = "intent_refinement" if is_intent_refinement else "error_refinement"
                        self.exemplars_manager.add_exemplar(
                            query=f"{refinement_type}: {query}",
                            sql=refined_sql,
                            metadata={
                                "original_sql": sql,
                                "refinement_type": refinement_type,
                                "error_message": error_message if not is_intent_refinement else "",
                                "intent_feedback": intent_feedback if is_intent_refinement else {},
                                "timestamp": time.time()
                            }
                        )
                else:
                    logger.warning(f"{Fore.YELLOW}Refined SQL validation failed: {new_error_message}{Fore.RESET}")
                    conversation_logger.log_trino_ai_processing("sql_agent_refinement_failed", {
                        "error_message": new_error_message,
                        "sql": refined_sql[:200] + "..." if len(refined_sql) > 200 else refined_sql
                    })
            
            # Update workflow context if provided
            if workflow_context:
                workflow_context.set_sql(refined_sql)
                workflow_context.add_metadata("sql_refinement", {
                    "is_valid": is_valid,
                    "error_message": new_error_message if not is_valid else "",
                    "refinement_type": "intent" if is_intent_refinement else "error"
                })
            
            return {
                "sql": refined_sql,
                "is_valid": is_valid,
                "error_message": new_error_message if not is_valid else ""
            }
            
        except Exception as e:
            logger.error(f"{Fore.RED}Error during SQL refinement: {str(e)}{Fore.RESET}")
            conversation_logger.log_error("sql_agent", f"Refinement error: {str(e)}")
            return {
                "sql": sql,
                "is_valid": False,
                "error_message": f"Refinement failed: {str(e)}"
            }
    
    def _format_dba_analysis(self, dba_analysis: Dict[str, Any]) -> str:
        """Format DBA analysis for inclusion in prompts"""
        formatted = ""
        
        # Format tables
        if "tables" in dba_analysis and dba_analysis["tables"]:
            formatted += "Tables:\n"
            for table in dba_analysis["tables"]:
                if isinstance(table, dict):
                    formatted += f"- {table.get('name', 'Unknown')}"
                    if "reason" in table:
                        formatted += f": {table['reason']}"
                    formatted += "\n"
                else:
                    formatted += f"- {table}\n"
            formatted += "\n"
        
        # Format columns
        if "columns" in dba_analysis and dba_analysis["columns"]:
            formatted += "Columns:\n"
            for column in dba_analysis["columns"]:
                if isinstance(column, dict):
                    formatted += f"- {column.get('table', 'Unknown')}.{column.get('name', 'Unknown')}"
                    if "purpose" in column:
                        formatted += f": {column['purpose']}"
                    formatted += "\n"
                else:
                    formatted += f"- {column}\n"
            formatted += "\n"
        
        # Format joins
        if "joins" in dba_analysis and dba_analysis["joins"]:
            formatted += "Joins:\n"
            for join in dba_analysis["joins"]:
                if isinstance(join, dict):
                    formatted += f"- {join.get('left_table', 'Unknown')}.{join.get('left_column', 'Unknown')} {join.get('join_type', 'JOIN')} {join.get('right_table', 'Unknown')}.{join.get('right_column', 'Unknown')}"
                    if "reason" in join:
                        formatted += f": {join['reason']}"
                    formatted += "\n"
                else:
                    formatted += f"- {join}\n"
            formatted += "\n"
        
        return formatted
            
    def get_intent_refinement_prompt(self) -> str:
        """Get the system prompt for intent-based SQL refinement"""
        return """
        You are an expert SQL developer specializing in Trino SQL. Your task is to refine a SQL query
        to better match the original query intent based on feedback from intent analysis.
        
        The SQL query you're refining has correct syntax but doesn't fully address all aspects of
        the original natural language query. Your job is to modify the SQL to ensure it captures
        ALL requirements from the original question.
        
        Common issues to fix:
        1. Missing tables that contain relevant information
        2. Missing columns needed to provide complete answers
        3. Incomplete filter conditions (WHERE clauses)
        4. Incorrect or missing JOIN conditions
        5. Missing or incorrect aggregations
        6. Missing or incorrect grouping
        7. Missing or incorrect ordering
        
        When refining the SQL:
        - Carefully analyze what aspects of the query are missing
        - Ensure your solution addresses ALL the requirements
        - Maintain correct Trino SQL syntax
        - Keep the solution as efficient as possible
        - Ensure all column references are properly qualified
        
        Return ONLY the refined SQL without explanations or markdown formatting.
        """
    
    def get_error_refinement_prompt(self) -> str:
        """Get the system prompt for error-based SQL refinement"""
        return """
        You are an expert SQL developer specializing in Trino SQL. Your task is to fix a SQL query
        that has syntax or semantic errors based on the provided error message.
        
        The SQL query you're fixing has errors that prevent it from executing. Your job is to analyze
        the error message and make the necessary corrections while ensuring the query still addresses
        the original question.
        
        Common errors to fix:
        1. Missing or misspelled table/column names
        2. Incorrect table qualifications (catalog.schema.table)
        3. Type mismatches in comparisons or joins
        4. Ambiguous column references
        5. Missing GROUP BY columns for aggregations
        6. Invalid function calls or syntax
        
        When fixing the SQL:
        - Focus on addressing the specific error in the message
        - Maintain the original query intent
        - Ensure all table and column references are valid
        - Keep the solution as clean and efficient as possible
        
        Return ONLY the fixed SQL without explanations or markdown formatting.
        """

    def generate_sql(self, nlq: str, dba_analysis: Dict[str, Any]) -> str:
        """
        Generate SQL for a natural language query based on DBA analysis
        
        Args:
            nlq: The natural language query to convert to SQL
            dba_analysis: The analysis from the DBA agent
            
        Returns:
            The generated SQL query
        """
        logger.info(f"{Fore.BLUE}SQL Agent generating SQL for: {nlq}{Fore.RESET}")
        logger.info(f"{Fore.BLUE}Using DBA analysis: {str(dba_analysis)[:100]}...{Fore.RESET}")
        
        # Log detailed reasoning
        logger.info(f"{Fore.BLUE}SQL Agent reasoning process:{Fore.RESET}")
        logger.info(f"{Fore.BLUE}1. Reviewing tables identified by DBA: {dba_analysis.get('tables', [])}{Fore.RESET}")
        logger.info(f"{Fore.BLUE}2. Planning SQL structure based on required columns: {dba_analysis.get('columns', [])}{Fore.RESET}")
        logger.info(f"{Fore.BLUE}3. Implementing joins for identified relationships{Fore.RESET}")
        logger.info(f"{Fore.BLUE}4. Adding filters, aggregations and ordering{Fore.RESET}")
        
        # Log the agent action
        conversation_logger.log_trino_ai_to_ollama(self.name, {
            "action": "generate_sql",
            "nlq": nlq,
            "dba_analysis": {k: v for k, v in dba_analysis.items() if k != "schema_context"}
        })
        
        # Build the prompt for the LLM
        # ... rest of the existing code ... 

    def generate_multiple_candidates(self, inputs: Dict[str, Any], workflow_context: Optional[WorkflowContext] = None, num_candidates: int = 3) -> Dict[str, Any]:
        """
        Generate multiple SQL candidates and select the best one
        
        Args:
            inputs: Dictionary containing:
                - query: The natural language query
                - schema_context: The schema context
                - dba_analysis: The DBA analysis results
            workflow_context: Optional workflow context for logging and tracking
            num_candidates: Number of candidates to generate
                
        Returns:
            Dictionary containing the best SQL and its validation status
        """
        query = inputs.get("query", "")
        schema_context = inputs.get("schema_context", "")
        dba_analysis = inputs.get("dba_analysis", {})
        
        logger.info(f"{Fore.CYAN}SQL Agent generating {num_candidates} SQL candidates for query: {query}{Fore.RESET}")
        conversation_logger.log_trino_ai_processing("sql_agent_multi_candidate_start", {
            "query": query,
            "num_candidates": num_candidates
        })
        
        # Get conversation context if available
        conversation_context = ""
        if workflow_context:
            conversation_context = self.get_conversation_context(workflow_context)
        conv_context_str = f"Conversation Context:\n{conversation_context}" if conversation_context else ""
        
        # Get relevant exemplars if exemplars manager is available
        exemplars_text = ""
        if self.exemplars_manager:
            logger.info(f"{Fore.CYAN}Retrieving relevant exemplars for query: {query}{Fore.RESET}")
            relevant_exemplars = self.exemplars_manager.get_relevant_exemplars(query, n=3)
            if relevant_exemplars:
                exemplars_text = self.exemplars_manager.format_exemplars_for_prompt(relevant_exemplars)
                logger.info(f"{Fore.GREEN}Found {len(relevant_exemplars)} relevant exemplars{Fore.RESET}")
            else:
                logger.info(f"{Fore.YELLOW}No relevant exemplars found{Fore.RESET}")
        
        # Format DBA analysis for the prompt
        dba_analysis_formatted = self._format_dba_analysis(dba_analysis)
        
        # Prepare the system prompt for multi-candidate generation
        system_prompt = """
        You are an expert SQL developer specializing in Trino SQL. Your task is to generate multiple different SQL queries
        that answer the same natural language question. Each query should be valid Trino SQL but may take a different approach
        to solving the problem.
        
        For each candidate, think about:
        1. Different table combinations that could answer the question
        2. Different join strategies
        3. Different ways to express filters
        4. Different aggregation approaches
        5. Different ways to order or limit results
        
        Return exactly the number of candidates requested, each clearly labeled as CANDIDATE 1, CANDIDATE 2, etc.
        Each candidate should be a complete, valid Trino SQL query.
        
        Do not include any explanations or markdown formatting in your response, just the labeled SQL candidates.
        """
        
        # Create the user prompt
        user_prompt = f"""
        Natural Language Query: {query}
        
        Schema Context:
        {schema_context}
        
        {conv_context_str}
        
        {exemplars_text if exemplars_text else ""}
        
        DBA Analysis:
        {dba_analysis_formatted}
        
        Generate {num_candidates} different valid Trino SQL queries that answer this question.
        Label each as CANDIDATE 1, CANDIDATE 2, etc.
        """
        
        # Get the SQL candidates from the LLM
        logger.info(f"{Fore.YELLOW}Sending query to LLM for multi-candidate SQL generation{Fore.RESET}")
        messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = self.ollama_client.chat_completion(messages, agent_name="sql_agent_multi")
            
            if "error" in response:
                logger.error(f"{Fore.RED}Error from LLM: {response['error']}{Fore.RESET}")
                conversation_logger.log_error("sql_agent", f"LLM error in multi-candidate generation: {response['error']}")
                # Fall back to single candidate generation
                return self.execute(inputs, workflow_context)
            
            response_text = response.get("message", {}).get("content", "")
            logger.info(f"{Fore.GREEN}Received SQL candidates from LLM ({len(response_text)} chars){Fore.RESET}")
            
            # Extract candidates from the response
            candidates = self._extract_candidates(response_text, num_candidates)
            
            if not candidates:
                logger.warning(f"{Fore.YELLOW}No SQL candidates found in response, falling back to single generation{Fore.RESET}")
                return self.execute(inputs, workflow_context)
            
            logger.info(f"{Fore.GREEN}Extracted {len(candidates)} SQL candidates{Fore.RESET}")
            conversation_logger.log_trino_ai_processing("sql_agent_candidates_extracted", {
                "num_candidates": len(candidates)
            })
            
            # Validate each candidate
            valid_candidates = []
            
            if "validate_sql" in self.tools:
                validate_tool = self.tools["validate_sql"]
                
                for i, candidate in enumerate(candidates):
                    logger.info(f"{Fore.YELLOW}Validating candidate {i+1}/{len(candidates)}{Fore.RESET}")
                    validation_result = validate_tool.execute({"sql": candidate})
                    
                    is_valid = validation_result.get("is_valid", False)
                    error_message = validation_result.get("error_message", "")
                    
                    if is_valid:
                        logger.info(f"{Fore.GREEN}Candidate {i+1} validation successful{Fore.RESET}")
                        valid_candidates.append({
                            "sql": candidate,
                            "candidate_num": i+1,
                            "is_valid": True
                        })
                    else:
                        logger.warning(f"{Fore.YELLOW}Candidate {i+1} validation failed: {error_message}{Fore.RESET}")
            
            # If no valid candidates, try to refine the first one
            if not valid_candidates:
                logger.warning(f"{Fore.YELLOW}No valid candidates found, attempting to refine first candidate{Fore.RESET}")
                
                if candidates:
                    refinement_result = self.refine_sql({
                        "query": query,
                        "sql": candidates[0],
                        "error_message": "No valid candidates found",
                        "schema_context": schema_context,
                        "dba_analysis": dba_analysis
                    }, workflow_context)
                    
                    if refinement_result.get("is_valid", False):
                        logger.info(f"{Fore.GREEN}Refinement successful{Fore.RESET}")
                        return refinement_result
                
                # If refinement fails or no candidates to refine, fall back to single generation
                logger.warning(f"{Fore.YELLOW}Refinement failed, falling back to single generation{Fore.RESET}")
                return self.execute(inputs, workflow_context)
            
            # If we have valid candidates, verify intent for each
            if "verify_intent" in self.tools and valid_candidates:
                intent_tool = self.tools["verify_intent"]
                
                for i, candidate in enumerate(valid_candidates):
                    logger.info(f"{Fore.YELLOW}Verifying intent for candidate {candidate['candidate_num']}{Fore.RESET}")
                    
                    intent_result = intent_tool.execute({
                        "query": query,
                        "sql": candidate["sql"],
                        "schema_context": schema_context,
                        "dba_analysis": dba_analysis
                    })
                    
                    matches_intent = intent_result.get("matches_intent", False)
                    candidate["matches_intent"] = matches_intent
                    candidate["intent_score"] = intent_result.get("confidence", 0)
                    candidate["missing_aspects"] = intent_result.get("missing_aspects", [])
                    
                    if matches_intent:
                        logger.info(f"{Fore.GREEN}Candidate {candidate['candidate_num']} matches intent{Fore.RESET}")
                    else:
                        logger.warning(f"{Fore.YELLOW}Candidate {candidate['candidate_num']} does not match intent{Fore.RESET}")
            
            # Select the best candidate
            best_candidate = self._select_best_candidate(valid_candidates)
            
            if best_candidate:
                logger.info(f"{Fore.GREEN}Selected candidate {best_candidate['candidate_num']} as best{Fore.RESET}")
                conversation_logger.log_trino_ai_processing("sql_agent_best_candidate_selected", {
                    "candidate_num": best_candidate['candidate_num'],
                    "matches_intent": best_candidate.get('matches_intent', False),
                    "intent_score": best_candidate.get('intent_score', 0)
                })
                
                # Update workflow context if provided
                if workflow_context:
                    workflow_context.set_sql(best_candidate["sql"])
                    workflow_context.add_metadata("sql_generation", {
                        "is_valid": True,
                        "candidate_num": best_candidate['candidate_num'],
                        "total_candidates": len(candidates),
                        "valid_candidates": len(valid_candidates),
                        "matches_intent": best_candidate.get('matches_intent', False)
                    })
                
                return {
                    "sql": best_candidate["sql"],
                    "is_valid": True,
                    "candidate_num": best_candidate['candidate_num'],
                    "total_candidates": len(candidates),
                    "valid_candidates": len(valid_candidates)
                }
            else:
                logger.warning(f"{Fore.YELLOW}No best candidate found, falling back to single generation{Fore.RESET}")
                return self.execute(inputs, workflow_context)
                
        except Exception as e:
            logger.error(f"{Fore.RED}Error during multi-candidate SQL generation: {str(e)}{Fore.RESET}")
            conversation_logger.log_error("sql_agent", f"Error during multi-candidate generation: {str(e)}")
            # Fall back to single candidate generation
            return self.execute(inputs, workflow_context)
    
    def _extract_candidates(self, text: str, expected_count: int) -> List[str]:
        """
        Extract SQL candidates from text
        
        Args:
            text: The text containing SQL candidates
            expected_count: The expected number of candidates
            
        Returns:
            List of SQL candidate strings
        """
        candidates = []
        
        # Try to find candidates by looking for "CANDIDATE X" markers
        import re
        candidate_blocks = re.split(r'CANDIDATE\s+\d+\s*:', text)
        
        # First element is usually empty or intro text
        if candidate_blocks and len(candidate_blocks) > 1:
            for block in candidate_blocks[1:]:  # Skip the first element
                # Extract SQL from the block
                sql = self._extract_sql(block)
                if sql:
                    candidates.append(sql)
        
        # If we couldn't find candidates using the markers, try to extract code blocks
        if not candidates:
            # Try to find SQL in code blocks
            code_blocks = re.findall(r'```(?:sql)?\s*([\s\S]*?)\s*```', text)
            
            if code_blocks:
                for block in code_blocks:
                    candidates.append(block.strip())
        
        # If we still don't have candidates, just use the whole text as one candidate
        if not candidates:
            candidates.append(text.strip())
        
        logger.info(f"{Fore.GREEN}Extracted {len(candidates)} candidates (expected {expected_count}){Fore.RESET}")
        return candidates
    
    def _select_best_candidate(self, candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Select the best SQL candidate based on intent matching and other criteria
        
        Args:
            candidates: List of candidate dictionaries with validation and intent info
            
        Returns:
            The best candidate or None if no candidates
        """
        if not candidates:
            return None
        
        # First, filter for candidates that match intent
        intent_matches = [c for c in candidates if c.get("matches_intent", False)]
        
        # If we have intent matches, select the one with highest score
        if intent_matches:
            return max(intent_matches, key=lambda c: c.get("intent_score", 0))
        
        # If no intent matches, select the one with fewest missing aspects
        min_missing = min(candidates, key=lambda c: len(c.get("missing_aspects", [])))
        
        # If we have a candidate with no missing aspects, return it
        if len(min_missing.get("missing_aspects", [])) == 0:
            return min_missing
        
        # Otherwise, just return the first valid candidate
        return candidates[0]
    
    def build_sql_progressively(self, inputs: Dict[str, Any], workflow_context: Optional[WorkflowContext] = None) -> Dict[str, Any]:
        """
        Build SQL progressively, validating at each step
        
        Args:
            inputs: Dictionary containing:
                - query: The natural language query
                - schema_context: The schema context
                - dba_analysis: The DBA analysis results
            workflow_context: Optional workflow context for logging and tracking
            
        Returns:
            Dictionary containing the built SQL and its validation status
        """
        query = inputs.get("query", "")
        schema_context = inputs.get("schema_context", "")
        dba_analysis = inputs.get("dba_analysis", {})
        
        logger.info(f"{Fore.CYAN}SQL Agent building SQL progressively for query: {query}{Fore.RESET}")
        conversation_logger.log_trino_ai_processing("sql_agent_progressive_build_start", {
            "query": query
        })
        
        # Get conversation context if available
        conversation_context = ""
        if workflow_context:
            conversation_context = self.get_conversation_context(workflow_context)
        
        # Get relevant exemplars if exemplars manager is available
        exemplars_text = ""
        if self.exemplars_manager:
            logger.info(f"{Fore.CYAN}Retrieving relevant exemplars for query: {query}{Fore.RESET}")
            relevant_exemplars = self.exemplars_manager.get_relevant_exemplars(query, n=2)
            if relevant_exemplars:
                exemplars_text = self.exemplars_manager.format_exemplars_for_prompt(relevant_exemplars)
                logger.info(f"{Fore.GREEN}Found {len(relevant_exemplars)} relevant exemplars{Fore.RESET}")
            else:
                logger.info(f"{Fore.YELLOW}No relevant exemplars found{Fore.RESET}")
        
        # Format DBA analysis for the prompt
        dba_analysis_formatted = self._format_dba_analysis(dba_analysis)
        
        # Define the steps for progressive SQL building
        build_steps = [
            {
                "name": "select_clause",
                "description": "Build the SELECT clause with all necessary columns",
                "prompt": "Generate only the SELECT clause of the SQL query, including all necessary columns with proper table qualifications."
            },
            {
                "name": "from_clause",
                "description": "Build the FROM clause with all necessary tables",
                "prompt": "Using the SELECT clause above, add the FROM clause with all necessary tables."
            },
            {
                "name": "join_clause",
                "description": "Add all necessary JOIN clauses",
                "prompt": "Using the SQL above, add all necessary JOIN clauses with the correct join conditions."
            },
            {
                "name": "where_clause",
                "description": "Add the WHERE clause with all necessary filters",
                "prompt": "Using the SQL above, add the WHERE clause with all necessary filters."
            },
            {
                "name": "group_by_clause",
                "description": "Add GROUP BY clause if needed",
                "prompt": "Using the SQL above, add a GROUP BY clause if needed for any aggregations."
            },
            {
                "name": "having_clause",
                "description": "Add HAVING clause if needed",
                "prompt": "Using the SQL above, add a HAVING clause if needed for filtering aggregated results."
            },
            {
                "name": "order_by_clause",
                "description": "Add ORDER BY clause",
                "prompt": "Using the SQL above, add an ORDER BY clause to sort the results appropriately."
            },
            {
                "name": "limit_clause",
                "description": "Add LIMIT clause if needed",
                "prompt": "Using the SQL above, add a LIMIT clause if needed to restrict the number of results."
            }
        ]
        
        # System prompt for progressive building
        system_prompt = """
        You are an expert SQL developer specializing in Trino SQL. Your task is to build a SQL query progressively,
        one clause at a time. Focus only on the current step and build upon the SQL that has been constructed so far.
        
        For each step:
        1. Carefully analyze what is needed for the current clause
        2. Ensure your addition is syntactically correct
        3. Make sure your addition integrates well with the existing SQL
        4. Use proper table qualifications for all column references
        5. Follow Trino SQL syntax precisely
        
        Return only the complete SQL query with your addition, without any explanations or markdown formatting.
        """
        
        # Start with an empty SQL query
        current_sql = ""
        
        # Track progress
        progress = []
        
        # Build the SQL progressively
        for step in build_steps:
            logger.info(f"{Fore.CYAN}Progressive build step: {step['name']} - {step['description']}{Fore.RESET}")
            
            # Create the user prompt for this step
            user_prompt = f"""
            Natural Language Query: {query}
            
            Schema Context:
            {schema_context}
            """
            
            # Add conversation context if available
            if conversation_context:
                user_prompt += f"""
                
                Conversation Context:
                {conversation_context}
                """
            
            # Add exemplars if available
            if exemplars_text:
                user_prompt += f"""
                
                {exemplars_text}
                """
            
            # Add the rest of the prompt
            user_prompt += f"""
            
            DBA Analysis:
            {dba_analysis_formatted}
            
            Current SQL:
            {current_sql if current_sql else "No SQL built yet."}
            
            Current Step: {step['name']} - {step['description']}
            
            {step['prompt']}
            """
            
            # Get the SQL addition from the LLM
            logger.info(f"{Fore.YELLOW}Sending progressive build request to LLM for step: {step['name']}{Fore.RESET}")
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            try:
                response = self.ollama_client.chat_completion(messages, agent_name=f"sql_agent_progressive_{step['name']}")
                
                if "error" in response:
                    logger.error(f"{Fore.RED}Error from LLM: {response['error']}{Fore.RESET}")
                    conversation_logger.log_error("sql_agent", f"LLM error in progressive build: {response['error']}")
                    continue
                
                response_text = response.get("message", {}).get("content", "")
                logger.info(f"{Fore.GREEN}Received SQL from LLM for step {step['name']} ({len(response_text)} chars){Fore.RESET}")
                
                # Extract SQL from the response
                step_sql = self._extract_sql(response_text)
                
                if not step_sql:
                    logger.warning(f"{Fore.YELLOW}No SQL found in response for step {step['name']}, using full response{Fore.RESET}")
                    step_sql = response_text
                
                # Update the current SQL
                current_sql = step_sql
                
                # Validate the current SQL
                is_valid = True
                error_message = ""
                
                if "validate_sql" in self.tools:
                    logger.info(f"{Fore.YELLOW}Validating SQL after step {step['name']}{Fore.RESET}")
                    validation_result = self.tools["validate_sql"].execute({"sql": current_sql})
                    
                    is_valid = validation_result.get("is_valid", False)
                    error_message = validation_result.get("error_message", "")
                    
                    if is_valid:
                        logger.info(f"{Fore.GREEN}SQL validation successful after step {step['name']}{Fore.RESET}")
                    else:
                        logger.warning(f"{Fore.YELLOW}SQL validation failed after step {step['name']}: {error_message}{Fore.RESET}")
                        
                        # Try to fix the SQL
                        refinement_result = self.refine_sql({
                            "query": query,
                            "sql": current_sql,
                            "error_message": error_message,
                            "schema_context": schema_context,
                            "dba_analysis": dba_analysis
                        }, workflow_context)
                        
                        current_sql = refinement_result.get("sql", current_sql)
                        is_valid = refinement_result.get("is_valid", is_valid)
                        error_message = refinement_result.get("error_message", error_message)
                
                # Record progress
                progress.append({
                    "step": step["name"],
                    "sql": current_sql,
                    "is_valid": is_valid,
                    "error_message": error_message if not is_valid else ""
                })
                
                # If still not valid after refinement, we might need to skip some steps
                if not is_valid and step["name"] in ["select_clause", "from_clause", "join_clause"]:
                    logger.error(f"{Fore.RED}Critical step {step['name']} failed validation, cannot continue progressive build{Fore.RESET}")
                    break
                
            except Exception as e:
                logger.error(f"{Fore.RED}Error during progressive build step {step['name']}: {str(e)}{Fore.RESET}")
                conversation_logger.log_error("sql_agent", f"Error during progressive build step {step['name']}: {str(e)}")
                continue
        
        # Final validation of the complete SQL
        is_valid = True
        error_message = ""
        
        if "validate_sql" in self.tools:
            logger.info(f"{Fore.YELLOW}Final validation of progressively built SQL{Fore.RESET}")
            validation_result = self.tools["validate_sql"].execute({"sql": current_sql})
            
            is_valid = validation_result.get("is_valid", False)
            error_message = validation_result.get("error_message", "")
            
            if is_valid:
                logger.info(f"{Fore.GREEN}Final SQL validation successful{Fore.RESET}")
                conversation_logger.log_trino_ai_processing("sql_agent_progressive_build_success", {
                    "sql": current_sql[:200] + "..." if len(current_sql) > 200 else current_sql
                })
            else:
                logger.warning(f"{Fore.YELLOW}Final SQL validation failed: {error_message}{Fore.RESET}")
                conversation_logger.log_trino_ai_processing("sql_agent_progressive_build_failed", {
                    "error_message": error_message,
                    "sql": current_sql[:200] + "..." if len(current_sql) > 200 else current_sql
                })
                
                # Try one final refinement
                refinement_result = self.refine_sql({
                    "query": query,
                    "sql": current_sql,
                    "error_message": error_message,
                    "schema_context": schema_context,
                    "dba_analysis": dba_analysis
                }, workflow_context)
                
                current_sql = refinement_result.get("sql", current_sql)
                is_valid = refinement_result.get("is_valid", is_valid)
                error_message = refinement_result.get("error_message", error_message)
        
        # Update workflow context if provided
        if workflow_context:
            workflow_context.set_sql(current_sql)
            workflow_context.add_metadata("sql_progressive_build", {
                "is_valid": is_valid,
                "error_message": error_message if not is_valid else "",
                "steps_completed": len(progress),
                "steps_total": len(build_steps)
            })
        
        return {
            "sql": current_sql,
            "is_valid": is_valid,
            "error_message": error_message if not is_valid else "",
            "progress": progress
        }
