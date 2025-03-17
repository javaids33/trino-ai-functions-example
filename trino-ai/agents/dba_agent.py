import logging
import json
import re
from typing import Dict, Any, List, Optional
import sys
import os

# Add the parent directory to the path so we can import from the parent module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.base_agent import Agent
from ollama_client import OllamaClient
from colorama import Fore
from conversation_logger import conversation_logger
from workflow_context import WorkflowContext

logger = logging.getLogger(__name__)

class DBAAgent(Agent):
    """Agent that analyzes natural language queries to identify relevant tables and columns"""
    
    def __init__(self, name: str = "DBA Agent", description: str = "Analyzes natural language queries to identify relevant tables and columns", 
                 ollama_client: Optional[OllamaClient] = None, tools: Optional[Dict[str, Any]] = None):
        super().__init__(name, description, ollama_client)
        self.tools = tools or {}
        logger.info(f"{Fore.CYAN}DBA Agent initialized with {len(self.tools)} tools{Fore.RESET}")
    
    def execute(self, inputs: Dict[str, Any], workflow_context: Optional[WorkflowContext] = None) -> Dict[str, Any]:
        """
        Analyze a natural language query to identify necessary tables, columns, joins, etc.
        
        Args:
            inputs: Dictionary containing:
                - query: The natural language query to analyze
                - schema_context: Optional schema context (if not provided, will be retrieved)
            workflow_context: Optional workflow context for logging and tracking
                
        Returns:
            Dictionary with analysis results including tables, columns, joins, etc.
        """
        # Validate inputs
        is_valid, error_message = self.validate_inputs(inputs, ["query"])
        if not is_valid:
            return self.handle_error(Exception(error_message), workflow_context)
            
        query = inputs.get("query", "")
        schema_context = inputs.get("schema_context", "")
        
        logger.info(f"{Fore.CYAN}DBA Agent analyzing query: {query}{Fore.RESET}")
        conversation_logger.log_trino_ai_processing("dba_agent_analysis_start", {
            "query": query,
            "schema_context_length": len(schema_context),
            "schema_context_preview": schema_context[:200] + "..." if len(schema_context) > 200 else schema_context
        })
        
        # If no schema context was provided, retrieve it
        if not schema_context and "get_schema_context" in self.tools:
            logger.info(f"{Fore.YELLOW}Retrieving schema context for query{Fore.RESET}")
            context_tool = self.tools["get_schema_context"]
            context_result = context_tool.execute({"query": query})
            schema_context = context_result.get("schema_context", "")
            logger.info(f"{Fore.GREEN}Retrieved schema context ({len(schema_context)} chars){Fore.RESET}")
            conversation_logger.log_trino_ai_processing("dba_agent_schema_context_retrieved", {
                "schema_context_length": len(schema_context),
                "schema_context_preview": schema_context[:200] + "..." if len(schema_context) > 200 else schema_context
            })
            
            # Update workflow context with schema context if provided
            if workflow_context:
                workflow_context.set_schema_context(schema_context)
        
        # Check if query likely requires subqueries
        requires_subquery = self.consider_subqueries(query)
        if requires_subquery:
            logger.info(f"{Fore.YELLOW}Query likely requires subqueries or CTEs{Fore.RESET}")
            if workflow_context:
                workflow_context.add_metadata("query_complexity", {
                    "requires_subquery": True,
                    "indicators": self._identify_subquery_indicators(query)
                })
        
        # Get conversation context if available
        conversation_context = ""
        if workflow_context:
            conversation_context = self.get_conversation_context(workflow_context)
        
        # Add subquery note if needed
        subquery_note = ""
        if requires_subquery:
            subquery_note = "Note that this query likely requires subqueries or CTEs. Consider how to structure the analysis to support this complexity."
        
        system_prompt = f"""
        You are a Database Architect (DBA) specialized in analyzing natural language queries and determining the database schema elements needed to answer them.
        
        Analyze the following query and identify the tables, columns, and joins needed to answer it:
        
        Query: {query}
        
        Schema Context:
        {schema_context}
        
        Provide a detailed analysis including:
        - Tables needed to answer the query
        - Columns to select from each table
        - Joins required between tables
        - Filters or conditions to apply
        - Aggregations or calculations needed
        - Reason for the join
        
        {subquery_note}
        
        Provide your analysis in JSON format.
        """
        
        # Get the analysis from the LLM
        logger.info(f"{Fore.YELLOW}Sending query to LLM for analysis{Fore.RESET}")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Conversation Context:\n{conversation_context}" if conversation_context else ""}
        ]
        
        try:
            response = self.ollama_client.chat_completion(messages, agent_name="dba_agent")
            
            if "error" in response:
                logger.error(f"{Fore.RED}Error from LLM: {response['error']}{Fore.RESET}")
                conversation_logger.log_error("dba_agent", f"LLM error: {response['error']}")
                return {"error": response["error"]}
            
            analysis_text = response.get("message", {}).get("content", "")
            logger.info(f"{Fore.GREEN}Received analysis from LLM ({len(analysis_text)} chars){Fore.RESET}")
            conversation_logger.log_ollama_to_trino_ai("dba_agent", analysis_text[:500] + "..." if len(analysis_text) > 500 else analysis_text)
            
            # Extract JSON from the response
            try:
                # Try to find JSON in the response
                json_match = re.search(r'```json\s*([\s\S]*?)\s*```', analysis_text)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    # Try to find any JSON-like structure
                    json_match = re.search(r'({[\s\S]*})', analysis_text)
                    if json_match:
                        json_str = json_match.group(1)
                    else:
                        # Just try to parse the whole thing
                        json_str = analysis_text
                
                # Clean up common formatting issues before parsing
                # Remove spaces between dots in table names (e.g., "iceberg". "table" -> "iceberg.table")
                json_str = re.sub(r'(["\']iceberg["\'])\s*\.\s*(["\']?[a-zA-Z_]+["\']?)', r'\1.\2', json_str)
                json_str = re.sub(r'(["\']iceberg["\'])\s*\.\s*(["\']?iceberg["\']?)\s*\.\s*(["\']?[a-zA-Z_]+["\']?)', r'\1.\2.\3', json_str)
                
                # Fix missing quotes around property names
                json_str = re.sub(r'([{,]\s*)([a-zA-Z_]+)(\s*:)', r'\1"\2"\3', json_str)
                
                # Fix missing commas in arrays
                json_str = re.sub(r'(["\'])\s*\n\s*(["\'])', r'\1,\n\2', json_str)
                
                logger.debug(f"Cleaned JSON string: {json_str}")
                
                # Parse the cleaned JSON
                analysis_json = json.loads(json_str)
                
                logger.info(f"{Fore.GREEN}Successfully parsed analysis JSON{Fore.RESET}")
                logger.info(f"{Fore.CYAN}Analysis: {json.dumps(analysis_json, indent=2)}{Fore.RESET}")
                conversation_logger.log_trino_ai_processing("dba_agent_analysis_complete", {
                    "tables": analysis_json.get("tables", []),
                    "columns": analysis_json.get("columns", []),
                    "joins": analysis_json.get("joins", []),
                    "filters": analysis_json.get("filters", []),
                    "aggregations": analysis_json.get("aggregations", [])
                })
                
                # If we have table relationships in the workflow context, enhance the analysis
                if workflow_context and workflow_context.table_relationships:
                    enhanced_analysis = self.analyze_complex_relationships(
                        analysis_json.get("tables", []),
                        workflow_context.table_relationships
                    )
                    
                    # Add enhanced relationships to the analysis
                    analysis_json["enhanced_relationships"] = enhanced_analysis
                    
                    logger.info(f"{Fore.GREEN}Enhanced analysis with complex relationships{Fore.RESET}")
                    
                    # Update workflow context with enhanced analysis
                    if workflow_context:
                        workflow_context.add_metadata("enhanced_dba_analysis", enhanced_analysis)
                
                return {
                    "tables": analysis_json.get("tables", []),
                    "columns": analysis_json.get("columns", []),
                    "joins": analysis_json.get("joins", []),
                    "filters": analysis_json.get("filters", []),
                    "aggregations": analysis_json.get("aggregations", []),
                    "enhanced_relationships": analysis_json.get("enhanced_relationships", {}),
                    "requires_subquery": requires_subquery,
                    "schema_context": schema_context
                }
                
            except Exception as e:
                logger.error(f"{Fore.RED}Error parsing analysis JSON: {str(e)}{Fore.RESET}")
                logger.error(f"{Fore.RED}Raw analysis text: {analysis_text}{Fore.RESET}")
                conversation_logger.log_error("dba_agent", f"JSON parsing error: {str(e)}")
                
                # Return a best-effort analysis
                return {
                    "error": f"Failed to parse analysis: {str(e)}",
                    "raw_analysis": analysis_text,
                    "schema_context": schema_context
                }
                
        except Exception as e:
            logger.error(f"{Fore.RED}Error during DBA analysis: {str(e)}{Fore.RESET}")
            conversation_logger.log_error("dba_agent", f"Execution error: {str(e)}")
            return {"error": f"DBA analysis failed: {str(e)}"}
    
    def analyze_complex_relationships(self, tables: List[Any], relationships_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Identify complex relationships between tables for advanced joins
        
        Args:
            tables: List of tables from the analysis
            relationships_data: Table relationships data
            
        Returns:
            Dictionary of enhanced relationship information
        """
        logger.info(f"{Fore.CYAN}Analyzing complex relationships for {len(tables)} tables{Fore.RESET}")
        
        # Extract table names from the tables list
        table_names = []
        for table in tables:
            if isinstance(table, dict) and "name" in table:
                table_names.append(table["name"])
            elif isinstance(table, str):
                table_names.append(table)
        
        # Initialize results
        enhanced_relationships = {
            "direct_joins": [],
            "indirect_joins": [],
            "join_paths": [],
            "multi_hop_paths": []
        }
        
        # Process foreign key relationships
        fk_relationships = relationships_data.get("foreign_keys", [])
        for rel in fk_relationships:
            foreign_table = rel.get("foreign_table", "")
            primary_table = rel.get("primary_table", "")
            
            # Check if both tables are in our analysis
            if any(foreign_table.endswith(t) for t in table_names) and any(primary_table.endswith(t) for t in table_names):
                enhanced_relationships["direct_joins"].append({
                    "foreign_table": foreign_table,
                    "foreign_column": rel.get("foreign_column", ""),
                    "primary_table": primary_table,
                    "primary_column": rel.get("primary_column", ""),
                    "relationship_type": "foreign_key",
                    "confidence": "high"
                })
        
        # Process naming pattern relationships
        naming_relationships = relationships_data.get("naming_patterns", [])
        for rel in naming_relationships:
            foreign_table = rel.get("foreign_table", "")
            primary_table = rel.get("primary_table", "")
            
            # Check if both tables are in our analysis
            if any(foreign_table.endswith(t) for t in table_names) and any(primary_table.endswith(t) for t in table_names):
                enhanced_relationships["direct_joins"].append({
                    "foreign_table": foreign_table,
                    "foreign_column": rel.get("foreign_column", ""),
                    "primary_table": primary_table,
                    "primary_column": rel.get("primary_column", ""),
                    "relationship_type": "naming_pattern",
                    "confidence": rel.get("confidence", "medium")
                })
        
        # Process type-based relationships
        type_relationships = relationships_data.get("type_based", [])
        for rel in type_relationships:
            table1 = rel.get("table1", "")
            table2 = rel.get("table2", "")
            
            # Check if both tables are in our analysis
            if any(table1.endswith(t) for t in table_names) and any(table2.endswith(t) for t in table_names):
                enhanced_relationships["direct_joins"].append({
                    "table1": table1,
                    "column1": rel.get("column1", ""),
                    "table2": table2,
                    "column2": rel.get("column2", ""),
                    "relationship_type": "common_column",
                    "confidence": rel.get("confidence", "low")
                })
        
        # Identify multi-hop paths (indirect joins)
        # This is a simple implementation - in a real system, you'd use a graph algorithm
        direct_joins = enhanced_relationships["direct_joins"]
        for i, join1 in enumerate(direct_joins):
            for join2 in direct_joins[i+1:]:
                # Check if joins share a common table
                tables1 = [join1.get("foreign_table", ""), join1.get("primary_table", ""), 
                          join1.get("table1", ""), join1.get("table2", "")]
                tables2 = [join2.get("foreign_table", ""), join2.get("primary_table", ""),
                          join2.get("table1", ""), join2.get("table2", "")]
                
                # Filter out empty strings
                tables1 = [t for t in tables1 if t]
                tables2 = [t for t in tables2 if t]
                
                # Find common tables
                common_tables = set(tables1).intersection(set(tables2))
                
                if common_tables:
                    # These joins can be connected
                    for common_table in common_tables:
                        # Find the other tables in each join
                        other_tables1 = [t for t in tables1 if t != common_table]
                        other_tables2 = [t for t in tables2 if t != common_table]
                        
                        if other_tables1 and other_tables2:
                            enhanced_relationships["multi_hop_paths"].append({
                                "start_table": other_tables1[0],
                                "intermediate_table": common_table,
                                "end_table": other_tables2[0],
                                "path": [
                                    {"from": other_tables1[0], "to": common_table, "join_info": join1},
                                    {"from": common_table, "to": other_tables2[0], "join_info": join2}
                                ],
                                "confidence": "medium"
                            })
        
        logger.info(f"{Fore.GREEN}Found {len(enhanced_relationships['direct_joins'])} direct joins and {len(enhanced_relationships['multi_hop_paths'])} multi-hop paths{Fore.RESET}")
        return enhanced_relationships
    
    def consider_subqueries(self, query_intent: str) -> bool:
        """
        Determine if query likely requires subqueries or CTEs
        
        Args:
            query_intent: The natural language query
            
        Returns:
            Boolean indicating if subqueries are likely needed
        """
        subquery_indicators = self._identify_subquery_indicators(query_intent)
        return len(subquery_indicators) > 0
    
    def _identify_subquery_indicators(self, query_intent: str) -> List[str]:
        """
        Identify indicators that suggest subqueries or CTEs are needed
        
        Args:
            query_intent: The natural language query
            
        Returns:
            List of identified indicators
        """
        query_lower = query_intent.lower()
        
        # Define indicators that suggest subqueries
        subquery_indicators = [
            "for each", "compared to", "more than average", 
            "ranked", "top performing", "percentage",
            "ratio", "proportion", "per", "within each",
            "relative to", "versus", "against", "rank",
            "running total", "cumulative", "rolling",
            "previous", "next", "prior", "following",
            "year over year", "month over month", "growth rate"
        ]
        
        # Check for indicators
        found_indicators = []
        for indicator in subquery_indicators:
            if indicator in query_lower:
                found_indicators.append(indicator)
        
        logger.info(f"{Fore.CYAN}Identified subquery indicators: {found_indicators}{Fore.RESET}")
        return found_indicators
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for the DBA agent"""
        return """
        You are an expert Database Administrator and Data Analyst. Your task is to analyze natural language queries
        and identify the database objects (tables, columns, etc.) needed to answer them.
        
        You have deep knowledge of SQL, database schema design, and query optimization. You can identify:
        - Which tables are relevant to a query
        - Which columns are needed from each table
        - How tables should be joined
        - What filters should be applied
        - What aggregations or calculations are needed
        
        Your analysis should be thorough, precise, and focused on only the elements needed to answer the query.
        
        IMPORTANT: Always provide your analysis in the exact JSON format requested. Follow these strict formatting rules:
        1. Use proper JSON syntax with no extra spaces in property names or values
        2. For table names, use the format "iceberg.iceberg.table_name" (no spaces between dots)
        3. Make sure all strings are properly quoted
        4. Ensure all arrays and objects have matching brackets and braces
        5. Use commas correctly to separate items in arrays and properties in objects
        """

    def analyze_query(self, nlq: str, schema_context: str) -> Dict[str, Any]:
        """
        Analyze a natural language query to identify relevant tables, columns, etc.
        
        Args:
            nlq: The natural language query to analyze
            schema_context: The schema context to use for analysis
            
        Returns:
            A dictionary containing the analysis results
        """
        logger.info(f"{Fore.BLUE}DBA Agent analyzing query: {nlq}{Fore.RESET}")
        logger.info(f"{Fore.BLUE}DBA Agent using schema context: {schema_context[:100]}...{Fore.RESET}")
        
        # Log the agent action
        conversation_logger.log_trino_ai_to_ollama(self.name, {
            "action": "analyze_query",
            "nlq": nlq,
            "schema_context": schema_context[:200] + "..." if len(schema_context) > 200 else schema_context
        })
        
        # Log the reasoning process in more detail
        logger.info(f"{Fore.BLUE}DBA Agent reasoning process:{Fore.RESET}")
        logger.info(f"{Fore.BLUE}1. Identifying query intent: {nlq}{Fore.RESET}")
        logger.info(f"{Fore.BLUE}2. Scanning available schema for relevant tables{Fore.RESET}")
        logger.info(f"{Fore.BLUE}3. Determining required columns and relationships{Fore.RESET}")
        logger.info(f"{Fore.BLUE}4. Identifying filters, aggregations and sort criteria{Fore.RESET}")
        
        # Check if query likely requires subqueries
        requires_subquery = self.consider_subqueries(nlq)
        if requires_subquery:
            logger.info(f"{Fore.BLUE}5. Query likely requires subqueries or CTEs{Fore.RESET}")
        
        # Build the prompt for the LLM
        system_prompt = self.get_system_prompt()
        user_prompt = f"""
        Natural Language Query: {nlq}
        
        Schema Context:
        {schema_context}
        
        Analyze the natural language query and identify the following:
        1. Tables needed to answer the query
        2. Columns needed from each table
        3. Any joins required between tables
        4. Any filters or conditions
        5. Any aggregations or groupings
        
        {f"Note that this query likely requires subqueries or CTEs. Consider how to structure the analysis to support this complexity." if requires_subquery else ""}
        
        Provide your analysis in JSON format.
        """
        
        # Get the analysis from the LLM
        logger.info(f"{Fore.YELLOW}Sending query to LLM for analysis{Fore.RESET}")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = self.ollama_client.chat_completion(messages, agent_name="dba_agent")
            
            if "error" in response:
                logger.error(f"{Fore.RED}Error from LLM: {response['error']}{Fore.RESET}")
                conversation_logger.log_error("dba_agent", f"LLM error: {response['error']}")
                return {"error": response["error"]}
            
            analysis_text = response.get("message", {}).get("content", "")
            logger.info(f"{Fore.GREEN}Received analysis from LLM ({len(analysis_text)} chars){Fore.RESET}")
            conversation_logger.log_ollama_to_trino_ai("dba_agent", analysis_text[:500] + "..." if len(analysis_text) > 500 else analysis_text)
            
            # Extract JSON from the response
            try:
                # Try to find JSON in the response
                json_match = re.search(r'```json\s*([\s\S]*?)\s*```', analysis_text)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    # Try to find any JSON-like structure
                    json_match = re.search(r'({[\s\S]*})', analysis_text)
                    if json_match:
                        json_str = json_match.group(1)
                    else:
                        # Just try to parse the whole thing
                        json_str = analysis_text
                
                # Clean up common formatting issues before parsing
                # Remove spaces between dots in table names (e.g., "iceberg". "table" -> "iceberg.table")
                json_str = re.sub(r'(["\']iceberg["\'])\s*\.\s*(["\']?[a-zA-Z_]+["\']?)', r'\1.\2', json_str)
                json_str = re.sub(r'(["\']iceberg["\'])\s*\.\s*(["\']?iceberg["\']?)\s*\.\s*(["\']?[a-zA-Z_]+["\']?)', r'\1.\2.\3', json_str)
                
                # Fix missing quotes around property names
                json_str = re.sub(r'([{,]\s*)([a-zA-Z_]+)(\s*:)', r'\1"\2"\3', json_str)
                
                # Fix missing commas in arrays
                json_str = re.sub(r'(["\'])\s*\n\s*(["\'])', r'\1,\n\2', json_str)
                
                logger.debug(f"Cleaned JSON string: {json_str}")
                
                # Parse the cleaned JSON
                analysis_json = json.loads(json_str)
                
                logger.info(f"{Fore.GREEN}Successfully parsed analysis JSON{Fore.RESET}")
                logger.info(f"{Fore.CYAN}Analysis: {json.dumps(analysis_json, indent=2)}{Fore.RESET}")
                conversation_logger.log_trino_ai_processing("dba_agent_analysis_complete", {
                    "tables": analysis_json.get("tables", []),
                    "columns": analysis_json.get("columns", []),
                    "joins": analysis_json.get("joins", []),
                    "filters": analysis_json.get("filters", []),
                    "aggregations": analysis_json.get("aggregations", [])
                })
                
                return {
                    "tables": analysis_json.get("tables", []),
                    "columns": analysis_json.get("columns", []),
                    "joins": analysis_json.get("joins", []),
                    "filters": analysis_json.get("filters", []),
                    "aggregations": analysis_json.get("aggregations", []),
                    "requires_subquery": requires_subquery,
                    "schema_context": schema_context
                }
                
            except Exception as e:
                logger.error(f"{Fore.RED}Error parsing analysis JSON: {str(e)}{Fore.RESET}")
                logger.error(f"{Fore.RED}Raw analysis text: {analysis_text}{Fore.RESET}")
                conversation_logger.log_error("dba_agent", f"JSON parsing error: {str(e)}")
                
                # Return a best-effort analysis
                return {
                    "error": f"Failed to parse analysis: {str(e)}",
                    "raw_analysis": analysis_text,
                    "schema_context": schema_context
                }
                
        except Exception as e:
            logger.error(f"{Fore.RED}Error during DBA analysis: {str(e)}{Fore.RESET}")
            conversation_logger.log_error("dba_agent", f"Execution error: {str(e)}")
            return {"error": f"DBA analysis failed: {str(e)}"} 