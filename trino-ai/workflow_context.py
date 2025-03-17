import time
from typing import Dict, Any, List, Optional
import copy
import uuid
import logging

logger = logging.getLogger(__name__)

class WorkflowContext:
    """Context object to track the workflow and decision-making process"""
    
    def __init__(self):
        self.context_id = str(uuid.uuid4())
        self.created_at = time.time()
        self.query = ""
        self.sql = ""
        self.is_data_query = False
        self.schema_context = ""
        self.metadata = {}
        self.decision_points = []
        
        # Enhanced conversation tracking
        self.conversation_history = []
        self.agent_reasoning = {}
        self.table_relationships = {}
        
        # Enhanced tracking for complex queries
        self.reasoning_chain = []
        self.errors = []
        self.query_decomposition = {}
        self.schema_usage = {}
        
    def add_to_conversation(self, sender: str, recipient: str, message: Dict[str, Any], message_type: str = "message"):
        """
        Add a message to the conversation history
        
        Args:
            sender: The sender of the message (e.g., "user", "dba_agent", "ollama")
            recipient: The recipient of the message
            message: The message content
            message_type: The type of message (message, error, thinking, etc.)
        """
        entry = {
            "timestamp": time.time(),
            "sender": sender,
            "recipient": recipient,
            "message": message,
            "type": message_type
        }
        self.conversation_history.append(entry)
        logger.debug(f"Added conversation entry: {sender} → {recipient}")
        return entry
    
    def get_conversation_for_agent(self, agent_name: str) -> List[Dict[str, Any]]:
        """
        Get conversation history filtered for a specific agent
        
        Args:
            agent_name: The name of the agent
            
        Returns:
            A list of conversation entries relevant to the agent
        """
        # Include messages to/from this agent and global messages
        return [
            entry for entry in self.conversation_history
            if (entry["sender"] == agent_name or 
                entry["recipient"] == agent_name or 
                entry["recipient"] == "all" or
                entry["sender"] == "user")
        ]
    
    def get_full_conversation(self) -> List[Dict[str, Any]]:
        """Get the full conversation history"""
        return self.conversation_history
    
    def add_agent_reasoning(self, agent_name: str, reasoning: str):
        """Add an agent's reasoning process"""
        self.agent_reasoning[agent_name] = reasoning
        logger.debug(f"Added reasoning for agent {agent_name}")
    
    def add_table_relationships(self, relationships_data: Dict[str, Any]):
        """Add information about table relationships to context"""
        self.table_relationships = relationships_data
        self.add_metadata("table_relationships", relationships_data)
    
    def track_query_decomposition(self, decomposition: Dict[str, Any]):
        """
        Track how a complex query was broken down into components
        
        Args:
            decomposition: Dictionary containing query decomposition information
        """
        self.query_decomposition = decomposition
        self.add_metadata("query_decomposition", decomposition)
        logger.debug(f"Tracked query decomposition with {len(decomposition)} components")
    
    def record_schema_usage(self, tables_used: List[str], columns_used: List[Dict[str, str]], relationships_used: List[Dict[str, Any]]):
        """
        Record which schema elements were actually used in the final query
        
        Args:
            tables_used: List of tables used in the query
            columns_used: List of columns used in the query
            relationships_used: List of relationships used in the query
        """
        self.schema_usage = {
            "tables": tables_used,
            "columns": columns_used,
            "relationships": relationships_used
        }
        self.add_metadata("schema_usage", self.schema_usage)
        logger.debug(f"Recorded schema usage: {len(tables_used)} tables, {len(columns_used)} columns, {len(relationships_used)} relationships")
    
    def track_reasoning_chain(self, reasoning_steps: List[Dict[str, Any]]):
        """
        Track the chain of reasoning that led to the final SQL
        
        Args:
            reasoning_steps: List of reasoning steps
        """
        self.reasoning_chain.extend(reasoning_steps)
        self.add_metadata("reasoning_chain", self.reasoning_chain)
        logger.debug(f"Added {len(reasoning_steps)} reasoning steps, total: {len(self.reasoning_chain)}")
    
    def add_error(self, source: str, error_message: str, error_type: str = "general"):
        """
        Add an error to the context
        
        Args:
            source: Source of the error (e.g., agent name)
            error_message: The error message
            error_type: Type of error
        """
        error = {
            "timestamp": time.time(),
            "source": source,
            "message": error_message,
            "type": error_type
        }
        self.errors.append(error)
        logger.debug(f"Added error from {source}: {error_message}")
    
    def get_errors(self, source: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get errors from the context
        
        Args:
            source: Optional source to filter by
            
        Returns:
            List of errors
        """
        if source:
            return [error for error in self.errors if error["source"] == source]
        return self.errors
    
    def set_table_relationships(self, relationships: Dict[str, Any]):
        """
        Set table relationships
        
        Args:
            relationships: Dictionary of table relationships
        """
        self.table_relationships = relationships
        self.add_metadata("table_relationships", relationships)
        logger.debug(f"Set table relationships")
    
    # Existing methods
    def set_query(self, query: str):
        """Set the natural language query"""
        self.query = query
        logger.debug(f"Set query: {query}")
        
    def set_sql(self, sql: str):
        """Set the generated SQL query"""
        self.sql = sql
        logger.debug(f"Set SQL: {sql}")
        
    def set_data_query_status(self, is_data_query: bool):
        """Set whether this is a data query or a general knowledge query"""
        self.is_data_query = is_data_query
        logger.debug(f"Set is_data_query: {is_data_query}")
        
    def add_metadata(self, key: str, value: Any):
        """Add a metadata item to the context"""
        self.metadata[key] = value
        logger.debug(f"Added metadata for key {key}")
        
    def add_decision_point(self, agent: str, decision: str, reason: str):
        """Add a decision point to the workflow"""
        self.decision_points.append({
            "timestamp": time.time(),
            "agent": agent,
            "decision": decision,
            "reason": reason
        })
        logger.debug(f"Added decision point for agent {agent}: {decision}")
        
    def set_schema_context(self, schema_context: str):
        """Set the schema context"""
        self.schema_context = schema_context
        logger.debug(f"Set schema context")
        
    def mark_metadata_used(self, agent_name: str, metadata_keys: List[str]):
        """
        Mark metadata as used by an agent
        
        Args:
            agent_name: The name of the agent
            metadata_keys: The keys of the metadata used
        """
        # This could be used to track which metadata is used by which agents
        logger.debug(f"Agent {agent_name} used metadata: {', '.join(metadata_keys)}")
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """
        Get a metadata item from the context
        
        Args:
            key: The metadata key
            default: Default value if key doesn't exist
            
        Returns:
            The metadata value or default
        """
        return self.metadata.get(key, default)
        
    def get_full_context(self) -> Dict[str, Any]:
        """Get the full context including all tracked information"""
        return {
            "context_id": self.context_id,
            "created_at": self.created_at,
            "query": self.query,
            "sql": self.sql,
            "is_data_query": self.is_data_query,
            "schema_context": self.schema_context,
            "metadata": self.metadata,
            "decision_points": self.decision_points,
            "conversation_history": self.conversation_history,
            "agent_reasoning": self.agent_reasoning,
            "table_relationships": self.table_relationships,
            "reasoning_chain": self.reasoning_chain,
            "errors": self.errors,
            "query_decomposition": self.query_decomposition,
            "schema_usage": self.schema_usage
        } 