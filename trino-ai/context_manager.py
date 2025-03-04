import logging
import json
from datetime import datetime
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class WorkflowContext:
    """Manages shared context across the agent workflow"""
    
    def __init__(self):
        self.conversation_id = f"conv-{int(datetime.now().timestamp())}"
        self.metadata = {}
        self.query_history = []
        self.agent_reasoning = {}
        self.schema_context = ""
        self.metadata_used = {}
        self.decision_points = []
        self.start_time = datetime.now()
        self.query = ""
        self.sql = ""
        self.is_data_query = None
        
    def add_metadata(self, step_name: str, metadata: Dict[str, Any]):
        """Add metadata for a specific workflow step"""
        self.metadata[step_name] = metadata
        logger.debug(f"Added metadata for step '{step_name}'")
        
    def add_agent_reasoning(self, agent_name: str, reasoning: str):
        """Track agent reasoning"""
        if agent_name not in self.agent_reasoning:
            self.agent_reasoning[agent_name] = []
        self.agent_reasoning[agent_name].append({
            "timestamp": datetime.now().isoformat(),
            "reasoning": reasoning
        })
        logger.debug(f"Added reasoning for agent '{agent_name}'")
        
    def mark_metadata_used(self, agent_name: str, metadata_keys: List[str]):
        """Track which metadata was used by which agent"""
        if agent_name not in self.metadata_used:
            self.metadata_used[agent_name] = []
        self.metadata_used[agent_name].extend(metadata_keys)
        logger.debug(f"Marked metadata used by agent '{agent_name}': {metadata_keys}")
        
    def add_decision_point(self, agent: str, decision: str, rationale: str):
        """Track key decisions in the workflow"""
        self.decision_points.append({
            "agent": agent,
            "decision": decision,
            "rationale": rationale,
            "timestamp": datetime.now().isoformat()
        })
        logger.debug(f"Added decision point for agent '{agent}': {decision}")
        
    def set_query(self, query: str):
        """Set the original natural language query"""
        self.query = query
        self.add_metadata("input_query", {
            "query": query,
            "timestamp": datetime.now().isoformat()
        })
        
    def set_sql(self, sql: str):
        """Set the generated SQL query"""
        self.sql = sql
        self.add_metadata("generated_sql", {
            "sql": sql,
            "timestamp": datetime.now().isoformat()
        })
        
    def set_data_query_status(self, is_data_query: bool):
        """Set whether this is a data query or knowledge query"""
        self.is_data_query = is_data_query
        self.add_metadata("query_type", {
            "is_data_query": is_data_query,
            "timestamp": datetime.now().isoformat()
        })
        
    def get_full_context(self) -> Dict[str, Any]:
        """Get the complete context object for logging/debugging"""
        return {
            "conversation_id": self.conversation_id,
            "query": self.query,
            "sql": self.sql,
            "is_data_query": self.is_data_query,
            "start_time": self.start_time.isoformat(),
            "duration": (datetime.now() - self.start_time).total_seconds(),
            "metadata": self.metadata,
            "query_history": self.query_history,
            "agent_reasoning": self.agent_reasoning,
            "metadata_used": self.metadata_used,
            "decision_points": self.decision_points,
            "schema_context_length": len(self.schema_context) if self.schema_context else 0
        } 