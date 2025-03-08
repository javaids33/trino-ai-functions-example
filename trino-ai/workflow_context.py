from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

class WorkflowContext:
    """
    Maintains context for a workflow, including metadata, agent reasoning, and decision points.
    This allows for tracking the workflow and providing explanations for decisions.
    """

    def __init__(self):
        """Initialize an empty workflow context"""
        self.metadata = {}
        self.agent_reasoning = {}
        self.decision_points = []
        self.query = None
        self.sql = None
        self.is_data_query = None
        self.schema_context = None

    def add_metadata(self, step_name: str, metadata: Dict[str, Any]):
        """
        Add metadata for a step in the workflow
        
        Args:
            step_name: The name of the step
            metadata: The metadata to add
        """
        self.metadata[step_name] = metadata
        logger.debug(f"Added metadata for step {step_name}")

    def add_agent_reasoning(self, agent_name: str, reasoning: str):
        """
        Add reasoning from an agent
        
        Args:
            agent_name: The name of the agent
            reasoning: The reasoning from the agent
        """
        self.agent_reasoning[agent_name] = reasoning
        logger.debug(f"Added reasoning for agent {agent_name}")

    def mark_metadata_used(self, agent_name: str, metadata_keys: List[str]):
        """
        Mark metadata as used by an agent
        
        Args:
            agent_name: The name of the agent
            metadata_keys: The keys of the metadata used
        """
        # This could be used to track which metadata is used by which agents
        logger.debug(f"Agent {agent_name} used metadata: {', '.join(metadata_keys)}")

    def add_decision_point(self, agent: str, decision: str, rationale: str):
        """
        Add a decision point to the workflow
        
        Args:
            agent: The agent making the decision
            decision: The decision made
            rationale: The rationale for the decision
        """
        self.decision_points.append({
            "agent": agent,
            "decision": decision,
            "explanation": rationale
        })
        logger.debug(f"Added decision point for agent {agent}: {decision}")

    def set_query(self, query: str):
        """
        Set the natural language query
        
        Args:
            query: The natural language query
        """
        self.query = query
        logger.debug(f"Set query: {query}")

    def set_sql(self, sql: str):
        """
        Set the generated SQL
        
        Args:
            sql: The generated SQL
        """
        self.sql = sql
        logger.debug(f"Set SQL: {sql}")

    def set_data_query_status(self, is_data_query: bool):
        """
        Set whether the query is a data query or a knowledge query
        
        Args:
            is_data_query: Whether the query is a data query
        """
        self.is_data_query = is_data_query
        logger.debug(f"Set is_data_query: {is_data_query}")
        
    def set_schema_context(self, schema_context: str):
        """
        Set the schema context
        
        Args:
            schema_context: The schema context
        """
        self.schema_context = schema_context
        logger.debug(f"Set schema context")

    def get_full_context(self) -> Dict[str, Any]:
        """
        Get the full workflow context
        
        Returns:
            The full workflow context
        """
        return {
            "metadata": self.metadata,
            "agent_reasoning": self.agent_reasoning,
            "decision_points": self.decision_points,
            "query": self.query,
            "sql": self.sql,
            "is_data_query": self.is_data_query,
            "schema_context": self.schema_context
        } 