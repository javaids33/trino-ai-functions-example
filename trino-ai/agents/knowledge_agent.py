import logging
import time
import sys
import os
from typing import Dict, Any, List, Optional

# Add the parent directory to the path so we can import from the parent module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.base_agent import Agent
from ollama_client import OllamaClient
from conversation_logger import conversation_logger
from embeddings import embedding_service

logger = logging.getLogger(__name__)

class KnowledgeAgent(Agent):
    """
    Agent for retrieving and synthesizing knowledge from various sources.
    """
    
    def __init__(self, name: str = "knowledge_agent", description: str = "Retrieves and synthesizes knowledge from various sources", ollama_client: Optional[OllamaClient] = None):
        """
        Initialize the Knowledge agent
        
        Args:
            name: The name of the agent
            description: A description of what the agent does
            ollama_client: An optional OllamaClient instance
        """
        super().__init__(name, description)
        self.ollama_client = ollama_client
        self.logger.info("Knowledge Agent initialized")
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieve and synthesize knowledge based on a query
        
        Args:
            inputs: A dictionary containing:
                - query: The knowledge query
                - max_results: Maximum number of knowledge items to return (default: 5)
                
        Returns:
            A dictionary containing:
                - knowledge: The synthesized knowledge
                - sources: The sources of the knowledge
        """
        query = inputs.get("query", "")
        max_results = inputs.get("max_results", 5)
        
        if not query:
            self.logger.error("No query provided to Knowledge Agent")
            return {
                "error": "No query provided"
            }
        
        self.logger.info(f"Knowledge Agent processing query: {query}")
        conversation_logger.log_trino_ai_processing("knowledge_retrieval_start", {
            "query": query,
            "max_results": max_results
        })
        
        try:
            # Retrieve relevant knowledge
            start_time = time.time()
            knowledge_items = self._retrieve_knowledge(query, max_results)
            retrieval_time = time.time() - start_time
            
            self.logger.info(f"Retrieved {len(knowledge_items)} knowledge items in {retrieval_time:.2f}s")
            
            # Synthesize knowledge
            synthesis_start_time = time.time()
            synthesis = self._synthesize_knowledge(query, knowledge_items)
            synthesis_time = time.time() - synthesis_start_time
            
            self.logger.info(f"Synthesized knowledge in {synthesis_time:.2f}s")
            conversation_logger.log_trino_ai_processing("knowledge_retrieval_complete", {
                "retrieval_time": retrieval_time,
                "synthesis_time": synthesis_time,
                "knowledge_items_count": len(knowledge_items)
            })
            
            return {
                "knowledge": synthesis,
                "sources": [item["source"] for item in knowledge_items]
            }
            
        except Exception as e:
            self.logger.error(f"Error in Knowledge Agent: {str(e)}")
            conversation_logger.log_error("knowledge_agent", f"Error in Knowledge Agent: {str(e)}")
            
            return {
                "error": f"Error in Knowledge Agent: {str(e)}"
            }
    
    def _retrieve_knowledge(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """
        Retrieve knowledge items relevant to the query
        
        Args:
            query: The knowledge query
            max_results: Maximum number of knowledge items to return
            
        Returns:
            A list of knowledge items
        """
        # In a real implementation, this would use a vector database or other knowledge retrieval system
        # For this example, we'll use a simple embedding-based retrieval
        
        self.logger.info(f"Retrieving knowledge for query: {query}")
        
        # Get embeddings for the query
        query_embedding = embedding_service.get_embedding(query)
        
        # Retrieve relevant knowledge items
        # This is a placeholder - in a real system, you would search a vector database
        knowledge_items = [
            {
                "content": "Trino is a distributed SQL query engine designed to query large data sets distributed over one or more heterogeneous data sources.",
                "source": "Trino Documentation",
                "relevance": 0.95
            },
            {
                "content": "Trino was formerly known as PrestoSQL and is a fork of the original Presto project.",
                "source": "Trino History",
                "relevance": 0.85
            },
            {
                "content": "Trino uses a coordinator-worker architecture and supports a wide variety of connectors to different data sources.",
                "source": "Trino Architecture Guide",
                "relevance": 0.80
            }
        ]
        
        # Sort by relevance and limit to max_results
        knowledge_items = sorted(knowledge_items, key=lambda x: x["relevance"], reverse=True)[:max_results]
        
        return knowledge_items
    
    def _synthesize_knowledge(self, query: str, knowledge_items: List[Dict[str, Any]]) -> str:
        """
        Synthesize knowledge items into a coherent response
        
        Args:
            query: The knowledge query
            knowledge_items: The knowledge items to synthesize
            
        Returns:
            The synthesized knowledge
        """
        if not self.ollama_client:
            raise ValueError("OllamaClient is required for knowledge synthesis")
        
        self.logger.info(f"Synthesizing knowledge for query: {query}")
        
        # Prepare the prompt for the LLM
        knowledge_context = "\n\n".join([f"Source: {item['source']}\nContent: {item['content']}" for item in knowledge_items])
        
        prompt = f"""
        You are a knowledgeable assistant. Synthesize the following information to answer the user's query.
        
        User query: {query}
        
        Knowledge items:
        {knowledge_context}
        
        Provide a comprehensive and accurate answer based on the knowledge items. If the knowledge items don't contain enough information to answer the query, acknowledge the limitations of your knowledge.
        """
        
        # Call the LLM to synthesize the knowledge
        synthesis = self.ollama_client.generate(prompt)
        
        return synthesis.strip()
    
    def get_parameters_schema(self) -> Dict[str, Any]:
        """
        Get the parameters schema for this agent
        
        Returns:
            The parameters schema for this agent
        """
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The knowledge query"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of knowledge items to return",
                    "default": 5
                }
            },
            "required": ["query"]
        } 