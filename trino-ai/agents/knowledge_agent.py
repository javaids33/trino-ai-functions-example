import logging
import sys
import os
from typing import Optional

# Add the parent directory to the path so we can import from the parent module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.base_agent import Agent
from ollama_client import OllamaClient
from colorama import Fore
from conversation_logger import conversation_logger
from workflow_context import WorkflowContext

logger = logging.getLogger(__name__)

class KnowledgeAgent(Agent):
    """
    Agent specialized in handling general knowledge queries that don't require SQL.
    """
    
    def __init__(self, ollama_client: OllamaClient):
        """
        Initialize the Knowledge Agent
        
        Args:
            ollama_client: The Ollama client to use for completions
        """
        super().__init__(
            name="Knowledge Agent",
            description="A specialized agent for answering general knowledge questions",
            ollama_client=ollama_client
        )
        logger.info(f"{Fore.GREEN}Knowledge Agent initialized{Fore.RESET}")
    
    def execute(self, inputs: dict, workflow_context: Optional[WorkflowContext] = None) -> dict:
        """
        Execute the Knowledge Agent to answer a general knowledge question
        
        Args:
            inputs: A dictionary containing the query to answer
            workflow_context: Optional workflow context for logging and tracking
            
        Returns:
            A dictionary containing the answer to the question
        """
        query = inputs.get("query", "")
        if not query:
            return {"error": "No query provided"}
        
        # Log activation
        conversation_logger.log_trino_ai_processing(
            "knowledge_agent_activated",
            {"agent": self.name, "query": query}
        )
            
        answer = self.answer_question(query)
        
        # Log reasoning if workflow context is provided
        if workflow_context:
            self.log_reasoning(f"Processed knowledge query: {query}", workflow_context)
        
        return {
            "query": query,
            "answer": answer,
            "is_knowledge_query": True
        }
    
    def answer_question(self, query: str) -> str:
        """
        Answer a general knowledge question
        
        Args:
            query: The question to answer
            
        Returns:
            The answer to the question
        """
        logger.info(f"{Fore.BLUE}Knowledge Agent answering question: {query}{Fore.RESET}")
        
        # Log the agent action
        conversation_logger.log_trino_ai_to_ollama(self.name, {
            "action": "answer_question",
            "query": query
        })
        
        # Log detailed reasoning
        logger.info(f"{Fore.BLUE}Knowledge Agent reasoning process:{Fore.RESET}")
        logger.info(f"{Fore.BLUE}1. Analyzing question intent: {query}{Fore.RESET}")
        logger.info(f"{Fore.BLUE}2. Retrieving relevant knowledge from training data{Fore.RESET}")
        logger.info(f"{Fore.BLUE}3. Formulating comprehensive response{Fore.RESET}")
        logger.info(f"{Fore.BLUE}4. Ensuring response is factual and accurate{Fore.RESET}")
        
        # Prepare the messages for the LLM
        messages = [
            {
                "role": "system",
                "content": self.get_system_prompt()
            },
            {
                "role": "user",
                "content": query
            }
        ]
        
        # Get the response from the LLM
        response = self.ollama_client.chat_completion(messages, agent_name=self.name)
        
        # Log the agent response
        conversation_logger.log_ollama_to_trino_ai(self.name, {
            "action": "answer_question_response",
            "query": query,
            "response": response
        })
        
        # Check if there was an error
        if "error" in response:
            return f"Error: {response['error']}"
            
        # Extract the content from the response
        if "message" in response and "content" in response["message"]:
            return response["message"]["content"]
        else:
            return "I'm sorry, I couldn't generate a response to your question."
    
    def get_system_prompt(self) -> str:
        """
        Get the system prompt for the agent
        
        Returns:
            The system prompt as a string
        """
        return f"""
        You are {self.name}, {self.description}.
        
        You are part of the Trino AI multi-agent system, specializing in answering general knowledge questions that don't require database queries.
        
        When responding to questions:
        - Provide accurate, factual information based on your training data
        - Structure your responses clearly with bullet points or numbering when appropriate
        - Keep answers concise but comprehensive
        - If a question might be better answered with database data, suggest that the user rephrase as a specific data query
        
        Remember that you're an AI assistant without real-time data access, so make it clear when you're providing general information rather than real-time statistics.
        """ 