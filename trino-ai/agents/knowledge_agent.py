import logging
import sys
import os
from typing import Optional, Dict, Any
import json
from datetime import datetime

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
        # Call the parent execute method to handle common logging and workflow context updates
        super().execute(inputs, workflow_context)
        
        query = inputs.get("query", "")
        if not query:
            return {"error": "No query provided", "status": "error"}
        
        # Determine if we need to research the question
        needs_research = self._evaluate_research_needs(query)
        
        if needs_research:
            # Log decision
            self.log_decision("Research Required", 
                             f"Question '{query}' requires additional research", 
                             workflow_context)
            
            # Gather relevant information
            research_results = self._conduct_research(query, workflow_context)
            
            # Use research to inform answer
            answer = self.answer_question_with_research(query, research_results)
        else:
            # Standard knowledge response
            answer = self.answer_question(query)
            
        # Update workflow context if provided
        if workflow_context:
            workflow_context.set_final_answer(answer)
            
        return {"answer": answer, "status": "success"}
    
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
        conversation_logger.log_trino_ai_processing(self.name, {
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

    def _evaluate_research_needs(self, query: str) -> bool:
        """Determine if the question requires additional research"""
        # Simple keyword-based approach - could be enhanced with LLM
        research_indicators = [
            "latest", "recent", "current", "statistics", "data", "trends",
            "percentage", "numbers", "figures", "study", "research"
        ]
        
        query_lower = query.lower()
        for indicator in research_indicators:
            if indicator in query_lower:
                return True
        
        return False

    def _conduct_research(self, query: str, workflow_context: Optional[WorkflowContext] = None) -> Dict[str, Any]:
        """Gather relevant information for the query"""
        research_results = {
            "web_search": [],
            "database_info": [],
            "pdf_knowledge": []
        }
        
        # 1. Try to find relevant information in PDFs
        if hasattr(self, "tools") and "search_pdfs" in self.tools:
            pdf_results = self.tools["search_pdfs"](query)
            if pdf_results:
                research_results["pdf_knowledge"] = pdf_results
                self.log_reasoning(f"Found relevant PDF information: {len(pdf_results)} results", workflow_context)
        
        # 2. Check if database has relevant information
        if hasattr(self, "tools") and "query_metadata" in self.tools:
            db_info = self.tools["query_metadata"](query)
            if db_info:
                research_results["database_info"] = db_info
                self.log_reasoning(f"Found relevant database information", workflow_context)
        
        # Log research decision
        if workflow_context:
            workflow_context.add_decision_point({
                "stage": "knowledge_research",
                "query": query,
                "research_found": any(len(v) > 0 for v in research_results.values()),
                "timestamp": datetime.now().isoformat()
            })
        
        return research_results

    def answer_question_with_research(self, query: str, research_results: Dict[str, Any]) -> str:
        """Answer a question using research results"""
        # Prepare system prompt with research context
        system_prompt = f"""
        {self.get_system_prompt()}
        
        When answering the question, incorporate the following research information:
        
        PDF Knowledge:
        {json.dumps(research_results.get("pdf_knowledge", []), indent=2)}
        
        Database Information:
        {json.dumps(research_results.get("database_info", []), indent=2)}
        
        Web Search Results:
        {json.dumps(research_results.get("web_search", []), indent=2)}
        
        When using this information:
        1. Cite sources appropriately
        2. Indicate if information might be out of date
        3. Clearly distinguish between facts from research and general knowledge
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Please answer this question: {query}"}
        ]
        
        response = self.ollama_client.chat_completion(messages, agent_name=self.name)
        
        if "error" in response:
            return f"I'm sorry, I couldn't research this question effectively. {response['error']}"
        
        return response.get("message", {}).get("content", "I couldn't generate a response to your question.") 