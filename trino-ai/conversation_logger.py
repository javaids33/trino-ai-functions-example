import logging
import json
import time
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from colorama import Fore, Back, Style

# Configure logging
logger = logging.getLogger(__name__)

class ConversationLogger:
    """
    Logs the full conversation flow between Trino, Trino-AI, and Ollama
    """
    
    def __init__(self, log_to_file: bool = True):
        """
        Initialize the conversation logger
        
        Args:
            log_to_file: Whether to log conversations to a file in addition to console
        """
        self.log_to_file = log_to_file
        self.conversation_id = f"conv-{int(time.time())}"
        
        # Create logs directory if it doesn't exist
        if self.log_to_file and not os.path.exists("logs"):
            os.makedirs("logs")
            
        # Initialize conversation log file
        self.log_file = f"logs/conversation-{self.conversation_id}.log"
        if self.log_to_file:
            with open(self.log_file, "w") as f:
                f.write(f"=== Conversation {self.conversation_id} started at {datetime.now().isoformat()} ===\n\n")
                
        logger.info(f"{Fore.CYAN}Conversation logger initialized with ID: {self.conversation_id}{Fore.RESET}")
    
    def _write_to_file(self, content: str):
        """Write content to the log file"""
        if self.log_to_file:
            with open(self.log_file, "a") as f:
                f.write(f"{content}\n")
    
    def log_trino_request(self, function_name: str, query: str):
        """
        Log a request from Trino to Trino-AI
        
        Args:
            function_name: The AI function being called
            query: The query or content of the request
        """
        timestamp = datetime.now().isoformat()
        message = f"\n{Back.BLUE}{Fore.WHITE}[{timestamp}] TRINO → TRINO-AI: Function: {function_name}{Style.RESET_ALL}\n"
        message += f"{Fore.BLUE}Query: {query}{Style.RESET_ALL}\n"
        
        logger.info(message)
        self._write_to_file(f"[{timestamp}] TRINO → TRINO-AI: Function: {function_name}\nQuery: {query}")
    
    def log_trino_ai_processing(self, stage: str, details: Dict[str, Any]):
        """
        Log processing within Trino-AI
        
        Args:
            stage: The processing stage (e.g., "schema_context", "dba_analysis")
            details: Details about the processing
        """
        timestamp = datetime.now().isoformat()
        message = f"\n{Back.CYAN}{Fore.BLACK}[{timestamp}] TRINO-AI PROCESSING: Stage: {stage}{Style.RESET_ALL}\n"
        
        # Format details for better readability
        formatted_details = json.dumps(details, indent=2) if isinstance(details, dict) else str(details)
        if len(formatted_details) > 500:
            formatted_details = formatted_details[:500] + "... [truncated]"
            
        message += f"{Fore.CYAN}Details: {formatted_details}{Style.RESET_ALL}\n"
        
        logger.info(message)
        self._write_to_file(f"[{timestamp}] TRINO-AI PROCESSING: Stage: {stage}\nDetails: {formatted_details}")
    
    def log_trino_ai_to_ollama(self, agent: str, prompt: str):
        """
        Log a request from Trino-AI to Ollama
        
        Args:
            agent: The agent making the request
            prompt: The prompt being sent to Ollama
        """
        timestamp = datetime.now().isoformat()
        message = f"\n{Back.MAGENTA}{Fore.WHITE}[{timestamp}] TRINO-AI → OLLAMA: Agent: {agent}{Style.RESET_ALL}\n"
        
        # Truncate prompt if too long
        display_prompt = prompt
        if len(display_prompt) > 500:
            display_prompt = display_prompt[:500] + "... [truncated]"
            
        message += f"{Fore.MAGENTA}Prompt: {display_prompt}{Style.RESET_ALL}\n"
        
        logger.info(message)
        self._write_to_file(f"[{timestamp}] TRINO-AI → OLLAMA: Agent: {agent}\nPrompt: {prompt}")
    
    def log_ollama_to_trino_ai(self, agent: str, response: str):
        """
        Log a response from Ollama to Trino-AI
        
        Args:
            agent: The agent receiving the response
            response: The response from Ollama
        """
        timestamp = datetime.now().isoformat()
        message = f"\n{Back.YELLOW}{Fore.BLACK}[{timestamp}] OLLAMA → TRINO-AI: Agent: {agent}{Style.RESET_ALL}\n"
        
        # Truncate response if too long
        display_response = response
        if len(display_response) > 500:
            display_response = display_response[:500] + "... [truncated]"
            
        message += f"{Fore.YELLOW}Response: {display_response}{Style.RESET_ALL}\n"
        
        logger.info(message)
        self._write_to_file(f"[{timestamp}] OLLAMA → TRINO-AI: Agent: {agent}\nResponse: {response}")
    
    def log_trino_ai_to_trino(self, function_name: str, result: Any):
        """
        Log a response from Trino-AI back to Trino
        
        Args:
            function_name: The AI function that was called
            result: The result being returned to Trino
        """
        timestamp = datetime.now().isoformat()
        message = f"\n{Back.GREEN}{Fore.BLACK}[{timestamp}] TRINO-AI → TRINO: Function: {function_name}{Style.RESET_ALL}\n"
        
        # Format result for better readability
        formatted_result = json.dumps(result, indent=2) if isinstance(result, dict) else str(result)
        if len(formatted_result) > 500:
            formatted_result = formatted_result[:500] + "... [truncated]"
            
        message += f"{Fore.GREEN}Result: {formatted_result}{Style.RESET_ALL}\n"
        
        logger.info(message)
        self._write_to_file(f"[{timestamp}] TRINO-AI → TRINO: Function: {function_name}\nResult: {formatted_result}")
    
    def log_error(self, source: str, error_message: str, details: Optional[Any] = None):
        """
        Log an error in the conversation flow
        
        Args:
            source: The source of the error (e.g., "trino", "trino-ai", "ollama")
            error_message: The error message
            details: Additional error details
        """
        timestamp = datetime.now().isoformat()
        message = f"\n{Back.RED}{Fore.WHITE}[{timestamp}] ERROR in {source.upper()}: {error_message}{Style.RESET_ALL}\n"
        
        if details:
            formatted_details = json.dumps(details, indent=2) if isinstance(details, dict) else str(details)
            message += f"{Fore.RED}Details: {formatted_details}{Style.RESET_ALL}\n"
        
        logger.error(message)
        self._write_to_file(f"[{timestamp}] ERROR in {source.upper()}: {error_message}")
        if details:
            self._write_to_file(f"Details: {details}")
    
    def get_conversation_summary(self) -> str:
        """
        Get a summary of the conversation
        
        Returns:
            A summary of the conversation
        """
        if self.log_to_file and os.path.exists(self.log_file):
            with open(self.log_file, "r") as f:
                content = f.read()
                
            # Count interactions
            trino_requests = content.count("TRINO → TRINO-AI")
            ollama_requests = content.count("TRINO-AI → OLLAMA")
            ollama_responses = content.count("OLLAMA → TRINO-AI")
            trino_responses = content.count("TRINO-AI → TRINO")
            errors = content.count("ERROR in")
            
            summary = f"""
=== Conversation {self.conversation_id} Summary ===
- Trino requests to Trino-AI: {trino_requests}
- Trino-AI requests to Ollama: {ollama_requests}
- Ollama responses to Trino-AI: {ollama_responses}
- Trino-AI responses to Trino: {trino_responses}
- Errors: {errors}
"""
            return summary
        
        return f"No summary available for conversation {self.conversation_id}"

# Create a singleton instance
conversation_logger = ConversationLogger() 