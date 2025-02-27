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
        self.conversation_log = []  # Store conversation entries in memory
        
        # Create logs directory if it doesn't exist
        if self.log_to_file and not os.path.exists("logs"):
            os.makedirs("logs")
            
        # Initialize conversation log file
        self.log_file = f"logs/conversation-{self.conversation_id}.log"
        if self.log_to_file:
            with open(self.log_file, "w") as f:
                f.write(f"=== Conversation {self.conversation_id} started at {datetime.now().isoformat()} ===\n\n")
                
        logger.info(f"Conversation logger initialized with ID: {self.conversation_id}")
    
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
        message = f"[{timestamp}] TRINO → TRINO-AI: Function: {function_name}"
        message += f"\nQuery: {query}"
        
        # Store in memory
        self.conversation_log.append({
            "timestamp": timestamp,
            "type": "trino_to_trino_ai",
            "function": function_name,
            "query": query
        })
        
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
        message = f"[{timestamp}] TRINO-AI PROCESSING: Stage: {stage}"
        
        # Format details for better readability
        formatted_details = json.dumps(details, indent=2) if isinstance(details, dict) else str(details)
        if len(formatted_details) > 500:
            formatted_details = formatted_details[:500] + "... [truncated]"
            
        message += f"\nDetails: {formatted_details}"
        
        # Store in memory
        self.conversation_log.append({
            "timestamp": timestamp,
            "type": "trino_ai_processing",
            "stage": stage,
            "data": details
        })
        
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
        message = f"[{timestamp}] TRINO-AI → OLLAMA: Agent: {agent}"
        
        # Truncate prompt if too long
        display_prompt = prompt
        if len(display_prompt) > 500:
            display_prompt = display_prompt[:500] + "... [truncated]"
            
        message += f"\nPrompt: {display_prompt}"
        
        # Store in memory
        self.conversation_log.append({
            "timestamp": timestamp,
            "type": "trino_ai_to_ollama",
            "agent": agent,
            "prompt": prompt
        })
        
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
        message = f"[{timestamp}] OLLAMA → TRINO-AI: Agent: {agent}"
        
        # Truncate response if too long
        display_response = response
        if len(display_response) > 500:
            display_response = display_response[:500] + "... [truncated]"
            
        message += f"\nResponse: {display_response}"
        
        # Store in memory
        self.conversation_log.append({
            "timestamp": timestamp,
            "type": "ollama_to_trino_ai",
            "agent": agent,
            "response": response
        })
        
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
        message = f"[{timestamp}] TRINO-AI → TRINO: Function: {function_name}"
        
        # Format result for better readability
        formatted_result = json.dumps(result, indent=2) if isinstance(result, dict) else str(result)
        if len(formatted_result) > 500:
            formatted_result = formatted_result[:500] + "... [truncated]"
            
        message += f"\nResult: {formatted_result}"
        
        # Store in memory
        self.conversation_log.append({
            "timestamp": timestamp,
            "type": "trino_ai_to_trino",
            "function": function_name,
            "result": result
        })
        
        logger.info(message)
        self._write_to_file(f"[{timestamp}] TRINO-AI → TRINO: Function: {function_name}\nResult: {formatted_result}")
    
    def log_error(self, source: str, error_message: str, details: Optional[Any] = None):
        """
        Log an error
        
        Args:
            source: The source of the error
            error_message: The error message
            details: Optional details about the error
        """
        entry = {
            "type": "error",
            "timestamp": datetime.now().isoformat(),
            "source": source,
            "error_message": error_message,
            "details": details
        }
        
        self.conversation_log.append(entry)
        
        # Format for console output
        console_output = f"{Fore.RED}[ERROR] {source}: {error_message}{Style.RESET_ALL}"
        if details:
            console_output += f"\nDetails: {json.dumps(details, indent=2) if isinstance(details, (dict, list)) else str(details)}"
        
        print(console_output)
        self._write_to_file(console_output)
    
    def log_agent_reasoning(self, agent_name: str, reasoning_steps: List[Dict[str, Any]]):
        """
        Log detailed reasoning steps from an agent
        
        Args:
            agent_name: The name of the agent
            reasoning_steps: List of reasoning steps with explanations
        """
        entry = {
            "type": "agent_reasoning",
            "timestamp": datetime.now().isoformat(),
            "agent": agent_name,
            "reasoning_steps": reasoning_steps
        }
        
        self.conversation_log.append(entry)
        
        # Format for console output
        console_output = f"[REASONING] {agent_name}\n"
        for i, step in enumerate(reasoning_steps):
            console_output += f"  {i+1}. {step['description']}\n"
        
        print(console_output)
        self._write_to_file(console_output)
    
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
        
    def get_recent_workflow(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get the most recent workflow steps from the conversation log
        
        Args:
            limit: Maximum number of workflow steps to return
            
        Returns:
            A list of workflow steps with their associated metadata
        """
        # Filter to get only agent processing steps in chronological order
        workflow_steps = []
        
        for entry in self.conversation_log[-100:]:  # Look at the last 100 entries maximum
            if entry["type"] in ["trino_ai_processing", "trino_ai_to_ollama", "ollama_to_trino_ai"]:
                # Format the entry for display
                step = {
                    "timestamp": entry["timestamp"],
                    "agent": entry.get("agent", "system"),
                    "action": entry.get("action", entry.get("stage", entry["type"])),
                    "details": entry.get("data", {})
                }
                
                # Clean up and limit the size of large data fields
                if "schema_context" in step["details"]:
                    step["details"]["schema_context"] = step["details"]["schema_context"][:200] + "..." \
                        if len(step["details"]["schema_context"]) > 200 else step["details"]["schema_context"]
                        
                if "response" in step["details"] and isinstance(step["details"]["response"], str):
                    step["details"]["response"] = step["details"]["response"][:200] + "..." \
                        if len(step["details"]["response"]) > 200 else step["details"]["response"]
                
                workflow_steps.append(step)
        
        # Return the most recent steps, up to the limit
        return workflow_steps[-limit:] if workflow_steps else []
        
    def get_workflow(self, conversation_id: str = None) -> Dict[str, Any]:
        """
        Get the complete workflow for a specific conversation
        
        Args:
            conversation_id: The ID of the conversation to get the workflow for
                            (defaults to the current conversation)
                            
        Returns:
            A dictionary containing the complete workflow information
        """
        # For now, we only support getting the workflow for the current conversation
        if conversation_id is not None and conversation_id != self.conversation_id:
            return {"error": f"Conversation {conversation_id} not found"}
            
        return {
            "conversation_id": self.conversation_id,
            "workflow": self.get_recent_workflow(limit=100)
        }

# Create a singleton instance
conversation_logger = ConversationLogger() 