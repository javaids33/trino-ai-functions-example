import logging
import json
import time
import os
import uuid
import threading
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
        self.conversations = {}
        self.current_conversation_id = None
        self.current_context = None
        self._lock = threading.Lock()
        
        # Create logs directory if it doesn't exist
        if self.log_to_file and not os.path.exists("logs"):
            os.makedirs("logs")
            
        logger.info("Conversation logger initialized")
    
    def set_current_context(self, context):
        """Set the current workflow context"""
        self.current_context = context
        logger.debug("Set current workflow context")
    
    @property
    def conversation_id(self) -> Optional[str]:
        """Alias for current_conversation_id for backward compatibility"""
        return self.current_conversation_id
    
    @property
    def conversation_log(self) -> List[Dict[str, Any]]:
        """Get the logs for the current conversation"""
        if not self.current_conversation_id:
            return []
        return self.conversations.get(self.current_conversation_id, {}).get("logs", [])
    
    def start_conversation(self) -> str:
        """
        Start a new conversation
        
        Returns:
            The conversation ID
        """
        conversation_id = str(uuid.uuid4())
        self.conversations[conversation_id] = {
            "id": conversation_id,
            "start_time": time.time(),
            "logs": [],
            "workflow_context": None
        }
        self.current_conversation_id = conversation_id
        logger.info(f"Started conversation {conversation_id}")
        return conversation_id
    
    def get_current_conversation_id(self) -> Optional[str]:
        """
        Get the current conversation ID
        
        Returns:
            The current conversation ID, or None if no conversation is active
        """
        return self.current_conversation_id
    
    def _write_to_file(self, content: str):
        """Write content to the log file"""
        if self.log_to_file and self.current_conversation_id:
            log_dir = "logs"
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            with open(f"{log_dir}/conversation-{self.current_conversation_id}.log", "a") as f:
                f.write(f"{content}\n")
    
    def log_trino_to_trino_ai(self, log_type: str, data: Any) -> None:
        """
        Log a message from Trino to Trino AI
        
        Args:
            log_type: The type of log
            data: The data to log
        """
        self._log_event("trino_to_trino_ai", log_type, data)
    
    def log_trino_request(self, function_name: str, content: Any) -> None:
        """
        Log a request from Trino to Trino AI
        
        Args:
            function_name: The name of the function being called
            content: The content of the request
        """
        self._log_event("trino_request", function_name, content)
    
    def log_trino_ai_to_trino(self, log_type: str, data: Any) -> None:
        """
        Log a message from Trino AI to Trino
        
        Args:
            log_type: The type of log
            data: The data to log
        """
        self._log_event("trino_ai_to_trino", log_type, data)
    
    def log_trino_ai_to_ollama(self, agent_name: str, data: Any) -> None:
        """
        Log a message from Trino AI to Ollama
        
        Args:
            agent_name: The name of the agent sending the message
            data: The data to log
        """
        self._log_event("trino_ai_to_ollama", agent_name, data)
    
    def log_ollama_to_trino_ai(self, agent_name: str, data: Any) -> None:
        """
        Log a message from Ollama to Trino AI
        
        Args:
            agent_name: The name of the agent receiving the message
            data: The data to log
        """
        self._log_event("ollama_to_trino_ai", agent_name, data)
    
    def log_trino_ai_processing(self, log_type: str, data: Any) -> None:
        """
        Log a processing event within Trino AI
        
        Args:
            log_type: The type of log
            data: The data to log
        """
        self._log_event("trino_ai_processing", log_type, data)
    
    def log_error(self, source: str, message: str, error: Any = None) -> None:
        """
        Log an error
        
        Args:
            source: The source of the error
            message: The error message
            error: The error object or additional error details
        """
        error_data = {"message": message}
        if error:
            error_data["error"] = str(error)
        self._log_event("error", source, error_data)
    
    def log_nl2sql_conversion(self, query: str, sql: str, metadata: Dict[str, Any]) -> None:
        """
        Log a natural language to SQL conversion
        
        Args:
            query: The natural language query
            sql: The generated SQL
            metadata: Additional metadata about the conversion
        """
        data = {
            "query": query,
            "sql": sql,
            **metadata
        }
        self._log_event("nl2sql_conversion", "conversion", data)
    
    def update_workflow_context(self, workflow_context: Dict[str, Any]) -> None:
        """
        Update the workflow context for the current conversation
        
        Args:
            workflow_context: The workflow context to update
        """
        if not self.current_conversation_id:
            logger.warning("No active conversation to update workflow context")
            return
            
        self.conversations[self.current_conversation_id]["workflow_context"] = workflow_context
        logger.info(f"Updated workflow context for conversation {self.current_conversation_id}")
        
        # Log the workflow context update
        self._log_event("workflow_context_update", "update", {
            "timestamp": datetime.now().isoformat(),
            "context_size": len(json.dumps(workflow_context))
        })
        
        # Write the workflow context to a separate file for easier access
        if self.log_to_file:
            log_dir = "logs"
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            with open(f"{log_dir}/workflow-context-{self.current_conversation_id}.json", "w") as f:
                json.dump(workflow_context, f, indent=2)
    
    def get_workflow(self, conversation_id: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
        """
        Get the workflow for a conversation
        
        Args:
            conversation_id: The conversation ID, or None for the current conversation
            
        Returns:
            The workflow, or None if the conversation doesn't exist
        """
        # Use the current conversation if no ID is provided
        if conversation_id is None:
            conversation_id = self.current_conversation_id
            
        if not conversation_id or conversation_id not in self.conversations:
            logger.warning(f"No conversation found with ID {conversation_id}")
            return None
            
        return self.conversations[conversation_id]["logs"]
    
    def get_workflow_context(self, conversation_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get the workflow context for a conversation
        
        Args:
            conversation_id: The conversation ID, or None for the current conversation
            
        Returns:
            The workflow context, or None if the conversation doesn't exist or has no context
        """
        # Use the current conversation if no ID is provided
        if conversation_id is None:
            conversation_id = self.current_conversation_id
            
        if not conversation_id or conversation_id not in self.conversations:
            logger.warning(f"No conversation found with ID {conversation_id}")
            return None
            
        return self.conversations[conversation_id].get("workflow_context")
    
    def get_all_conversations(self) -> List[Dict[str, Any]]:
        """
        Get a list of all conversations with summary information
        
        Returns:
            A list of conversation summaries
        """
        result = []
        for conv_id, conv_data in self.conversations.items():
            # Find the original query
            original_query = "Unknown query"
            for log in conv_data.get("logs", []):
                # Check for translation_agent_activated logs which contain the query
                if log["type"] == "trino_ai_processing" and log["log_type"] == "translation_agent_activated" and "query" in log.get("data", {}):
                    original_query = log["data"]["query"]
                    break
                    
            result.append({
                "id": conv_id,
                "timestamp": conv_data.get("start_time", 0),
                "original_query": original_query,
                "log_count": len(conv_data.get("logs", []))
            })
        
        # Sort by timestamp, newest first
        result.sort(key=lambda x: x["timestamp"], reverse=True)
        return result
    
    def get_conversation_summary(self) -> str:
        """Get a summary of the conversation for debugging"""
        summary = f"{Fore.MAGENTA}Conversation Summary ({len(self.conversation_log)} events):{Fore.RESET}\n"
        
        for event in self.conversation_log[-10:]:  # Show last 10 events only
            summary += f"{Fore.BLUE}[{event['formatted_time']}] {Fore.GREEN}{event['from']} → {event['to']}{Fore.RESET}: {event['type']}\n"
            
        return summary
    
    def _log_event(self, event_type: str, log_type: str, data: Any) -> None:
        """
        Log an event
        
        Args:
            event_type: The type of event (trino_to_trino_ai, trino_ai_to_ollama, etc.)
            log_type: The specific type of log (nlq, agent_name, etc.)
            data: The data to log
        """
        if not self.current_conversation_id:
            logger.warning("No active conversation for logging")
            self.start_conversation()
            
        timestamp = time.time()
        formatted_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        
        # Determine from and to entities based on event type
        from_entity = ""
        to_entity = ""
        
        if event_type == "trino_to_trino_ai":
            from_entity = "trino"
            to_entity = "trino_ai"
        elif event_type == "trino_request":
            from_entity = "trino"
            to_entity = "trino_ai"
        elif event_type == "trino_ai_to_trino":
            from_entity = "trino_ai"
            to_entity = "trino"
        elif event_type == "trino_ai_to_ollama":
            from_entity = "trino_ai"
            to_entity = log_type  # Agent name
        elif event_type == "ollama_to_trino_ai":
            from_entity = "ollama"
            to_entity = log_type  # Agent name
        elif event_type == "trino_ai_processing":
            from_entity = "trino_ai"
            to_entity = "trino_ai"
        elif event_type == "error":
            from_entity = log_type  # Source of error
            to_entity = "trino_ai"
        elif event_type == "nl2sql_conversion":
            from_entity = "trino_ai"
            to_entity = "trino"
        
        # Create the log entry
        log_entry = {
            "timestamp": timestamp,
            "formatted_time": formatted_time,
            "type": event_type,
            "log_type": log_type,
            "from": from_entity,
            "to": to_entity,
            "data": data
        }
        
        with self._lock:
            # Add to conversation logs
            self.conversations[self.current_conversation_id]["logs"].append(log_entry)
            
            # If we have a current context, also add to its conversation history
            if self.current_context:
                message_data = data
                if not isinstance(data, dict):
                    message_data = {"content": str(data)}
                
                self.current_context.add_to_conversation(
                    sender=from_entity,
                    recipient=to_entity,
                    message=message_data,
                    message_type=event_type
                )
        
        # Log to console
        if isinstance(data, dict) and "query" in data:
            log_message = f"{Fore.CYAN}[{formatted_time}] {from_entity} → {to_entity}: {event_type} - {log_type} - {data.get('query', '')[:100]}{Fore.RESET}"
        elif isinstance(data, dict) and "content" in data:
            log_message = f"{Fore.CYAN}[{formatted_time}] {from_entity} → {to_entity}: {event_type} - {log_type} - {data.get('content', '')[:100]}{Fore.RESET}"
        elif isinstance(data, str):
            log_message = f"{Fore.CYAN}[{formatted_time}] {from_entity} → {to_entity}: {event_type} - {log_type} - {data[:100]}{Fore.RESET}"
        else:
            log_message = f"{Fore.CYAN}[{formatted_time}] {from_entity} → {to_entity}: {event_type} - {log_type}{Fore.RESET}"
            
        logger.debug(log_message)
        
        # Write to file
        if self.log_to_file:
            try:
                file_content = json.dumps(log_entry, default=str)
                self._write_to_file(file_content)
            except Exception as e:
                logger.error(f"Error writing to log file: {str(e)}")

# Singleton instance
conversation_logger = ConversationLogger() 