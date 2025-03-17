import logging
import json
import os
from typing import Dict, Any, List, Optional
import numpy as np
from colorama import Fore

logger = logging.getLogger(__name__)

class ExemplarsManager:
    """Manages exemplar NLQ-SQL pairs for few-shot learning"""
    
    def __init__(self, exemplars_file: Optional[str] = None, embeddings_module=None):
        """
        Initialize the exemplars manager
        
        Args:
            exemplars_file: Optional path to a JSON file containing exemplars
            embeddings_module: Optional module for computing embeddings
        """
        self.exemplars = []
        self.embeddings_module = embeddings_module
        self.exemplar_embeddings = []
        
        if exemplars_file:
            self.load_exemplars(exemplars_file)
            
        logger.info(f"{Fore.CYAN}Exemplars Manager initialized with {len(self.exemplars)} exemplars{Fore.RESET}")
    
    def load_exemplars(self, file_path: str) -> None:
        """
        Load exemplars from a JSON file
        
        Args:
            file_path: Path to the JSON file containing exemplars
        """
        try:
            if not os.path.exists(file_path):
                logger.warning(f"{Fore.YELLOW}Exemplars file not found: {file_path}{Fore.RESET}")
                return
                
            with open(file_path, 'r') as f:
                self.exemplars = json.load(f)
                
            logger.info(f"{Fore.GREEN}Loaded {len(self.exemplars)} exemplars from {file_path}{Fore.RESET}")
            
            # Compute embeddings if embeddings module is available
            if self.embeddings_module and hasattr(self.embeddings_module, 'get_embedding'):
                self._compute_exemplar_embeddings()
        except Exception as e:
            logger.error(f"{Fore.RED}Error loading exemplars: {str(e)}{Fore.RESET}")
    
    def add_exemplar(self, query: str, sql: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a new exemplar to the collection
        
        Args:
            query: The natural language query
            sql: The SQL query
            metadata: Optional metadata about the exemplar
        """
        exemplar = {
            "query": query,
            "sql": sql,
            "metadata": metadata or {}
        }
        
        self.exemplars.append(exemplar)
        
        # Update embeddings if available
        if self.embeddings_module and hasattr(self.embeddings_module, 'get_embedding'):
            embedding = self.embeddings_module.get_embedding(query)
            self.exemplar_embeddings.append(embedding)
            
        logger.info(f"{Fore.GREEN}Added new exemplar: {query}{Fore.RESET}")
    
    def save_exemplars(self, file_path: str) -> None:
        """
        Save exemplars to a JSON file
        
        Args:
            file_path: Path to save the exemplars
        """
        try:
            with open(file_path, 'w') as f:
                json.dump(self.exemplars, f, indent=2)
                
            logger.info(f"{Fore.GREEN}Saved {len(self.exemplars)} exemplars to {file_path}{Fore.RESET}")
        except Exception as e:
            logger.error(f"{Fore.RED}Error saving exemplars: {str(e)}{Fore.RESET}")
    
    def get_relevant_exemplars(self, query: str, n: int = 3, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Get the most relevant exemplars for a query
        
        Args:
            query: The query to find relevant exemplars for
            n: Maximum number of exemplars to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of relevant exemplars
        """
        if not self.exemplars:
            logger.warning(f"{Fore.YELLOW}No exemplars available{Fore.RESET}")
            return []
            
        # If embeddings are available, use semantic similarity
        if self.embeddings_module and hasattr(self.embeddings_module, 'get_embedding') and self.exemplar_embeddings:
            return self._get_relevant_by_embedding(query, n, threshold)
        
        # Fall back to keyword matching
        return self._get_relevant_by_keywords(query, n)
    
    def _compute_exemplar_embeddings(self) -> None:
        """Compute embeddings for all exemplars"""
        self.exemplar_embeddings = []
        
        for exemplar in self.exemplars:
            embedding = self.embeddings_module.get_embedding(exemplar["query"])
            self.exemplar_embeddings.append(embedding)
            
        logger.info(f"{Fore.GREEN}Computed embeddings for {len(self.exemplars)} exemplars{Fore.RESET}")
    
    def _get_relevant_by_embedding(self, query: str, n: int, threshold: float) -> List[Dict[str, Any]]:
        """Get relevant exemplars using embedding similarity"""
        query_embedding = self.embeddings_module.get_embedding(query)
        
        # Compute similarities
        similarities = []
        for i, exemplar_embedding in enumerate(self.exemplar_embeddings):
            similarity = self._cosine_similarity(query_embedding, exemplar_embedding)
            similarities.append((similarity, i))
        
        # Sort by similarity (descending)
        similarities.sort(reverse=True)
        
        # Filter by threshold and take top n
        relevant_exemplars = []
        for similarity, idx in similarities:
            if similarity >= threshold and len(relevant_exemplars) < n:
                exemplar = self.exemplars[idx].copy()
                exemplar["similarity"] = float(similarity)
                relevant_exemplars.append(exemplar)
        
        logger.info(f"{Fore.GREEN}Found {len(relevant_exemplars)} relevant exemplars using embeddings{Fore.RESET}")
        return relevant_exemplars
    
    def _get_relevant_by_keywords(self, query: str, n: int) -> List[Dict[str, Any]]:
        """Get relevant exemplars using keyword matching"""
        # Simple keyword matching
        query_words = set(query.lower().split())
        
        # Compute similarities based on word overlap
        similarities = []
        for i, exemplar in enumerate(self.exemplars):
            exemplar_words = set(exemplar["query"].lower().split())
            overlap = len(query_words.intersection(exemplar_words))
            similarity = overlap / max(len(query_words), len(exemplar_words))
            similarities.append((similarity, i))
        
        # Sort by similarity (descending)
        similarities.sort(reverse=True)
        
        # Take top n
        relevant_exemplars = []
        for similarity, idx in similarities[:n]:
            exemplar = self.exemplars[idx].copy()
            exemplar["similarity"] = float(similarity)
            relevant_exemplars.append(exemplar)
        
        logger.info(f"{Fore.GREEN}Found {len(relevant_exemplars)} relevant exemplars using keywords{Fore.RESET}")
        return relevant_exemplars
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors"""
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def format_exemplars_for_prompt(self, exemplars: List[Dict[str, Any]]) -> str:
        """
        Format exemplars for inclusion in a prompt
        
        Args:
            exemplars: List of exemplars to format
            
        Returns:
            Formatted exemplars string
        """
        if not exemplars:
            return ""
            
        formatted = "Here are some examples of similar queries and their SQL translations:\n\n"
        
        for i, exemplar in enumerate(exemplars):
            formatted += f"Example {i+1}:\n"
            formatted += f"Query: {exemplar['query']}\n"
            formatted += f"SQL: {exemplar['sql']}\n\n"
            
        return formatted 