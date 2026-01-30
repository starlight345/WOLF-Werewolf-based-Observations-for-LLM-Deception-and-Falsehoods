"""
Base interface for LLM backends.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import numpy as np


@dataclass
class LLMResponse:
    """
    Unified response format from LLM backends.
    
    Attributes:
        text: Generated text response
        activations: Optional dictionary containing internal activations
            - 'hidden_states': numpy array of shape [n_layers, hidden_dim] or [n_layers, seq_len, hidden_dim]
            - 'layer_indices': list of layer indices that were captured
            - 'reduction': type of reduction applied ('last_token', 'mean_pool', 'none')
            - 'dtype': data type of saved activations
        metadata: Additional metadata (tokens, timing, etc.)
    """
    text: str
    activations: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseLLMBackend(ABC):
    """
    Abstract base class for LLM backends.
    
    All backends must implement this interface to be compatible with
    the WOLF game system.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the backend with configuration.
        
        Args:
            config: Dictionary containing backend-specific configuration
        """
        self.config = config
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 200,
        timeout: int = 15,
        return_activations: bool = False,
        **kwargs
    ) -> LLMResponse:
        """
        Generate text from the model.
        
        Args:
            prompt: Input prompt string or messages
            max_tokens: Maximum number of tokens to generate
            timeout: Timeout in seconds
            return_activations: Whether to capture and return internal activations
            **kwargs: Additional backend-specific arguments
            
        Returns:
            LLMResponse object containing text, activations (if requested), and metadata
        """
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Return the name/identifier of the underlying model."""
        pass
    
    def cleanup(self):
        """Optional cleanup method for resources (GPU memory, etc.)."""
        pass
