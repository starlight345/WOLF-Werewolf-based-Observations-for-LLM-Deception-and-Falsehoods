"""
OpenAI-compatible backend wrapper.

This maintains backward compatibility with existing OpenAI/LangChain code.
"""

from typing import Dict, Any
from langchain_openai import ChatOpenAI
from .base import BaseLLMBackend, LLMResponse


class OpenAICompatBackend(BaseLLMBackend):
    """
    Wrapper for OpenAI-compatible APIs (OpenAI, Gemini via LangChain).
    
    Note: This backend does NOT support activation capture since we don't
    have access to internal model states through API calls.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.model_name = config.get("model_name", "gpt-4o")
        self.temperature = config.get("temperature", 0.7)
        self.api_key = config.get("api_key", None)
        
        # Initialize LangChain ChatOpenAI
        init_kwargs = {
            "model": self.model_name,
            "temperature": self.temperature,
        }
        
        if self.api_key:
            init_kwargs["api_key"] = self.api_key
        
        self.llm = ChatOpenAI(**init_kwargs)
    
    def get_model_name(self) -> str:
        return self.model_name
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 200,
        timeout: int = 15,
        return_activations: bool = False,
        **kwargs
    ) -> LLMResponse:
        """
        Generate text using OpenAI API.
        
        Note: return_activations is ignored as API models don't expose internals.
        """
        if return_activations:
            print("Warning: OpenAI-compatible backends do not support activation capture")
        
        # Call the model
        response = self.llm.invoke(
            prompt,
            max_tokens=max_tokens,
            timeout=timeout
        )
        
        text = response.content.strip()
        
        metadata = {
            "model": self.model_name,
            "backend": "openai_compat",
        }
        
        return LLMResponse(
            text=text,
            activations=None,  # Not available for API models
            metadata=metadata
        )
