"""
LLM Backend Abstraction Layer

This module provides a unified interface for different LLM backends,
including OpenAI-compatible APIs and local models with activation logging.
"""

from .base import BaseLLMBackend, LLMResponse
from .llama_hf import LlamaHFBackend
from .openai_compat import OpenAICompatBackend

__all__ = [
    "BaseLLMBackend",
    "LLMResponse",
    "LlamaHFBackend",
    "OpenAICompatBackend",
]
