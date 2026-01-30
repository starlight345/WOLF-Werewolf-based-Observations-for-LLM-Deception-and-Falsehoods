"""
LLaMA backend using HuggingFace Transformers with activation capture.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
from .base import BaseLLMBackend, LLMResponse


class LlamaHFBackend(BaseLLMBackend):
    """HuggingFace Transformers backend for LLaMA models with activation logging."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.model_name = config.get("model_name_or_path", "meta-llama/Llama-3.1-8B-Instruct")
        self.device = config.get("device", "auto")
        
        # Parse dtype
        dtype_str = config.get("dtype", "float16")
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        self.dtype = dtype_map.get(dtype_str, torch.float16)
        
        # Quantization options
        self.load_in_8bit = config.get("load_in_8bit", False)
        self.load_in_4bit = config.get("load_in_4bit", False)
        
        # Activation capture config
        self.activation_layers = config.get("activation_layers", "all")
        self.activation_reduce = config.get("activation_reduce", "last_token")
        
        # Load model and tokenizer
        print(f"Loading LLaMA model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": self.dtype,
        }
        
        if self.load_in_8bit:
            model_kwargs["load_in_8bit"] = True
        elif self.load_in_4bit:
            model_kwargs["load_in_4bit"] = True
        
        if self.device != "auto" and not (self.load_in_8bit or self.load_in_4bit):
            model_kwargs["device_map"] = None
        else:
            model_kwargs["device_map"] = "auto"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        
        if self.device != "auto" and not (self.load_in_8bit or self.load_in_4bit):
            self.model = self.model.to(self.device)
        
        self.model.eval()
        print(f"Model loaded successfully on {self.device}")
    
    def get_model_name(self) -> str:
        return self.model_name
    
    def _prepare_prompt(self, prompt: str) -> str:
        """Prepare prompt for LLaMA instruction-tuned models."""
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            return formatted_prompt
        return prompt
    
    def _extract_activations(self, hidden_states: tuple, input_length: int, output_length: int) -> Optional[Dict[str, Any]]:
        """Extract and reduce hidden states according to config."""
        if hidden_states is None or len(hidden_states) == 0:
            return None
        
        n_layers = len(hidden_states)
        if self.activation_layers == "all":
            layer_indices = list(range(n_layers))
        elif isinstance(self.activation_layers, list):
            layer_indices = [i for i in self.activation_layers if 0 <= i < n_layers]
        else:
            step = max(1, n_layers // 8)
            layer_indices = list(range(0, n_layers, step))
        
        activations_list = []
        for layer_idx in layer_indices:
            layer_hidden = hidden_states[layer_idx]
            if layer_hidden.dim() == 3:
                layer_hidden = layer_hidden[0]
            
            if self.activation_reduce == "last_token":
                reduced = layer_hidden[-1, :].cpu().numpy()
            elif self.activation_reduce == "mean_pool":
                reduced = layer_hidden.mean(dim=0).cpu().numpy()
            elif self.activation_reduce == "output_tokens":
                if output_length > 0:
                    output_hidden = layer_hidden[-output_length:, :]
                    reduced = output_hidden.mean(dim=0).cpu().numpy()
                else:
                    reduced = layer_hidden[-1, :].cpu().numpy()
            else:
                reduced = layer_hidden.cpu().numpy()
            
            activations_list.append(reduced)
        
        activations_array = np.stack(activations_list, axis=0).astype(np.float16)
        
        return {
            "hidden_states": activations_array,
            "layer_indices": layer_indices,
            "reduction": self.activation_reduce,
            "dtype": "float16",
            "n_layers": len(layer_indices),
            "hidden_dim": activations_array.shape[-1],
        }
    
    def generate(self, prompt: str, max_tokens: int = 200, timeout: int = 15, return_activations: bool = False, **kwargs) -> LLMResponse:
        """Generate text with optional activation capture."""
        start_time = time.time()
        
        formatted_prompt = self._prepare_prompt(prompt)
        
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.get("max_length", 2048)
        )
        
        if self.device != "auto":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        input_length = inputs["input_ids"].shape[1]
        
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": kwargs.get("do_sample", True),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        if return_activations:
            gen_kwargs["output_hidden_states"] = True
            gen_kwargs["return_dict_in_generate"] = True
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        
        if return_activations:
            generated_ids = outputs.sequences[0]
            hidden_states = outputs.hidden_states if hasattr(outputs, 'hidden_states') else None
        else:
            generated_ids = outputs[0]
            hidden_states = None
        
        generated_text = self.tokenizer.decode(
            generated_ids[input_length:],
            skip_special_tokens=True
        ).strip()
        
        activations_dict = None
        if return_activations and hidden_states is not None and len(hidden_states) > 0:
            output_length = len(generated_ids) - input_length
            last_step_hidden = hidden_states[-1]
            activations_dict = self._extract_activations(last_step_hidden, input_length, output_length)
        
        elapsed = time.time() - start_time
        
        metadata = {
            "input_length": input_length,
            "output_length": len(generated_ids) - input_length,
            "total_tokens": len(generated_ids),
            "elapsed_seconds": elapsed,
            "model": self.model_name,
        }
        
        return LLMResponse(text=generated_text, activations=activations_dict, metadata=metadata)
    
    def cleanup(self):
        """Free GPU memory."""
        if hasattr(self, 'model'):
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("LLaMA model cleanup complete")
