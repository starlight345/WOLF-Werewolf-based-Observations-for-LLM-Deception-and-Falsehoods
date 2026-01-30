"""
LLaMA backend using TransformerLens with activation capture.

TransformerLens is the standard library for mechanistic interpretability research,
providing clean hooks and standardized activation access.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional
from transformer_lens import HookedTransformer
import time
from .base import BaseLLMBackend, LLMResponse


class LlamaHFBackend(BaseLLMBackend):
    """
    TransformerLens-based backend for LLaMA models with activation logging.
    
    Uses HookedTransformer for standardized activation capture, following
    best practices in mechanistic interpretability research.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.model_name = config.get("model_name_or_path", "meta-llama/Llama-3.1-8B-Instruct")
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        
        # Parse dtype
        dtype_str = config.get("dtype", "float16")
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        self.dtype = dtype_map.get(dtype_str, torch.float16)
        
        # Activation capture config
        self.activation_layers = config.get("activation_layers", "all")
        self.activation_reduce = config.get("activation_reduce", "last_token")
        
        # Load model using TransformerLens
        print(f"Loading LLaMA model with TransformerLens: {self.model_name}")
        print("This may take a few minutes on first load...")
        
        try:
            # TransformerLens HookedTransformer.from_pretrained
            self.model = HookedTransformer.from_pretrained(
                self.model_name,
                device=self.device,
                dtype=self.dtype,
                fold_ln=False,  # Keep LayerNorm separate for better interpretability
                center_writing_weights=False,  # Keep original weights
                center_unembed=False,
            )
            
            self.tokenizer = self.model.tokenizer
            
            # Set pad token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print(f"✓ Model loaded successfully on {self.device}")
            print(f"  Layers: {self.model.cfg.n_layers}")
            print(f"  Hidden dim: {self.model.cfg.d_model}")
            
        except Exception as e:
            print(f"✗ Failed to load with TransformerLens: {e}")
            print("Falling back to standard loading...")
            raise
    
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
    
    def _extract_activations_from_cache(
        self,
        cache: Dict[str, torch.Tensor],
        input_length: int,
        output_length: int
    ) -> Optional[Dict[str, Any]]:
        """
        Extract activations from TransformerLens cache.
        
        TransformerLens stores activations with standard naming:
        - blocks.{layer}.hook_resid_pre: Residual stream before layer
        - blocks.{layer}.hook_resid_post: Residual stream after layer (most common)
        - blocks.{layer}.hook_mlp_out: MLP output
        - blocks.{layer}.hook_attn_out: Attention output
        
        For IMT strategy research, we typically use hook_resid_post (final residual stream).
        """
        if cache is None or len(cache) == 0:
            return None
        
        n_layers = self.model.cfg.n_layers
        
        # Determine which layers to capture
        if self.activation_layers == "all":
            layer_indices = list(range(n_layers))
        elif isinstance(self.activation_layers, list):
            layer_indices = [i for i in self.activation_layers if 0 <= i < n_layers]
        else:
            step = max(1, n_layers // 8)
            layer_indices = list(range(0, n_layers, step))
        
        activations_list = []
        
        for layer_idx in layer_indices:
            # Use residual stream post-layer (standard in interpretability)
            hook_name = f"blocks.{layer_idx}.hook_resid_post"
            
            if hook_name not in cache:
                print(f"Warning: {hook_name} not found in cache")
                continue
            
            layer_hidden = cache[hook_name]  # [batch, seq_len, d_model]
            
            # Remove batch dimension (we assume batch_size=1)
            if layer_hidden.dim() == 3:
                layer_hidden = layer_hidden[0]  # [seq_len, d_model]
            
            # Apply reduction strategy
            if self.activation_reduce == "last_token":
                # Last token position (standard for sequence classification)
                reduced = layer_hidden[-1, :].cpu().numpy()
            
            elif self.activation_reduce == "mean_pool":
                # Average across all tokens
                reduced = layer_hidden.mean(dim=0).cpu().numpy()
            
            elif self.activation_reduce == "output_tokens":
                # Average only over generated tokens (exclude prompt)
                if output_length > 0:
                    output_hidden = layer_hidden[-output_length:, :]
                    reduced = output_hidden.mean(dim=0).cpu().numpy()
                else:
                    reduced = layer_hidden[-1, :].cpu().numpy()
            
            elif self.activation_reduce == "none" or self.activation_reduce == "full":
                # Keep full sequence (memory intensive!)
                reduced = layer_hidden.cpu().numpy()
            
            else:
                # Default to last token
                reduced = layer_hidden[-1, :].cpu().numpy()
            
            activations_list.append(reduced)
        
        if len(activations_list) == 0:
            return None
        
        # Stack and convert to float16 for storage efficiency
        activations_array = np.stack(activations_list, axis=0).astype(np.float16)
        
        return {
            "hidden_states": activations_array,
            "layer_indices": layer_indices,
            "reduction": self.activation_reduce,
            "dtype": "float16",
            "n_layers": len(layer_indices),
            "hidden_dim": activations_array.shape[-1],
            "hook_type": "resid_post",  # TransformerLens standard
        }
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 200,
        timeout: int = 15,
        return_activations: bool = False,
        **kwargs
    ) -> LLMResponse:
        """
        Generate text with optional activation capture using TransformerLens.
        
        When return_activations=True, uses run_with_cache() for the prompt forward pass
        to capture activations, then generates normally.
        
        Note: This captures activations from the prompt + first token generation,
        not the full generation sequence (which would be memory-intensive).
        """
        start_time = time.time()
        
        formatted_prompt = self._prepare_prompt(prompt)
        
        # Tokenize
        tokens = self.tokenizer.encode(formatted_prompt, return_tensors="pt")
        tokens = tokens.to(self.device)
        input_length = tokens.shape[1]
        
        activations_dict = None
        
        if return_activations:
            # Use TransformerLens run_with_cache for activation capture
            # This runs forward pass and caches all activations
            with torch.no_grad():
                logits, cache = self.model.run_with_cache(
                    tokens,
                    remove_batch_dim=False
                )
            
            # Generate continuation from the cached logits
            # For simplicity, we do a separate generation call
            # (In practice, you could use the cached KV for efficiency)
            with torch.no_grad():
                generated_ids = self.model.generate(
                    tokens,
                    max_new_tokens=max_tokens,
                    do_sample=kwargs.get("do_sample", True),
                    temperature=kwargs.get("temperature", 0.7),
                    top_p=kwargs.get("top_p", 0.9),
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            output_length = generated_ids.shape[1] - input_length
            
            # Extract activations from cache
            activations_dict = self._extract_activations_from_cache(
                cache,
                input_length,
                output_length
            )
            
        else:
            # Standard generation without activation capture
            with torch.no_grad():
                generated_ids = self.model.generate(
                    tokens,
                    max_new_tokens=max_tokens,
                    do_sample=kwargs.get("do_sample", True),
                    temperature=kwargs.get("temperature", 0.7),
                    top_p=kwargs.get("top_p", 0.9),
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
        
        # Decode generated text (excluding input prompt)
        generated_text = self.tokenizer.decode(
            generated_ids[0, input_length:],
            skip_special_tokens=True
        ).strip()
        
        elapsed = time.time() - start_time
        
        metadata = {
            "input_length": input_length,
            "output_length": generated_ids.shape[1] - input_length,
            "total_tokens": generated_ids.shape[1],
            "elapsed_seconds": elapsed,
            "model": self.model_name,
            "backend": "transformer_lens",
        }
        
        return LLMResponse(
            text=generated_text,
            activations=activations_dict,
            metadata=metadata
        )
    
    def get_hook_names(self) -> List[str]:
        """
        Get all available hook names in the model.
        
        Useful for debugging and exploring what activations are available.
        """
        if hasattr(self.model, 'hook_dict'):
            return list(self.model.hook_dict.keys())
        return []
    
    def cleanup(self):
        """Free GPU memory."""
        if hasattr(self, 'model'):
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("LLaMA model cleanup complete")
