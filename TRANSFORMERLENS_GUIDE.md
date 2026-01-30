# TransformerLens Guide for WOLF

## Why TransformerLens?

TransformerLens is the **industry standard** for mechanistic interpretability research. It's used by:
- **Anthropic** (Claude interpretability team)
- **EleutherAI** (GPT-NeoX analysis)
- **Redwood Research** (alignment research)
- Most academic interpretability labs

### Key Advantages

1. **Standardized Hook System**: Clean, consistent naming across all models
2. **Research Reproducibility**: Same tools everyone else uses
3. **Rich Ecosystem**: Compatible with libraries like CircuitsVis, PySvelte
4. **Active Development**: Maintained by Neel Nanda and the interpretability community

## Hook Naming Convention

TransformerLens uses a standardized naming scheme:

```python
# Residual stream activations (most common)
blocks.{layer}.hook_resid_pre   # Before the layer
blocks.{layer}.hook_resid_post  # After the layer (default in WOLF)

# Attention activations
blocks.{layer}.attn.hook_q      # Query projections
blocks.{layer}.attn.hook_k      # Key projections
blocks.{layer}.attn.hook_v      # Value projections
blocks.{layer}.attn.hook_attn_scores  # Attention scores
blocks.{layer}.attn.hook_pattern      # Attention patterns (softmax)

# MLP activations
blocks.{layer}.mlp.hook_pre     # Before MLP activation
blocks.{layer}.mlp.hook_post    # After MLP activation

# Final outputs
blocks.{layer}.hook_mlp_out     # MLP output
blocks.{layer}.hook_attn_out    # Attention output
```

## WOLF Integration

WOLF uses `hook_resid_post` by default - the residual stream after each layer. This is the standard choice because:

1. **Residual stream = information highway**: All information flows through it
2. **Easier interpretation**: Direct representation of model's "thoughts"
3. **Standard in research**: Most papers use residual stream activations

## Advanced Usage

### 1. Accessing Different Activation Types

Edit `config.py` to customize which hooks to capture:

```python
ACTIVATION_CONFIG = {
    "enabled": True,
    "layers": "all",
    "reduction": "last_token",
    "hook_type": "resid_post",  # Can change to: attn_out, mlp_out, etc.
}
```

Then modify `llama_hf.py` line 141 to use different hooks:

```python
# Current (default)
hook_name = f"blocks.{layer_idx}.hook_resid_post"

# Alternatives
hook_name = f"blocks.{layer_idx}.hook_attn_out"  # Attention output only
hook_name = f"blocks.{layer_idx}.hook_mlp_out"   # MLP output only
```

### 2. Capturing Multiple Hook Types

For comprehensive analysis, capture multiple activation types:

```python
# In llama_hf.py, modify _extract_activations_from_cache()
for layer_idx in layer_indices:
    # Residual stream
    resid_post = cache[f"blocks.{layer_idx}.hook_resid_post"]
    
    # Attention output
    attn_out = cache[f"blocks.{layer_idx}.hook_attn_out"]
    
    # MLP output
    mlp_out = cache[f"blocks.{layer_idx}.hook_mlp_out"]
    
    # Store all three
    activations_list.append({
        "resid_post": resid_post[-1].cpu().numpy(),
        "attn_out": attn_out[-1].cpu().numpy(),
        "mlp_out": mlp_out[-1].cpu().numpy(),
    })
```

### 3. Attention Pattern Analysis

To analyze attention patterns (which tokens attend to which):

```python
# In llama_hf.py
for layer_idx in layer_indices:
    attn_pattern = cache[f"blocks.{layer_idx}.attn.hook_pattern"]
    # Shape: [batch, n_heads, seq_len, seq_len]
    # attn_pattern[0, head_idx, query_pos, key_pos] = attention weight
```

### 4. Per-Head Analysis

Decompose by attention head:

```python
for layer_idx in layer_indices:
    for head_idx in range(model.cfg.n_heads):
        # Get per-head attention output
        hook_name = f"blocks.{layer_idx}.attn.hook_result"
        per_head_out = cache[hook_name]  # [batch, seq, n_heads, d_head]
        head_output = per_head_out[0, -1, head_idx, :]  # Last token, specific head
```

## Example: Custom Intervention

Using TransformerLens hooks for causal interventions:

```python
from transformer_lens import HookedTransformer

# Load model
model = HookedTransformer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# Define intervention
def ablate_layer(activations, hook):
    """Zero out a specific layer's residual stream."""
    activations[:, :, :] = 0.0
    return activations

# Run with intervention
with model.hooks(fwd_hooks=[(f"blocks.10.hook_resid_post", ablate_layer)]):
    output = model.generate(tokens, max_new_tokens=50)

# Compare with baseline to test necessity
```

## Example: Contrastive Steering

Inject a direction into the residual stream:

```python
def steer_with_vector(activations, hook, vector, alpha=1.0):
    """Add a steering vector to residual stream."""
    activations[:, -1, :] += alpha * vector  # Add to last token
    return activations

# Create steering hook
steering_vector = torch.tensor(contrast_vector, device=model.cfg.device)
hook_fn = lambda act, hook: steer_with_vector(act, hook, steering_vector, alpha=2.0)

# Generate with steering
with model.hooks(fwd_hooks=[(f"blocks.20.hook_resid_post", hook_fn)]):
    steered_output = model.generate(tokens, max_new_tokens=50)
```

## Common Patterns in IMT Research

### 1. Residual Stream Analysis (Default)
```python
hook_name = "blocks.{layer}.hook_resid_post"
```
**Use for:** Overall information flow, strategy primitives

### 2. Attention-Only Analysis
```python
hook_name = "blocks.{layer}.hook_attn_out"
```
**Use for:** Finding which context matters, copying mechanisms

### 3. MLP-Only Analysis
```python
hook_name = "blocks.{layer}.hook_mlp_out"
```
**Use for:** Factual recall, feature detection

### 4. Decomposed Analysis
```python
# Compute contribution of each component
resid_pre = cache[f"blocks.{layer}.hook_resid_pre"]
attn_out = cache[f"blocks.{layer}.hook_attn_out"]
mlp_out = cache[f"blocks.{layer}.hook_mlp_out"]
resid_post = cache[f"blocks.{layer}.hook_resid_post"]

# Verify: resid_post ≈ resid_pre + attn_out + mlp_out
```

## Debugging

### List All Available Hooks

```python
from llm_backends import LlamaHFBackend

backend = LlamaHFBackend(config)
print(backend.get_hook_names())
```

### Inspect Cache Contents

```python
# After run_with_cache
for hook_name, activation in cache.items():
    print(f"{hook_name}: {activation.shape}")
```

### Verify Activation Capture

```python
import numpy as np

# Load saved activation
data = np.load("logs/.../activations/001_003_Alice_debate.npz")
print("Keys:", list(data.keys()))
print("Shape:", data["hidden_states"].shape)
print("Hook type:", data["hook_type"])
```

## Further Reading

- [TransformerLens Documentation](https://transformerlensorg.github.io/TransformerLens/)
- [Neel Nanda's Tutorials](https://www.neelnanda.io/mechanistic-interpretability)
- [Anthropic's Interpretability Research](https://www.anthropic.com/research/interpretability)
- [Alignment Forum - Mechanistic Interpretability](https://www.alignmentforum.org/tag/mechanistic-interpretability)

## Citation

If you use TransformerLens in your research:

```bibtex
@misc{nanda2022transformerlens,
    title = {TransformerLens},
    author = {Nanda, Neel and others},
    year = {2022},
    howpublished = {\url{https://github.com/neelnanda-io/TransformerLens}},
}
```
