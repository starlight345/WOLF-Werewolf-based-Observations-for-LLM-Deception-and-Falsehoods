# WOLF with Internal Activation Logging

This is an enhanced version of WOLF (Werewolf-based Observations for LLM Deception and Falsehoods) that supports:

1. **Local LLaMA execution** via HuggingFace Transformers
2. **Internal activation capture** during generation
3. **Structured activation logging** for downstream analysis

## New Features

### 1. LLaMA Backend with Activation Capture

Instead of using API-based models (GPT-4, Gemini), you can now run local LLaMA models with full access to internal hidden states:

```bash
python run.py --model llama-3.1-8b --log-activations
```

### 2. Activation Logging System

When enabled, the system captures and saves:
- **Hidden states** from selected layers (configurable)
- **Reduction strategies**: last token, mean pooling, or full sequence
- **Memory-efficient storage**: float16, compressed NPZ format
- **Event linking**: Each activation file is linked to its corresponding game event

### 3. Unified Backend Architecture

The new `llm_backends/` module provides a unified interface for:
- **LlamaHFBackend**: Local LLaMA with activation capture
- **OpenAICompatBackend**: API-based models (OpenAI, Gemini)

## Installation

### For Local LLaMA Models

```bash
# Install dependencies
pip install -r requirements.txt

# If you need 4-bit quantization for memory efficiency
pip install bitsandbytes

# Ensure you have CUDA installed for GPU acceleration
```

### For API-based Models (Original)

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="your-key-here"
```

## Usage

### Basic: Run with LLaMA and Activation Logging

```bash
python run.py --model llama-3.1-8b --log-activations
```

### Memory-efficient: Use 4-bit Quantization

```bash
python run.py --model llama-3.1-8b-4bit --log-activations
```

### Original: Use GPT-4

```bash
export OPENAI_API_KEY="your-key-here"
python run.py --model gpt-4o
```

### Available Models

Check `config.py` for all available models. Current options include:

**Local Models (with activation capture):**
- `llama-3.1-8b`: Full precision LLaMA 3.1 8B
- `llama-3.1-8b-4bit`: 4-bit quantized for lower memory

**API Models (no activation capture):**
- `gpt-4o`, `gpt-4o-mini`: OpenAI models
- `gemini-pro`, `gemini-1.5-pro`, `gemini-1.5-flash`: Google models

### Command-line Options

```bash
python run.py \
  --model llama-3.1-8b \
  --log-activations \
  --log-dir ./my_logs
```

Options:
- `--model`: Model to use (default: from `config.py`)
- `--log-activations`: Enable activation logging (only for local models)
- `--no-log-activations`: Disable activation logging
- `--log-dir`: Directory for logs (default: `./logs`)
- `--no-file-logging`: Disable file logging entirely

## Output Structure

After running with activation logging enabled:

```
logs/
  20260130-123456-789012/
    events.ndjson           # Event stream (one JSON per line)
    game_state.json         # Final game state
    final_metrics.json      # Aggregated metrics
    run_meta.json           # Run metadata
    activations/            # Activation files directory
      001_003_Alice_debate.npz
      001_004_Bob_debate.npz
      ...
```

### Activation File Format

Each `.npz` file contains:
- `hidden_states`: numpy array `[n_layers, hidden_dim]` (float16)
- `layer_indices`: list of captured layer indices
- `reduction`: reduction method used
- `dtype`: data type
- `n_layers`: number of layers captured
- `hidden_dim`: dimensionality of hidden states

### Event NDJSON Schema

Each line in `events.ndjson` is a JSON object:

```json
{
  "timestamp": "2026-01-30T12:34:56.789Z",
  "round": 1,
  "step": 3,
  "phase": "debate",
  "event": "debate",
  "actor": "Alice",
  "details": {
    "dialogue": "I think Bob is suspicious...",
    "bids": {...},
    "raw_output": {...}
  },
  "activation_file": "001_003_Alice_debate.npz",
  "activation_metadata": {
    "layer_indices": [0, 4, 8, 12, 16, 20, 24, 28, 31],
    "reduction": "last_token",
    "dtype": "float16",
    "n_layers": 9,
    "hidden_dim": 4096
  }
}
```

## Configuration

Edit `config.py` to customize:

### Activation Logging Config

```python
ACTIVATION_CONFIG = {
    "enabled": True,
    "layers": "all",  # or [0, 8, 16, 24, 31]
    "reduction": "last_token",  # 'last_token', 'mean_pool', 'output_tokens'
    "save_format": "npz",
    "compress": True,
}
```

### Model Config

```python
AVAILABLE_MODELS = {
    "llama-3.1-8b": {
        "name": "meta-llama/Llama-3.1-8B-Instruct",
        "backend": "llama_hf",
        "device": "cuda",
        "dtype": "float16",
        ...
    },
}
```

## Downstream Analysis Pipeline

### 1. Load Activations

```python
import numpy as np
import json

# Load events
events = []
with open("logs/RUN_ID/events.ndjson") as f:
    for line in f:
        events.append(json.loads(line))

# Load activations for a specific event
event = events[0]
if "activation_file" in event:
    act_path = f"logs/RUN_ID/activations/{event['activation_file']}"
    act_data = np.load(act_path)
    hidden_states = act_data["hidden_states"]  # [n_layers, hidden_dim]
    print(f"Shape: {hidden_states.shape}")
```

### 2. Extract Representation Primitives

```python
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA

# Collect all activations for deceptive vs truthful statements
deceptive_acts = []
truthful_acts = []

for event in events:
    if event["event"] == "debate" and "activation_file" in event:
        act = load_activation(event)
        is_deceptive = event["details"]["raw_output"].get("is_deceptive", False)
        
        if is_deceptive:
            deceptive_acts.append(act[-1])  # Last layer
        else:
            truthful_acts.append(act[-1])

# Contrastive direction
mean_deceptive = np.mean(deceptive_acts, axis=0)
mean_truthful = np.mean(truthful_acts, axis=0)
contrast_vector = mean_deceptive - mean_truthful

# PCA on deceptive cluster
pca = PCA(n_components=10)
pca.fit(deceptive_acts)
```

### 3. Causal Intervention (Necessity/Sufficiency)

See the original ACL paper methodology. With activations logged, you can now:

1. **Identify primitive operations** (e.g., "evasion vector", "misdirection rotation")
2. **Test necessity**: Remove/zero-out the primitive → does strategy disappear?
3. **Test sufficiency**: Inject the primitive → does strategy appear?

Example intervention code (pseudocode):

```python
# Necessity test: block the evasion vector
def intervene_necessity(model, input, evasion_vector, layer_idx):
    def hook_fn(module, input, output):
        # Project out the evasion direction
        hidden = output[0]  # [batch, seq, hidden]
        hidden_proj = hidden - (hidden @ evasion_vector[:, None]) @ evasion_vector[None, :]
        return (hidden_proj,) + output[1:]
    
    handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)
    output = model.generate(input)
    handle.remove()
    return output

# Sufficiency test: add the evasion vector
def intervene_sufficiency(model, input, evasion_vector, layer_idx, alpha=1.0):
    def hook_fn(module, input, output):
        hidden = output[0]
        hidden_steered = hidden + alpha * evasion_vector
        return (hidden_steered,) + output[1:]
    
    handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)
    output = model.generate(input)
    handle.remove()
    return output
```

## Research Questions Enabled

With internal activation logging, you can now investigate:

1. **What are the geometric primitives of deception strategies?**
   - Energy, rank, rotation patterns in representation space
   
2. **Are these primitives causally necessary?**
   - Ablation → does deception rate drop?
   
3. **Are they sufficient?**
   - Injection → does deception rate increase?
   
4. **How do they evolve across layers?**
   - Layer-wise fingerprinting with SHAP/attention analysis

5. **Can we build a closed-loop "definition → intervention → verification" pipeline?**

## Troubleshooting

### CUDA Out of Memory

Use 4-bit quantization:
```bash
python run.py --model llama-3.1-8b-4bit
```

Or reduce batch size / max length in `config.py`.

### Model Download Issues

Ensure you have:
1. Hugging Face account and accepted model terms (for LLaMA)
2. HF token set: `huggingface-cli login`

### Activation Files Too Large

Adjust in `config.py`:
```python
ACTIVATION_CONFIG = {
    "layers": [0, 8, 16, 24, 31],  # Only 5 layers
    "reduction": "last_token",     # vs full sequence
}
```

## Citation

If you use this enhanced WOLF with activation logging, please cite:

- Original WOLF paper: [link]
- Your ACL paper on IMT strategies and geometric patterns
- This repository

## License

Apache License 2.0 (same as original WOLF)
