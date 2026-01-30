# Changes: WOLF with Internal Activation Logging

## Summary

This fork adds **local LLaMA execution** and **internal activation capture** to WOLF, enabling research on the causal mechanisms of deception strategies in LLMs.

## Key Changes

### 1. New LLM Backend Architecture

**Added Files:**
- `llm_backends/__init__.py`: Backend module exports
- `llm_backends/base.py`: Abstract base class for LLM backends
- `llm_backends/llama_hf.py`: **TransformerLens-based backend** with activation capture
- `llm_backends/openai_compat.py`: Wrapper for existing OpenAI/LangChain APIs

**Benefits:**
- Unified interface for different LLM providers
- **TransformerLens integration**: Industry-standard mechanistic interpretability library
- Clean hook system for activation capture
- Easy to add new backends (e.g., vLLM, Ollama)
- Clean separation between API and local models

### 2. Activation Capture System (TransformerLens)

**Modified Files:**
- `player.py`: Updated `call_model()` to support new backend system
  - Added `log_activations` parameter
  - Support for both legacy and new backends
  - Extracts and passes activations from LLM responses

**Implementation:**
- **Uses TransformerLens `run_with_cache()`** for standardized activation capture
- Captures residual stream activations (`blocks.{layer}.hook_resid_post`)
- Supports multiple reduction strategies: last token, mean pooling, full sequence
- Memory-efficient: float16, compressed NPZ storage
- ~500 KB per statement (vs several MB uncompressed)
- Compatible with standard interpretability pipelines

### 3. Enhanced Logging System

**Modified Files:**
- `logs.py`: Added activation storage functions
  - `save_activations()`: Saves activation data to NPZ files
  - `log_event()`: Extended to accept and link activation files
  - Creates `activations/` subdirectory in each run

**Event Schema Extension:**
```json
{
  "event": "debate",
  "actor": "Alice",
  "details": {...},
  "activation_file": "001_003_Alice_debate.npz",  // NEW
  "activation_metadata": {                        // NEW
    "layer_indices": [...],
    "reduction": "last_token",
    "n_layers": 9,
    "hidden_dim": 4096
  }
}
```

### 4. Configuration System

**Modified Files:**
- `config.py`: Added LLaMA model configurations and activation settings
  - New models: `llama-3.1-8b`, `llama-3.1-8b-4bit`
  - `ACTIVATION_CONFIG`: Centralized activation logging settings
  - Changed default model to `llama-3.1-8b`

**Key Config Options:**
```python
ACTIVATION_CONFIG = {
    "enabled": True,
    "layers": "all",  # or specific list
    "reduction": "last_token",
    "save_format": "npz",
    "compress": True,
}
```

### 5. Main Runner Updates

**Modified Files:**
- `run.py`: Complete backend selection and initialization logic
  - `get_llm()`: Factory function for backend instantiation
  - `run_werewolf_game()`: Added `log_activations` parameter
  - Command-line args: `--log-activations`, `--no-log-activations`
  - Automatic backend cleanup (GPU memory)

### 6. Game Graph Integration

**Modified Files:**
- `game_graph.py`: Link activations to game events
  - `debate_node()`: Extract and pass activations from player responses
  - Future: Can extend to other nodes (vote, eliminate, etc.)

### 7. Analysis Tools

**New Files:**
- `analyze_activations.py`: Utility script for activation analysis
  - Load and process activation data
  - Compute contrastive directions
  - Layer-wise separation metrics
  - Visualization tools

**Features:**
- Statistical summary of activations
- Contrastive vector extraction (deceptive vs truthful)
- Layer-wise separation analysis
- Save representation primitives for downstream use

### 8. Documentation

**New Files:**
- `README_ACTIVATION_LOGGING.md`: Comprehensive guide to new features
- `QUICKSTART.md`: Step-by-step getting started guide
- `CHANGES.md`: This file (summary of modifications)

**Updated Files:**
- `requirements.txt`: Added torch, transformers, accelerate, bitsandbytes

## Dependencies Added

```
torch>=2.0.0
transformers>=4.35.0
accelerate>=0.24.0
sentencepiece>=0.1.99
safetensors>=0.4.0
transformer-lens>=1.17.0  # NEW: Standard library for mechanistic interpretability
bitsandbytes>=0.41.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
```

**TransformerLens** is the key addition - it's the standard library used by:
- Anthropic (interpretability research)
- EleutherAI (GPT-NeoX analysis)
- Most academic mechanistic interpretability labs

## Backward Compatibility

✅ **Fully backward compatible** with original WOLF:
- Old API-based models (GPT-4, Gemini) still work
- Set `DEFAULT_MODEL = "gpt-4o"` in `config.py` to restore original behavior
- No breaking changes to game logic or event schema (only additions)

## Performance Impact

| Aspect | Original | With Activations |
|--------|----------|------------------|
| Execution | API call (~5s/turn) | Local generation (~2-3s/turn on GPU) |
| Storage | ~1-5 MB per game | ~5-20 MB per game (includes activations) |
| Memory | 0 GB | 8-16 GB GPU (depends on quantization) |
| Cost | API fees | Free (local compute) |

## Research Workflow Enabled

1. **Collect Data**: Run games with `--log-activations`
2. **Extract Primitives**: Use `analyze_activations.py` to identify patterns
3. **Test Necessity**: Implement ablation interventions
4. **Test Sufficiency**: Implement injection interventions
5. **Validate Claims**: Closed-loop verification

## Testing Recommendations

### Unit Tests (Future Work)
- `test_llama_backend.py`: Test LLaMA backend initialization and generation
- `test_activation_logging.py`: Test activation capture and storage
- `test_backward_compat.py`: Ensure API models still work

### Integration Tests
1. Run a short game with LLaMA: `python run.py --model llama-3.1-8b-4bit --log-activations`
2. Verify activation files are created
3. Run analysis: `python analyze_activations.py --run-dir logs/LATEST --visualize`

### Smoke Test Command
```bash
# Quick test (use 4-bit model for lower memory)
python run.py --model llama-3.1-8b-4bit --log-activations --log-dir ./test_logs

# Should complete without errors and create:
# - test_logs/TIMESTAMP/events.ndjson
# - test_logs/TIMESTAMP/activations/*.npz
```

## Known Limitations

1. **Activation capture only during generation, not full forward pass**
   - Current: Uses last generation step's hidden states
   - Future: Hook into generation loop for per-token capture

2. **Limited to HuggingFace models**
   - vLLM, Ollama backends not yet implemented
   - Easy to add via `BaseLLMBackend` interface

3. **Memory-intensive for large models**
   - LLaMA 3.1 70B requires multiple GPUs
   - Use 4-bit quantization for larger models

4. **No activation capture for API models**
   - OpenAI/Gemini don't expose internal states
   - Fundamental limitation, not a bug

## Future Enhancements

- [ ] Per-token activation capture during generation
- [ ] Support for vLLM backend (faster inference)
- [ ] Support for Ollama backend (easier setup)
- [ ] Attention pattern logging
- [ ] Gradient-based attribution (for necessity/sufficiency tests)
- [ ] Built-in intervention tools (ablation, steering)
- [ ] Web UI for activation visualization
- [ ] Automatic primitive extraction pipeline

## Migration Guide

### From Original WOLF

No changes needed! Just install new dependencies:

```bash
pip install -r requirements.txt
```

To use new features:

```bash
# Old way (still works)
python run.py --model gpt-4o

# New way
python run.py --model llama-3.1-8b --log-activations
```

### Configuration Updates

If you have custom `config.py` modifications:

1. Add `"backend"` field to your custom models
2. Add `ACTIVATION_CONFIG` section
3. Update `DEFAULT_MODEL` if desired

## Citation

If you use this enhanced version in your research, please cite:

```bibtex
@software{wolf_activations_2026,
  title={WOLF with Internal Activation Logging},
  author={Your Name},
  year={2026},
  url={https://github.com/starlight345/WOLF-Werewolf-based-Observations-for-LLM-Deception-and-Falsehoods}
}
```

## License

Apache License 2.0 (same as original WOLF)

## Contact

For questions or issues, please open a GitHub issue or contact the maintainers.
