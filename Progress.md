# Phase 1: Hidden State Extraction and Deception Signal Analysis

## Setup

- **Model**: Meta-Llama-3-8B-Instruct (32 transformer layers, 4096 hidden dim)
- **Hardware**: 2x NVIDIA RTX A6000 (48GB each)
- **Approach**: Controlled single-turn prompts with ground-truth deception labels. Extract last-token hidden states at every layer before generation begins (the model's "intent vector").

## Dataset

48 prompts across 8 conditions:

| Condition | Count | Description |
|-----------|-------|-------------|
| Truthful (game) | 16 | Villager/Seer/Doctor answering honestly in Werewolf context |
| Fabrication | 5 | Werewolf inventing false information (fake role, fake evidence) |
| Omission | 4 | Werewolf deliberately withholding key information |
| Misdirection | 4 | Werewolf redirecting suspicion toward innocent players |
| Distortion | 4 | Werewolf exaggerating or twisting real events |
| Creative (control) | 5 | Imaginative writing with no deceptive intent |
| Factual recall (control) | 5 | Retrieving known information accurately |
| Suppression (control) | 5 | Honestly withholding information without lying |

Control conditions test whether deception types map onto interpretable cognitive primitives (e.g., does fabrication look like creativity internally?).

## Key Findings

### 1. Task context dominates over deception signal

The model's primary organizational axis at mid-to-late layers is **task type** (game vs recall vs creative vs suppression), not truthful vs deceptive. All game conditions -- both honest and deceptive -- cluster tightly together in PCA/UMAP space, while control conditions form separate isolated clusters. The deception signal is a **secondary perturbation within the game representation**, not a primary axis of variation.

Evidence: UMAP at layer 16 shows three macro-clusters (game, recall, creative) with suppression forming a fourth. All four deception types are interleaved with truthful game responses within the game cluster.

### 2. Fabrication does NOT recruit creativity circuits

**Hypothesis**: Fabrication (inventing false information) might activate the same internal circuits as creative writing.

**Result**: Not supported. Cosine similarity between fabrication and creative control centroids is only 0.561 at layer 16 -- far lower than fabrication's similarity to truthful game responses (0.898). PCA shows complete spatial separation between fabrication and creative writing. The model treats "lying in a game" as fundamentally different from "writing fiction."

**Implication**: Fabrication is not a borrowing from creative circuits. The model's representation of "I am generating false information within a strategic context" is distinct from "I am being imaginatively creative."

### 3. Omission partially shares circuits with honest suppression

**Hypothesis**: Omission (deliberately withholding information) might activate similar circuits to honest suppression (withholding information without lying).

**Result**: Partially supported. Omission-Suppression cosine similarity is 0.761 -- the highest cross-domain pair in the entire matrix. However, this is still much lower than within-game similarities (0.910 for omission-truthful). Omission is more "game" than "suppression," but among all deception types, it's the closest to honest restraint.

### 4. Deception types have distinct internal signatures

Different deception strategies produce measurably different activation patterns:

| Deception Type | Cosine sim to Truthful (Layer 16) | Interpretation |
|----------------|-----------------------------------|----------------|
| Omission | 0.910 | Closest to truthful -- hardest to detect internally |
| Fabrication | 0.898 | Moderately distinct |
| Misdirection | 0.871 | More distinct -- active redirection |
| Distortion | 0.846 | Most distinct -- requires active transformation of information |

This ordering is stable across layers and makes intuitive sense: omission is passive (just don't say something), while distortion requires actively transforming and exaggerating information.

Top discriminative dimensions show **type-specific patterns**:
- Dimension d4055: Fabrication strongly negative (-0.6), omission positive (+0.23) -- this single dimension partially separates fabrication from omission.
- Dimension d1189: Fabrication-specific positive activation (+0.33).
- Dimension d568: Misdirection peaks here (+0.27) -- potentially related to "redirect attention" computations.
- Several dimensions (d2932, d219) show all deception types moving in the same direction, suggesting a shared "deception" component.

### 5. The deception signal emerges at layers 8-14

Cosine similarity between truthful and deceptive centroids drops sharply from ~0.999 (layers 0-4) to ~0.93 (layers 10-14). This transition region is where the model builds its representation of deceptive intent. By layer 16 the signal stabilizes. Late layers (28-31) show a second dip, consistent with the model preparing different output distributions for truthful vs deceptive responses.

The "Fabrication vs Creative" similarity follows a parallel trajectory: starting at ~0.96 in early layers (both involve generating plausible text), then separating to ~0.78 by mid-network.

### 6. Cosine similarity heatmap reveals block structure

At layer 16, the centroid similarity matrix shows clear block structure:

- **Game block** (truthful + all deception types): 0.846-0.910 within-block similarity
- **Control conditions**: 0.505-0.677 between each other
- **Cross-block**: 0.50-0.77 (game vs controls)

The model organizes its representations primarily by task domain, with deception as a within-domain modulation.

## Implications for Probe Design

1. **A deception probe should be trained on game-context data only.** Cross-domain signals (creative, recall, suppression) live in a different part of activation space and would not help detect in-game deception.

2. **Layers 10-16 are the sweet spot** for probing. The deception signal has emerged but hasn't yet been confounded by output-level processing.

3. **Omission will be hardest to detect** (closest to truthful). Distortion will be easiest. A multiclass probe may benefit from type-specific features.

4. **The deception signal is subtle but real.** Within the game cluster, deception types show consistent spatial tendencies and specific discriminative dimensions. A linear probe should be able to exploit these.

## Next Steps

- Train binary deception probes (logistic regression) at each layer, report accuracy/precision/recall
- Train multiclass probe (deception type classification)
- Increase sample size for more robust estimates
- Extend to multi-turn game dialogues (Phase 2)
