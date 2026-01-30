"""
Utility script for analyzing activations logged during WOLF games.

This script provides tools to:
1. Load activation data from a game run
2. Compute basic statistics and patterns
3. Extract representation primitives (contrastive directions, PCA, etc.)
4. Visualize activation patterns

Usage:
    python analyze_activations.py --run-dir logs/TIMESTAMP
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from collections import defaultdict


def load_events(events_path: Path) -> List[Dict]:
    """Load all events from NDJSON file."""
    events = []
    with open(events_path, 'r') as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line))
    return events


def load_activation(activation_path: Path) -> Dict[str, np.ndarray]:
    """Load activation data from NPZ file."""
    data = np.load(activation_path)
    return {key: data[key] for key in data.files}


def collect_activations_by_label(
    events: List[Dict],
    activations_dir: Path,
    label_key: str = "is_deceptive"
) -> Dict[bool, List[np.ndarray]]:
    """
    Collect activations grouped by a boolean label (e.g., deceptive vs truthful).
    
    Returns:
        Dictionary mapping label value to list of activation arrays
    """
    grouped = defaultdict(list)
    
    for event in events:
        if event.get("event") != "debate":
            continue
        
        activation_file = event.get("activation_file")
        if not activation_file:
            continue
        
        # Get label from event details
        raw_output = event.get("details", {}).get("raw_output", {})
        label = raw_output.get(label_key, False)
        
        # Load activation
        act_path = activations_dir / activation_file
        if not act_path.exists():
            print(f"Warning: Activation file not found: {act_path}")
            continue
        
        act_data = load_activation(act_path)
        hidden_states = act_data["hidden_states"]  # [n_layers, hidden_dim]
        
        grouped[label].append(hidden_states)
    
    return dict(grouped)


def compute_contrastive_direction(
    deceptive_acts: List[np.ndarray],
    truthful_acts: List[np.ndarray],
    layer_idx: int = -1
) -> np.ndarray:
    """
    Compute the contrastive direction between deceptive and truthful activations.
    
    Args:
        deceptive_acts: List of activation arrays [n_layers, hidden_dim]
        truthful_acts: List of activation arrays [n_layers, hidden_dim]
        layer_idx: Which layer to use (default: -1 = last layer)
    
    Returns:
        Normalized contrastive direction vector
    """
    # Extract specified layer from all activations
    deceptive_layer = np.stack([act[layer_idx] for act in deceptive_acts])
    truthful_layer = np.stack([act[layer_idx] for act in truthful_acts])
    
    # Compute mean difference
    mean_deceptive = np.mean(deceptive_layer, axis=0)
    mean_truthful = np.mean(truthful_layer, axis=0)
    
    contrast_vector = mean_deceptive - mean_truthful
    
    # Normalize
    contrast_vector = contrast_vector / (np.linalg.norm(contrast_vector) + 1e-8)
    
    return contrast_vector


def compute_layer_wise_separation(
    deceptive_acts: List[np.ndarray],
    truthful_acts: List[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute separation distance and statistical significance across all layers.
    
    Returns:
        distances: Array of distances for each layer
        t_stats: Array of t-statistics for each layer
    """
    n_layers = deceptive_acts[0].shape[0]
    distances = []
    t_stats = []
    
    for layer_idx in range(n_layers):
        deceptive_layer = np.stack([act[layer_idx] for act in deceptive_acts])
        truthful_layer = np.stack([act[layer_idx] for act in truthful_acts])
        
        # Mean distance
        mean_dec = np.mean(deceptive_layer, axis=0)
        mean_truth = np.mean(truthful_layer, axis=0)
        distance = np.linalg.norm(mean_dec - mean_truth)
        distances.append(distance)
        
        # T-statistic (simplified)
        std_dec = np.std(deceptive_layer, axis=0).mean()
        std_truth = np.std(truthful_layer, axis=0).mean()
        pooled_std = np.sqrt((std_dec**2 + std_truth**2) / 2)
        t_stat = distance / (pooled_std + 1e-8)
        t_stats.append(t_stat)
    
    return np.array(distances), np.array(t_stats)


def visualize_layer_separation(
    distances: np.ndarray,
    layer_indices: List[int],
    save_path: Path = None
):
    """Visualize layer-wise separation between deceptive and truthful activations."""
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(layer_indices, distances, marker='o')
    plt.xlabel("Layer Index")
    plt.ylabel("Separation Distance")
    plt.title("Layer-wise Separation: Deceptive vs Truthful")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.bar(layer_indices, distances)
    plt.xlabel("Layer Index")
    plt.ylabel("Separation Distance")
    plt.title("Layer-wise Separation (Bar Chart)")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()


def print_summary_statistics(
    deceptive_acts: List[np.ndarray],
    truthful_acts: List[np.ndarray]
):
    """Print summary statistics of the activation data."""
    print("\n" + "="*60)
    print("ACTIVATION DATA SUMMARY")
    print("="*60)
    
    print(f"\nNumber of deceptive statements: {len(deceptive_acts)}")
    print(f"Number of truthful statements: {len(truthful_acts)}")
    
    if len(deceptive_acts) > 0:
        n_layers, hidden_dim = deceptive_acts[0].shape
        print(f"\nActivation shape: [{n_layers} layers, {hidden_dim} hidden_dim]")
        
        # Compute norms
        dec_norms = [np.linalg.norm(act[-1]) for act in deceptive_acts]
        truth_norms = [np.linalg.norm(act[-1]) for act in truthful_acts]
        
        print(f"\nLast layer activation norms:")
        print(f"  Deceptive: mean={np.mean(dec_norms):.2f}, std={np.std(dec_norms):.2f}")
        print(f"  Truthful:  mean={np.mean(truth_norms):.2f}, std={np.std(truth_norms):.2f}")


def main():
    parser = argparse.ArgumentParser(description="Analyze WOLF activation logs")
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to the run directory (e.g., logs/TIMESTAMP)"
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=-1,
        help="Layer index to analyze (default: -1 = last layer)"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization plots"
    )
    parser.add_argument(
        "--save-primitives",
        action="store_true",
        help="Save extracted primitives (contrastive vectors, etc.) to disk"
    )
    
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}")
        return
    
    # Load events
    events_path = run_dir / "events.ndjson"
    if not events_path.exists():
        print(f"Error: Events file not found: {events_path}")
        return
    
    print(f"Loading events from {events_path}...")
    events = load_events(events_path)
    print(f"Loaded {len(events)} events")
    
    # Collect activations
    activations_dir = run_dir / "activations"
    if not activations_dir.exists():
        print(f"Error: Activations directory not found: {activations_dir}")
        return
    
    print(f"\nCollecting activations by deception label...")
    grouped = collect_activations_by_label(events, activations_dir)
    
    deceptive_acts = grouped.get(True, [])
    truthful_acts = grouped.get(False, [])
    
    if len(deceptive_acts) == 0 and len(truthful_acts) == 0:
        print("Error: No activations found with deception labels")
        return
    
    # Print summary
    print_summary_statistics(deceptive_acts, truthful_acts)
    
    # Compute contrastive direction
    if len(deceptive_acts) > 0 and len(truthful_acts) > 0:
        print(f"\nComputing contrastive direction for layer {args.layer}...")
        contrast_vector = compute_contrastive_direction(
            deceptive_acts,
            truthful_acts,
            layer_idx=args.layer
        )
        print(f"Contrastive vector norm: {np.linalg.norm(contrast_vector):.4f}")
        print(f"Contrastive vector shape: {contrast_vector.shape}")
        
        # Layer-wise separation
        print("\nComputing layer-wise separation...")
        distances, t_stats = compute_layer_wise_separation(deceptive_acts, truthful_acts)
        
        n_layers = len(distances)
        layer_indices = list(range(n_layers))
        
        print("\nLayer-wise separation distances:")
        for i, (dist, t) in enumerate(zip(distances, t_stats)):
            print(f"  Layer {i:2d}: distance={dist:8.2f}, t-stat={t:6.2f}")
        
        # Visualization
        if args.visualize:
            print("\nGenerating visualizations...")
            viz_path = run_dir / "layer_separation.png"
            visualize_layer_separation(distances, layer_indices, save_path=viz_path)
        
        # Save primitives
        if args.save_primitives:
            print("\nSaving representation primitives...")
            primitives_path = run_dir / "representation_primitives.npz"
            np.savez_compressed(
                primitives_path,
                contrast_vector=contrast_vector,
                layer_distances=distances,
                layer_t_stats=t_stats,
                layer_indices=layer_indices
            )
            print(f"Saved primitives to {primitives_path}")
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)


if __name__ == "__main__":
    main()
