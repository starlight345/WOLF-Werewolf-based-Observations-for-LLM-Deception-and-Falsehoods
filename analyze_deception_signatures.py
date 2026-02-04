"""
Analyze HOW different deception types manifest in hidden states.

Instead of asking "is the model lying?" this asks "how is it lying?"
by computing deception delta vectors (deception_centroid - truthful_centroid)
and comparing those deltas across types.

Produces:
  1. Deception delta heatmap: fingerprint of each lie type on top dimensions
  2. Delta direction similarity: do different lies modify representations the same way?
  3. Shared vs unique dimensions: which dimensions are pan-deceptive vs type-specific?
  4. Radar chart: per-type activation profiles on top discriminative dimensions
  5. Dendrogram: hierarchical clustering of deception types by their delta vectors
  6. Layer evolution: how each type's delta magnitude evolves across layers

Usage:
    python analyze_deception_signatures.py [--data-dir ./hidden_states] [--output-dir ./figures] [--layer 16]
"""

import argparse
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from collections import OrderedDict

DECEPTION_TYPES = OrderedDict([
    ("fabrication",  {"color": "#E53935", "name": "Fabrication"}),
    ("omission",     {"color": "#FF9800", "name": "Omission"}),
    ("misdirection", {"color": "#9C27B0", "name": "Misdirection"}),
    ("distortion",   {"color": "#D32F2F", "name": "Distortion"}),
])

TRUTHFUL_COLOR = "#2196F3"


def load_data(data_dir):
    path = os.path.join(data_dir, "hidden_states.pt")
    data = torch.load(path, map_location="cpu", weights_only=False)
    hs = data["hidden_states"].float().numpy()
    labels = data["labels"].numpy()
    dtypes = data["deception_types"]
    return hs, labels, dtypes


def get_masks(dtypes):
    truth_mask = np.array([d == "none" for d in dtypes])
    type_masks = {}
    for dtype_name in DECEPTION_TYPES:
        type_masks[dtype_name] = np.array([d == dtype_name for d in dtypes])
    return truth_mask, type_masks


def compute_deltas(hs, truth_mask, type_masks, layer):
    """Compute delta vectors: deception_centroid - truthful_centroid at given layer."""
    truth_cent = hs[truth_mask, layer, :].mean(axis=0)
    deltas = {}
    for dtype_name, mask in type_masks.items():
        if mask.sum() > 0:
            deltas[dtype_name] = hs[mask, layer, :].mean(axis=0) - truth_cent
    return deltas, truth_cent


def select_top_dims(deltas, top_k=40):
    """Select dimensions with largest absolute delta across any deception type."""
    all_deltas = np.stack(list(deltas.values()))  # [n_types, hidden_dim]
    max_abs = np.max(np.abs(all_deltas), axis=0)
    top_dims = np.argsort(max_abs)[::-1][:top_k]
    return top_dims


# ── Figure 1: Deception delta heatmap ────────────────────────────────────────

def plot_delta_heatmap(hs, truth_mask, type_masks, output_dir, layer=16, top_k=50):
    deltas, _ = compute_deltas(hs, truth_mask, type_masks, layer)
    top_dims = select_top_dims(deltas, top_k)

    # Build matrix [n_types x top_k]
    type_names = list(deltas.keys())
    matrix = np.stack([deltas[t][top_dims] for t in type_names])

    fig, ax = plt.subplots(figsize=(16, 4))
    vmax = np.max(np.abs(matrix)) * 0.9
    im = ax.imshow(matrix, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax,
                   interpolation="nearest")

    ax.set_yticks(range(len(type_names)))
    ax.set_yticklabels([DECEPTION_TYPES[t]["name"] for t in type_names], fontsize=10)
    ax.set_xticks(range(top_k))
    ax.set_xticklabels([f"d{d}" for d in top_dims], rotation=90, fontsize=5.5)
    ax.set_xlabel("Hidden Dimension Index", fontsize=10)
    ax.set_title(f"Deception Delta Fingerprints (Layer {layer})\n"
                 f"Each cell = (deception_type_mean - truthful_mean) for that dimension",
                 fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.7, label="Activation delta from truthful",
                 orientation="horizontal", pad=0.25)
    fig.tight_layout()
    path = os.path.join(output_dir, f"delta_heatmap_layer{layer}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 2: Delta direction similarity ─────────────────────────────────────

def plot_delta_similarity(hs, truth_mask, type_masks, output_dir, layer=16):
    """Cosine similarity between the delta DIRECTIONS (not raw activations).
    This answers: do different lies modify the representation in the same way?"""
    deltas, _ = compute_deltas(hs, truth_mask, type_masks, layer)
    type_names = list(deltas.keys())
    delta_vecs = np.stack([deltas[t] for t in type_names])

    # Normalize
    norms = np.linalg.norm(delta_vecs, axis=1, keepdims=True)
    normed = delta_vecs / (norms + 1e-8)
    cos_sim = normed @ normed.T

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cos_sim, cmap="YlOrRd", vmin=-0.2, vmax=1.0, aspect="auto")
    labels = [DECEPTION_TYPES[t]["name"] for t in type_names]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    for i in range(len(labels)):
        for j in range(len(labels)):
            color = "white" if cos_sim[i, j] > 0.6 else "black"
            ax.text(j, i, f"{cos_sim[i, j]:.3f}", ha="center", va="center",
                    fontsize=11, fontweight="bold", color=color)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Cosine similarity of delta directions")
    ax.set_title(f"How Similarly Do Different Lies Modify the Representation?\n"
                 f"(Delta direction cosine similarity, Layer {layer})",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(output_dir, f"delta_similarity_layer{layer}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    return cos_sim, type_names


# ── Figure 3: Shared vs unique dimensions ────────────────────────────────────

def plot_shared_vs_unique(hs, truth_mask, type_masks, output_dir, layer=16,
                          top_k=30, threshold_ratio=0.5):
    """Categorize top dimensions as shared (all types shift same direction)
    vs type-specific (only one type shifts significantly)."""
    deltas, _ = compute_deltas(hs, truth_mask, type_masks, layer)
    top_dims = select_top_dims(deltas, top_k)
    type_names = list(deltas.keys())
    matrix = np.stack([deltas[t][top_dims] for t in type_names])  # [4, top_k]

    # Classify each dimension
    categories = []  # 'shared', 'type-specific: X', or 'mixed'
    cat_colors = []
    for j in range(top_k):
        col = matrix[:, j]
        abs_col = np.abs(col)
        max_val = abs_col.max()
        if max_val < 1e-6:
            categories.append("negligible")
            cat_colors.append("#CCCCCC")
            continue

        # Check if all types shift in the same direction with significant magnitude
        signs = np.sign(col)
        same_sign = np.all(signs == signs[0]) and np.all(abs_col > max_val * threshold_ratio)
        if same_sign:
            categories.append("shared")
            cat_colors.append("#607D8B")
        else:
            # Find dominant type
            dominant_idx = np.argmax(abs_col)
            # Check if dominant type is >2x the next highest
            sorted_abs = np.sort(abs_col)[::-1]
            if sorted_abs[0] > 2 * sorted_abs[1]:
                dominant_name = type_names[dominant_idx]
                categories.append(f"specific: {dominant_name}")
                cat_colors.append(DECEPTION_TYPES[dominant_name]["color"])
            else:
                categories.append("mixed")
                cat_colors.append("#9E9E9E")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), gridspec_kw={"height_ratios": [3, 1]})

    # Top: grouped bar chart showing per-type deltas
    x = np.arange(top_k)
    width = 0.2
    for i, tname in enumerate(type_names):
        vals = matrix[i]
        ax1.bar(x + i * width, vals, width,
                label=DECEPTION_TYPES[tname]["name"],
                color=DECEPTION_TYPES[tname]["color"], alpha=0.85,
                edgecolor="k", linewidth=0.2)
    ax1.set_xticks(x + 1.5 * width)
    ax1.set_xticklabels([f"d{d}" for d in top_dims], rotation=90, fontsize=6)
    ax1.axhline(0, color="k", linewidth=0.5)
    ax1.set_ylabel("Activation delta from truthful", fontsize=9)
    ax1.set_title(f"Shared vs Type-Specific Deception Dimensions (Layer {layer})",
                  fontsize=11, fontweight="bold")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(True, alpha=0.15, axis="y")

    # Bottom: category strip
    for j in range(top_k):
        ax2.barh(0, 1, left=j, color=cat_colors[j], edgecolor="white", linewidth=0.5)
    ax2.set_xlim(0, top_k)
    ax2.set_yticks([])
    ax2.set_xticks(x + 0.5)
    ax2.set_xticklabels([f"d{d}" for d in top_dims], rotation=90, fontsize=6)
    ax2.set_xlabel("Hidden Dimension Index", fontsize=9)

    # Legend for categories
    legend_items = [
        mpatches.Patch(color="#607D8B", label="Shared (all types)"),
        mpatches.Patch(color="#9E9E9E", label="Mixed"),
    ]
    for tname in type_names:
        legend_items.append(
            mpatches.Patch(color=DECEPTION_TYPES[tname]["color"],
                           label=f"Specific: {DECEPTION_TYPES[tname]['name']}")
        )
    ax2.legend(handles=legend_items, loc="upper right", fontsize=7, ncol=3)

    fig.tight_layout()
    path = os.path.join(output_dir, f"shared_vs_unique_layer{layer}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # Print summary
    from collections import Counter
    cat_counts = Counter([c.split(":")[0].strip() for c in categories])
    print(f"    Dimension classification: {dict(cat_counts)}")


# ── Figure 4: Radar chart ────────────────────────────────────────────────────

def plot_radar(hs, truth_mask, type_masks, output_dir, layer=16, n_dims=12):
    """Radar/spider chart showing each type's activation profile on top dimensions."""
    deltas, _ = compute_deltas(hs, truth_mask, type_masks, layer)
    top_dims = select_top_dims(deltas, n_dims)
    type_names = list(deltas.keys())

    # Normalize deltas to [-1, 1] range per dimension for visibility
    matrix = np.stack([deltas[t][top_dims] for t in type_names])
    max_abs = np.max(np.abs(matrix), axis=0, keepdims=True)
    matrix_norm = matrix / (max_abs + 1e-8)

    angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for i, tname in enumerate(type_names):
        values = matrix_norm[i].tolist()
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, markersize=5,
                color=DECEPTION_TYPES[tname]["color"],
                label=DECEPTION_TYPES[tname]["name"])
        ax.fill(angles, values, alpha=0.08, color=DECEPTION_TYPES[tname]["color"])

    dim_labels = [f"d{d}" for d in top_dims]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dim_labels, fontsize=8)
    ax.set_ylim(-1.1, 1.1)
    ax.set_title(f"Deception Type Activation Profiles (Layer {layer})\n"
                 f"Normalized delta from truthful on top {n_dims} dimensions",
                 fontsize=11, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    fig.tight_layout()
    path = os.path.join(output_dir, f"radar_profiles_layer{layer}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 5: Dendrogram ─────────────────────────────────────────────────────

def plot_dendrogram(hs, truth_mask, type_masks, output_dir, layer=16):
    """Hierarchical clustering of deception types by their full delta vectors."""
    deltas, _ = compute_deltas(hs, truth_mask, type_masks, layer)
    type_names = list(deltas.keys())
    delta_matrix = np.stack([deltas[t] for t in type_names])

    # Use cosine distance for clustering
    dists = pdist(delta_matrix, metric="cosine")
    Z = linkage(dists, method="average")

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [DECEPTION_TYPES[t]["color"] for t in type_names]
    labels = [DECEPTION_TYPES[t]["name"] for t in type_names]

    dendrogram(Z, labels=labels, ax=ax, leaf_font_size=11,
               above_threshold_color="#999999")
    ax.set_ylabel("Cosine Distance Between Delta Vectors", fontsize=10)
    ax.set_title(f"Mechanistic Similarity Between Deception Types (Layer {layer})\n"
                 f"Closer = lies modify the representation more similarly",
                 fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.15, axis="y")
    fig.tight_layout()
    path = os.path.join(output_dir, f"dendrogram_layer{layer}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 6: Delta magnitude across layers ──────────────────────────────────

def plot_delta_evolution(hs, truth_mask, type_masks, output_dir):
    """How the magnitude (L2 norm) of each type's delta vector evolves across layers."""
    num_layers = hs.shape[1]
    type_names = list(DECEPTION_TYPES.keys())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: L2 norm of delta vectors
    for tname in type_names:
        mask = type_masks[tname]
        if mask.sum() == 0:
            continue
        norms = []
        for l in range(num_layers):
            delta, _ = compute_deltas(hs, truth_mask, type_masks, l)
            norms.append(np.linalg.norm(delta[tname]))
        ax1.plot(range(num_layers), norms, linewidth=2,
                 color=DECEPTION_TYPES[tname]["color"],
                 label=DECEPTION_TYPES[tname]["name"])

    ax1.set_xlabel("Transformer Layer", fontsize=10)
    ax1.set_ylabel("L2 Norm of Delta Vector", fontsize=10)
    ax1.set_title("How Much Does Each Lie Type\nShift the Representation?",
                  fontsize=11, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.2)

    # Right: pairwise delta cosine similarity across layers
    # Track how fabrication vs omission delta directions change across layers
    pairs = [
        ("fabrication", "omission"),
        ("fabrication", "misdirection"),
        ("fabrication", "distortion"),
        ("omission", "misdirection"),
        ("omission", "distortion"),
        ("misdirection", "distortion"),
    ]
    pair_colors = ["#E53935", "#7B1FA2", "#C62828", "#F57C00", "#BF360C", "#6A1B9A"]

    for (t1, t2), pcol in zip(pairs, pair_colors):
        sims = []
        for l in range(num_layers):
            delta, _ = compute_deltas(hs, truth_mask, type_masks, l)
            d1 = delta[t1]
            d2 = delta[t2]
            cos = np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2) + 1e-8)
            sims.append(cos)
        short_names = (DECEPTION_TYPES[t1]["name"][:4],
                       DECEPTION_TYPES[t2]["name"][:4])
        ax2.plot(range(num_layers), sims, linewidth=1.5, color=pcol,
                 label=f"{short_names[0]} vs {short_names[1]}", alpha=0.85)

    ax2.set_xlabel("Transformer Layer", fontsize=10)
    ax2.set_ylabel("Cosine Similarity of Delta Directions", fontsize=10)
    ax2.set_title("Do Lie Types Diverge or Converge\nAcross Layers?",
                  fontsize=11, fontweight="bold")
    ax2.legend(fontsize=7, loc="lower left")
    ax2.grid(True, alpha=0.2)
    ax2.set_ylim(-0.5, 1.05)
    ax2.axhline(0, color="k", linewidth=0.5, linestyle="--")

    fig.tight_layout()
    path = os.path.join(output_dir, "delta_evolution_across_layers.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze deception type signatures in hidden states")
    parser.add_argument("--data-dir", default="./hidden_states")
    parser.add_argument("--output-dir", default="./figures")
    parser.add_argument("--layer", type=int, default=16,
                        help="Primary layer for detailed analysis")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading data...")
    hs, labels, dtypes = load_data(args.data_dir)
    truth_mask, type_masks = get_masks(dtypes)
    print(f"  Samples: {len(labels)}, Layers: {hs.shape[1]}, Hidden dim: {hs.shape[2]}")
    print(f"  Truthful: {truth_mask.sum()}, " +
          ", ".join(f"{DECEPTION_TYPES[t]['name']}: {m.sum()}"
                    for t, m in type_masks.items()))

    print(f"\n[1/6] Deception delta heatmap (layer {args.layer})...")
    plot_delta_heatmap(hs, truth_mask, type_masks, args.output_dir, layer=args.layer)

    print(f"[2/6] Delta direction similarity (layer {args.layer})...")
    cos_sim, type_names = plot_delta_similarity(hs, truth_mask, type_masks,
                                                 args.output_dir, layer=args.layer)
    print(f"    Delta similarity matrix:")
    for i, t1 in enumerate(type_names):
        for j, t2 in enumerate(type_names):
            if j > i:
                print(f"      {DECEPTION_TYPES[t1]['name']:15s} vs "
                      f"{DECEPTION_TYPES[t2]['name']:15s}: {cos_sim[i, j]:.4f}")

    print(f"\n[3/6] Shared vs unique dimensions (layer {args.layer})...")
    plot_shared_vs_unique(hs, truth_mask, type_masks, args.output_dir, layer=args.layer)

    print(f"\n[4/6] Radar profiles (layer {args.layer})...")
    plot_radar(hs, truth_mask, type_masks, args.output_dir, layer=args.layer)

    print(f"\n[5/6] Dendrogram (layer {args.layer})...")
    plot_dendrogram(hs, truth_mask, type_masks, args.output_dir, layer=args.layer)

    print(f"\n[6/6] Delta evolution across layers...")
    plot_delta_evolution(hs, truth_mask, type_masks, args.output_dir)

    print(f"\nAll figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
