"""
Visualize hidden state activations across deception types and cognitive controls.

Produces:
  1. PCA grid across layers (all conditions)
  2. UMAP at best layer
  3. Cosine similarity heatmap between condition centroids
  4. Fabrication-vs-creativity / omission-vs-suppression overlay
  5. Top discriminative dimensions bar chart

Usage:
    python visualize_hidden_states.py [--data-dir ./hidden_states] [--output-dir ./figures]
"""

import argparse
import json
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from collections import OrderedDict

# Condition display config: color, marker, display name
CONDITION_STYLE = OrderedDict([
    ("none",               {"color": "#2196F3", "marker": "o", "name": "Truthful (game)"}),
    ("fabrication",        {"color": "#E53935", "marker": "X", "name": "Fabrication"}),
    ("omission",           {"color": "#FF9800", "marker": "s", "name": "Omission"}),
    ("misdirection",       {"color": "#9C27B0", "marker": "D", "name": "Misdirection"}),
    ("distortion",         {"color": "#F44336", "marker": "^", "name": "Distortion"}),
    ("control_creative",   {"color": "#4CAF50", "marker": "P", "name": "Creative (ctrl)"}),
    ("control_recall",     {"color": "#00BCD4", "marker": "v", "name": "Factual recall (ctrl)"}),
    ("control_suppression",{"color": "#FFC107", "marker": "h", "name": "Suppression (ctrl)"}),
])

DECEPTIVE_TYPES = {"fabrication", "omission", "misdirection", "distortion"}


def load_data(data_dir):
    path = os.path.join(data_dir, "hidden_states.pt")
    data = torch.load(path, map_location="cpu", weights_only=False)
    hs = data["hidden_states"].float().numpy()  # [N, num_layers, hidden_dim]
    labels = data["labels"].numpy()
    dtypes = data["deception_types"]
    metadata = data["metadata"]
    return hs, labels, dtypes, metadata


def build_legend(ax, conditions_present):
    handles = []
    for cond, style in CONDITION_STYLE.items():
        if cond in conditions_present:
            handles.append(
                ax.scatter([], [], c=style["color"], marker=style["marker"],
                           s=60, label=style["name"], edgecolors="k", linewidths=0.3)
            )
    ax.legend(handles=handles, loc="best", fontsize=7, framealpha=0.8)


def scatter_by_condition(ax, coords_2d, dtypes, title, conditions_present=None):
    if conditions_present is None:
        conditions_present = set(dtypes)
    for cond, style in CONDITION_STYLE.items():
        mask = np.array([d == cond for d in dtypes])
        if mask.sum() == 0:
            continue
        ax.scatter(
            coords_2d[mask, 0], coords_2d[mask, 1],
            c=style["color"], marker=style["marker"],
            s=70, edgecolors="k", linewidths=0.3, zorder=3,
            alpha=0.85,
        )
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True, alpha=0.15)


# ── Figure 1: PCA grid across layers ─────────────────────────────────────

def plot_pca_grid(hs, dtypes, output_dir):
    layers_to_show = [1, 4, 8, 12, 16, 20, 24, 28, 31]
    n = len(layers_to_show)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 4))
    axes = axes.flatten()

    conditions_present = set(dtypes)

    for idx, layer in enumerate(layers_to_show):
        ax = axes[idx]
        X = hs[:, layer, :]
        X_scaled = StandardScaler().fit_transform(X)
        pca = PCA(n_components=2)
        coords = pca.fit_transform(X_scaled)
        ev = pca.explained_variance_ratio_
        scatter_by_condition(ax, coords, dtypes,
                            f"Layer {layer}  (PC1={ev[0]:.1%}, PC2={ev[1]:.1%})",
                            conditions_present)

    # Hide unused subplots
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    # Single shared legend
    build_legend(axes[0], conditions_present)

    fig.suptitle("PCA of Hidden States Across Transformer Layers", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = os.path.join(output_dir, "pca_grid_layers.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 2: UMAP at best layer ─────────────────────────────────────────

def plot_umap(hs, dtypes, output_dir, layer=16):
    try:
        from umap import UMAP
    except ImportError:
        print("  [skip] umap-learn not installed, skipping UMAP plot")
        return

    X = hs[:, layer, :]
    X_scaled = StandardScaler().fit_transform(X)
    reducer = UMAP(n_components=2, n_neighbors=min(10, len(X) - 1),
                   min_dist=0.3, metric="cosine", random_state=42)
    coords = reducer.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter_by_condition(ax, coords, dtypes, f"UMAP of Layer {layer} Hidden States")
    build_legend(ax, set(dtypes))
    fig.tight_layout()
    path = os.path.join(output_dir, f"umap_layer{layer}.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 3: Cosine similarity heatmap between condition centroids ───────

def plot_cosine_heatmap(hs, dtypes, output_dir, layer=16):
    conditions = [c for c in CONDITION_STYLE if c in set(dtypes)]
    centroids = []
    for cond in conditions:
        mask = np.array([d == cond for d in dtypes])
        centroids.append(hs[mask, layer, :].mean(axis=0))
    centroids = np.stack(centroids)

    # Cosine similarity matrix
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    normed = centroids / (norms + 1e-8)
    cos_sim = normed @ normed.T

    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(cos_sim, cmap="RdYlBu_r", vmin=0.75, vmax=1.0, aspect="auto")
    cond_names = [CONDITION_STYLE[c]["name"] for c in conditions]
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(cond_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(conditions)))
    ax.set_yticklabels(cond_names, fontsize=8)
    for i in range(len(conditions)):
        for j in range(len(conditions)):
            ax.text(j, i, f"{cos_sim[i, j]:.3f}", ha="center", va="center", fontsize=7,
                    color="white" if cos_sim[i, j] < 0.88 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Cosine similarity")
    ax.set_title(f"Centroid Cosine Similarity (Layer {layer})", fontsize=11, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(output_dir, f"cosine_heatmap_layer{layer}.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 4: Deception-type vs cognitive-control overlays ────────────────

def plot_deception_vs_controls(hs, dtypes, output_dir, layer=16):
    """
    Two side-by-side PCA plots:
      Left:  fabrication vs creative (does fabrication look like creativity?)
      Right: omission vs suppression (does omission look like restraint?)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # -- Left panel: fabrication + creative + truthful baseline --
    pairs_left = ["none", "fabrication", "control_creative"]
    mask_left = np.array([d in pairs_left for d in dtypes])
    X_left = hs[mask_left, layer, :]
    X_left_scaled = StandardScaler().fit_transform(X_left)
    pca_left = PCA(n_components=2).fit_transform(X_left_scaled)
    dtypes_left = [d for d, m in zip(dtypes, mask_left) if m]
    scatter_by_condition(ax1, pca_left, dtypes_left,
                         f"Fabrication vs Creativity (Layer {layer})")
    build_legend(ax1, set(dtypes_left))

    # -- Right panel: omission + suppression + truthful baseline --
    pairs_right = ["none", "omission", "control_suppression"]
    mask_right = np.array([d in pairs_right for d in dtypes])
    X_right = hs[mask_right, layer, :]
    X_right_scaled = StandardScaler().fit_transform(X_right)
    pca_right = PCA(n_components=2).fit_transform(X_right_scaled)
    dtypes_right = [d for d, m in zip(dtypes, mask_right) if m]
    scatter_by_condition(ax2, pca_right, dtypes_right,
                         f"Omission vs Suppression (Layer {layer})")
    build_legend(ax2, set(dtypes_right))

    fig.suptitle("Do Deception Types Share Circuits with Cognitive Controls?",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = os.path.join(output_dir, f"deception_vs_controls_layer{layer}.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 5: Cosine similarity progression across layers ─────────────────

def plot_similarity_across_layers(hs, dtypes, output_dir):
    """
    Line plot showing how each deception type's centroid similarity to
    truthful (and to its cognitive control) evolves across layers.
    """
    num_layers = hs.shape[1]
    truth_mask = np.array([d == "none" for d in dtypes])

    # Pairs: deception_type -> which control to compare against
    compare_pairs = [
        ("fabrication",  "control_creative",    "#E53935", "#4CAF50"),
        ("omission",     "control_suppression",  "#FF9800", "#FFC107"),
        ("misdirection", None,                   "#9C27B0", None),
        ("distortion",   None,                   "#F44336", None),
    ]

    fig, ax = plt.subplots(figsize=(12, 5))
    layers = np.arange(num_layers)

    for dtype_name, ctrl_name, dcol, ccol in compare_pairs:
        d_mask = np.array([d == dtype_name for d in dtypes])
        if d_mask.sum() == 0:
            continue

        # Cosine sim: deception_type centroid vs truthful centroid per layer
        sims_truth = []
        sims_ctrl = []
        for l in range(num_layers):
            t_cent = hs[truth_mask, l, :].mean(axis=0)
            d_cent = hs[d_mask, l, :].mean(axis=0)
            cos = np.dot(t_cent, d_cent) / (np.linalg.norm(t_cent) * np.linalg.norm(d_cent) + 1e-8)
            sims_truth.append(cos)
            if ctrl_name:
                c_mask = np.array([d == ctrl_name for d in dtypes])
                c_cent = hs[c_mask, l, :].mean(axis=0)
                cos_dc = np.dot(d_cent, c_cent) / (np.linalg.norm(d_cent) * np.linalg.norm(c_cent) + 1e-8)
                sims_ctrl.append(cos_dc)

        label_t = f"{CONDITION_STYLE[dtype_name]['name']} vs Truthful"
        ax.plot(layers, sims_truth, color=dcol, linewidth=2, label=label_t)

        if ctrl_name and sims_ctrl:
            label_c = f"{CONDITION_STYLE[dtype_name]['name']} vs {CONDITION_STYLE[ctrl_name]['name']}"
            ax.plot(layers, sims_ctrl, color=dcol, linewidth=1.5, linestyle="--", alpha=0.7,
                    label=label_c)

    ax.set_xlabel("Transformer Layer", fontsize=10)
    ax.set_ylabel("Cosine Similarity (centroids)", fontsize=10)
    ax.set_title("Centroid Similarity Across Layers", fontsize=12, fontweight="bold")
    ax.legend(fontsize=7, loc="lower left", ncol=2)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(0, num_layers - 1)
    ax.set_ylim(0.7, 1.01)
    fig.tight_layout()
    path = os.path.join(output_dir, "similarity_across_layers.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 6: Top discriminative dimensions ───────────────────────────────

def plot_top_dimensions(hs, dtypes, output_dir, layer=16, top_k=30):
    """
    For a given layer, find which hidden dimensions differ most between
    truthful and deceptive (mean difference), and show per-type breakdown.
    """
    truth_mask = np.array([d == "none" for d in dtypes])
    decep_mask = np.array([d in DECEPTIVE_TYPES for d in dtypes])

    truth_mean = hs[truth_mask, layer, :].mean(axis=0)
    decep_mean = hs[decep_mask, layer, :].mean(axis=0)
    diff = decep_mean - truth_mean
    top_dims = np.argsort(np.abs(diff))[::-1][:top_k]

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(top_k)
    width = 0.18

    # Show per deception-type contribution
    for i, (dtype_name, style) in enumerate(
        [(k, v) for k, v in CONDITION_STYLE.items() if k in DECEPTIVE_TYPES]
    ):
        d_mask = np.array([d == dtype_name for d in dtypes])
        if d_mask.sum() == 0:
            continue
        d_mean = hs[d_mask, layer, :].mean(axis=0)
        vals = (d_mean - truth_mean)[top_dims]
        ax.bar(x + i * width, vals, width, label=style["name"],
               color=style["color"], alpha=0.8, edgecolor="k", linewidth=0.3)

    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([f"d{d}" for d in top_dims], rotation=90, fontsize=6)
    ax.set_xlabel(f"Hidden Dimension Index (Layer {layer})", fontsize=9)
    ax.set_ylabel("Mean Activation Difference from Truthful", fontsize=9)
    ax.set_title(f"Top {top_k} Most Discriminative Dimensions (Layer {layer})", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.grid(True, alpha=0.15, axis="y")
    fig.tight_layout()
    path = os.path.join(output_dir, f"top_dimensions_layer{layer}.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize hidden states for deception analysis")
    parser.add_argument("--data-dir", default="./hidden_states")
    parser.add_argument("--output-dir", default="./figures")
    parser.add_argument("--layer", type=int, default=16, help="Primary layer for detailed plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading data...")
    hs, labels, dtypes, metadata = load_data(args.data_dir)
    print(f"  Samples: {len(labels)}, Layers: {hs.shape[1]}, Hidden dim: {hs.shape[2]}")
    print(f"  Conditions: {dict(zip(*np.unique(dtypes, return_counts=True)))}")

    print("\n[1/6] PCA grid across layers...")
    plot_pca_grid(hs, dtypes, args.output_dir)

    print("[2/6] UMAP at layer {args.layer}...")
    plot_umap(hs, dtypes, args.output_dir, layer=args.layer)

    print(f"[3/6] Cosine similarity heatmap (layer {args.layer})...")
    plot_cosine_heatmap(hs, dtypes, args.output_dir, layer=args.layer)

    print(f"[4/6] Deception vs cognitive controls (layer {args.layer})...")
    plot_deception_vs_controls(hs, dtypes, args.output_dir, layer=args.layer)

    print("[5/6] Similarity progression across layers...")
    plot_similarity_across_layers(hs, dtypes, args.output_dir)

    print(f"[6/6] Top discriminative dimensions (layer {args.layer})...")
    plot_top_dimensions(hs, dtypes, args.output_dir, layer=args.layer)

    print(f"\nAll figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
