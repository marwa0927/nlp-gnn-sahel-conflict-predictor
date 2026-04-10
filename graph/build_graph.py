"""
graph/build_graph.py


Builds the distance-based adjacency matrix A for the Sahel conflict GNN.
- Nodes:  25 key cities (defined in data/nodes.csv)
- Edges:  pairs within 300km threshold, weighted by inverse distance
- Output: data/adjacency_matrix.npy  (normalized, with self-loops)
          data/edge_index.npy        (COO format for PyTorch Geometric)
          data/edge_weight.npy       (corresponding edge weights)

NOTE: OSMnx road-weighted edges are deferred to v2.
      This distance-based graph is the v1 baseline.
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from geopy.distance import geodesic
from pathlib import Path
import json

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR        = Path(__file__).parent.parent / "data"
NODES_CSV       = DATA_DIR / "nodes.csv"
ADJ_OUT         = DATA_DIR / "adjacency_matrix.npy"
EDGE_INDEX_OUT  = DATA_DIR / "edge_index.npy"
EDGE_WEIGHT_OUT = DATA_DIR / "edge_weight.npy"
NODE_META_OUT   = DATA_DIR / "node_metadata.json"

DISTANCE_THRESHOLD_KM = 700   # maximum distance for an edge to exist
DISTANCE_DECAY_KM     = 400   # decay constant for inverse-distance weighting


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Load nodes
# ─────────────────────────────────────────────────────────────────────────────
def load_nodes(path: Path) -> pd.DataFrame:
    """
    Load node list from CSV.
    Returns DataFrame indexed 0..N-1 with columns:
        id, name, lat, lon, country, admin_level
    """
    df = pd.read_csv(path)
    required = {"id", "name", "lat", "lon", "country"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"nodes.csv is missing columns: {missing}")

    print(f"✅ Loaded {len(df)} nodes from {path.name}")
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Compute pairwise distances
# ─────────────────────────────────────────────────────────────────────────────
def compute_distance_matrix(nodes: pd.DataFrame) -> np.ndarray:
    """
    Returns (N, N) matrix of great-circle distances in km.
    Diagonal is 0.
    """
    N = len(nodes)
    D = np.zeros((N, N), dtype=np.float32)
    coords = list(zip(nodes["lat"], nodes["lon"]))

    for i in range(N):
        for j in range(i + 1, N):
            d = geodesic(coords[i], coords[j]).km
            D[i, j] = d
            D[j, i] = d

    print(f"   Distance matrix computed — shape {D.shape}")
    print(f"   Min non-zero distance: {D[D > 0].min():.1f} km")
    print(f"   Max distance:          {D.max():.1f} km")
    return D


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Build raw adjacency matrix
# ─────────────────────────────────────────────────────────────────────────────
def build_adjacency(D: np.ndarray, threshold_km=700, decay_km=400):
    N = D.shape[0]
    A = np.zeros((N, N), dtype=np.float32)

    # Apply distance threshold
    mask = (D > 0) & (D < threshold_km)
    A[mask] = np.exp(-D[mask] / decay_km)

    # ENSURE CONNECTIVITY: Each node connects to at least its 2 closest neighbors
    for i in range(N):
        # Get indices of the 2 smallest non-zero distances
        closest_indices = np.argsort(D[i])[1:3] 
        for idx in closest_indices:
            if A[i, idx] == 0: # If not already added by threshold
                A[i, idx] = np.exp(-D[i, idx] / decay_km)
                A[idx, i] = A[i, idx]

    return A


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Symmetric normalization + self-loops
# ─────────────────────────────────────────────────────────────────────────────
def normalize_adjacency(A: np.ndarray) -> np.ndarray:
    """
    Standard GCN normalization: D^{-1/2} * A_hat * D^{-1/2}
    where A_hat = A + I  (self-loops).

    This ensures each node aggregates its own features alongside neighbors,
    and degree differences don't dominate the message passing.
    """
    N = A.shape[0]
    A_hat = A + np.eye(N, dtype=np.float32)            # add self-loops

    degree = A_hat.sum(axis=1)                          # (N,)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degree + 1e-8)) # avoid div-by-zero

    A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt
    print(f"✅ Adjacency normalized (D^-½ A_hat D^-½)")
    return A_norm


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Convert to COO format for PyTorch Geometric
# ─────────────────────────────────────────────────────────────────────────────
def to_coo(A_norm: np.ndarray):
    """
    Convert dense adjacency matrix to COO (edge_index, edge_weight).
    edge_index : (2, E) int64  — source and target node indices
    edge_weight: (E,)  float32 — corresponding edge weights
    """
    rows, cols = np.nonzero(A_norm)
    weights = A_norm[rows, cols]

    edge_index  = np.vstack([rows, cols]).astype(np.int64)
    edge_weight = weights.astype(np.float32)

    print(f"COO format: {edge_index.shape[1]} directed edges (includes self-loops)")
    return edge_index, edge_weight


# ─────────────────────────────────────────────────────────────────────────────
# Step 6: Build NetworkX graph for validation & visualization
# ─────────────────────────────────────────────────────────────────────────────
def build_networkx_graph(nodes: pd.DataFrame, A: np.ndarray) -> nx.Graph:
    """
    Build a NetworkX graph from the raw (un-normalized) adjacency matrix.
    Used for sanity checking and visualization only.
    """
    G = nx.Graph()

    for i, row in nodes.iterrows():
        G.add_node(
            i,
            id=row["id"],
            name=row["name"],
            lat=row["lat"],
            lon=row["lon"],
            country=row["country"],
        )

    N = len(nodes)
    for i in range(N):
        for j in range(i + 1, N):
            if A[i, j] > 0:
                G.add_edge(i, j, weight=float(A[i, j]))

    return G


# ─────────────────────────────────────────────────────────────────────────────
# Step 7: Sanity checks
# ─────────────────────────────────────────────────────────────────────────────
def run_sanity_checks(nodes: pd.DataFrame, A: np.ndarray, G: nx.Graph):
    """
    Validate the graph structure before saving.
    Raises AssertionError if any check fails.
    """
    N = len(nodes)
    node_ids = nodes["id"].tolist()

    print("\n── Sanity checks ──────────────────────────────────────")

    # 1. Shape
    assert A.shape == (N, N), f"A shape {A.shape} != ({N}, {N})"
    print(f"Shape: {A.shape}")

    # 2. Symmetry
    assert np.allclose(A, A.T, atol=1e-6), "Adjacency matrix is not symmetric"
    print(f"Symmetric")

    # 3. No NaNs / Infs
    assert not np.any(np.isnan(A)), "NaN found in A"
    assert not np.any(np.isinf(A)), "Inf found in A"
    print(f"No NaN / Inf")

    # 4. Values in [0, 1]
    assert A.min() >= 0.0, f"Negative weight found: {A.min()}"
    assert A.max() <= 1.0, f"Weight > 1 found: {A.max()}"
    print(f"Weights in [0, 1]")

    # 5. Connectivity — graph should be connected
    assert nx.is_connected(G), "Graph is NOT connected — some nodes are isolated"
    print(f" Graph is connected")

    # 6. Key corridor checks — known conflict corridors must be edges
    corridors = [
        ("mopti",    "gao",       "Mali central corridor"),
        ("gao",      "kidal",     "Mali northeast corridor"),
        ("agadez",   "diffa",     "Niger corridor"),
        ("ouagadougou", "dori",   "Burkina corridor"),
    ]
    for src, dst, label in corridors:
        if src in node_ids and dst in node_ids:
            i, j = node_ids.index(src), node_ids.index(dst)
            assert A[i, j] > 0, f"Expected edge {src}↔{dst} ({label}) but got 0"
            print(f"Edge {src} ↔ {dst} ({label}): weight={A[i,j]:.4f}")

    # 7. Degree stats
    degrees = np.array([d for _, d in G.degree()])
    print(f"\n  Node degree stats:")
    print(f"    Mean:  {degrees.mean():.1f}")
    print(f"    Min:   {degrees.min()} ({nodes.iloc[degrees.argmin()]['name']})")
    print(f"    Max:   {degrees.max()} ({nodes.iloc[degrees.argmax()]['name']})")

    print("── All checks passed ──────────────────────────────────\n")


# ─────────────────────────────────────────────────────────────────────────────
# Step 8: Visualization
# ─────────────────────────────────────────────────────────────────────────────
def visualize_graph(nodes: pd.DataFrame, G: nx.Graph, save_path: Path = None):
    """
    Draw the graph using lat/lon as node positions.
    Color nodes by country; edge thickness by weight.
    """
    # Position: lon → x, lat → y
    pos = {i: (row["lon"], row["lat"]) for i, row in nodes.iterrows()}
    labels = {i: row["name"] for i, row in nodes.iterrows()}

    country_colors = {
        "Mali":         "#E63946",
        "Niger":        "#F4A261",
        "Burkina Faso": "#2A9D8F",
        "Chad":         "#457B9D",
        "Mauritania":   "#6A4C93",
    }
    node_colors = [country_colors.get(nodes.iloc[i]["country"], "#888") for i in G.nodes()]

    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
    edge_widths  = [1 + w * 4 for w in edge_weights]

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        width=edge_widths,
        edge_color=[mcolors.to_rgba("white", alpha=w * 0.6) for w in edge_weights],
    )
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=200,
        linewidths=1.5,
        edgecolors="white",
    )
    nx.draw_networkx_labels(
        G, pos, labels, ax=ax,
        font_size=7, font_color="white", font_weight="bold",
    )

    # Legend
    for country, color in country_colors.items():
        ax.plot([], [], "o", color=color, label=country, markersize=8)
    ax.legend(loc="lower left", facecolor="#0d0d1a", labelcolor="white", fontsize=9)

    ax.set_title("Sahel Conflict GNN — Node Graph (v1 distance-based)",
                 color="white", fontsize=13, pad=15)
    ax.set_xlabel("Longitude", color="#aaa", fontsize=9)
    ax.set_ylabel("Latitude",  color="#aaa", fontsize=9)
    ax.tick_params(colors="#aaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Graph visualization saved → {save_path}")
    else:
        plt.show()

    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def build_graph(visualize: bool = True) -> dict:
    """
    Full pipeline: load nodes → distances → adjacency → normalize → save.
    Returns dict with key arrays for downstream use.
    """
    print("=" * 60)
    print("  Sahel GNN — Graph Construction (v1)")
    print("=" * 60)

    # Load
    nodes = load_nodes(NODES_CSV)

    # Distance matrix
    D = compute_distance_matrix(nodes)

    # Raw adjacency
    A_raw = build_adjacency(D)

    # Normalized adjacency
    A_norm = normalize_adjacency(A_raw)

    # NetworkX graph (from raw, for interpretability)
    G = build_networkx_graph(nodes, A_raw)

    # Sanity checks
    run_sanity_checks(nodes, A_raw, G)

    # COO format
    edge_index, edge_weight = to_coo(A_norm)

    # Save
    np.save(ADJ_OUT,         A_norm)
    np.save(EDGE_INDEX_OUT,  edge_index)
    np.save(EDGE_WEIGHT_OUT, edge_weight)

    # Save node metadata as JSON (node_id → index mapping)
    node_meta = {
        row["id"]: {
            "index":   int(i),
            "name":    row["name"],
            "lat":     float(row["lat"]),
            "lon":     float(row["lon"]),
            "country": row["country"],
        }
        for i, row in nodes.iterrows()
    }
    with open(NODE_META_OUT, "w") as f:
        json.dump(node_meta, f, indent=2)

    print(f"Saved: {ADJ_OUT.name}")
    print(f"Saved: {EDGE_INDEX_OUT.name}")
    print(f"Saved: {EDGE_WEIGHT_OUT.name}")
    print(f"Saved: {NODE_META_OUT.name}")

    # Visualization
    if visualize:
        viz_path = DATA_DIR.parent / "notebooks" / "graph_visualization.png"
        visualize_graph(nodes, G, save_path=viz_path)

    print("\nGraph construction complete.")
    return {
        "nodes":       nodes,
        "D":           D,
        "A_raw":       A_raw,
        "A_norm":      A_norm,
        "G":           G,
        "edge_index":  edge_index,
        "edge_weight": edge_weight,
    }


if __name__ == "__main__":
    build_graph(visualize=True)
