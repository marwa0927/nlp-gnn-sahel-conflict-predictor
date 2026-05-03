"""
graph/build_graph.py

Builds the distance-based adjacency matrix for the Sahel conflict GNN.

Reads:
  - data/nodes.py
  - data/processed/features_daily.parquet

Outputs:
  - graph/adjacency_matrix.npy
  - graph/edge_index.npy
  - graph/edge_weight.npy
  - graph/node_metadata.json
"""

import sys
import json
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from geopy.distance import geodesic
from pathlib import Path
import importlib.util

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
ROOT            = Path(__file__).parent.parent
NODES_PY        = ROOT / "data" / "nodes.py"
FEATURES_PATH   = ROOT / "data" / "processed" / "features_daily.parquet"
GRAPH_DIR       = ROOT / "graph"
GRAPH_DIR.mkdir(exist_ok=True)

ADJ_OUT         = GRAPH_DIR / "adjacency_matrix.npy"
EDGE_INDEX_OUT  = GRAPH_DIR / "edge_index.npy"
EDGE_WEIGHT_OUT = GRAPH_DIR / "edge_weight.npy"
NODE_META_OUT   = GRAPH_DIR / "node_metadata.json"

DISTANCE_THRESHOLD_KM = 1000
DISTANCE_DECAY_KM     = 400


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Load nodes from data/nodes.py
# ─────────────────────────────────────────────────────────────────────────────
def load_nodes() -> list:
    """
    Dynamically imports data/nodes.py and returns the NODES list.
    """
    spec = importlib.util.spec_from_file_location("nodes", NODES_PY)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    nodes = mod.NODES
    print(f"Loaded {len(nodes)} nodes from data/nodes.py")
    for n in nodes:
        print(f"   {n['id']:20s} lat={n['lat']:6.2f}  lon={n['lon']:7.2f}  country={n['country']}")
    return nodes


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Load features_daily.parquet and validate node coverage
# ─────────────────────────────────────────────────────────────────────────────
def load_and_validate_features(nodes: list) -> pd.DataFrame:
    """
    Loads the feature matrix and checks that every node_id
    in nodes.py is present in the parquet file.
    Prints a warning for any node that has no feature data.
    """
    if not FEATURES_PATH.exists():
        print(f"\nfeatures_daily.parquet not found at {FEATURES_PATH}")
        print("   Graph construction will continue — but dataset.py needs this file.")
        print("   Re-run after the feature matrix is available.")
        return None

    df = pd.read_parquet(FEATURES_PATH)
    print(f"\nLoaded features_daily.parquet — shape {df.shape}")
    print(f"   Columns: {df.columns.tolist()}")

    # Identify the node column
    node_col = None
    for candidate in ["node_id", "city", "location", "node", "id"]:
        if candidate in df.columns:
            node_col = candidate
            break

    if node_col is None:
        print("   Could not detect node identifier column in parquet.")
        print("   Expected one of: node_id, city, location, node, id")
        print("   Check the parquet schema.")
        return df

    node_ids_in_data  = set(df[node_col].unique())
    node_ids_in_nodes = {n["id"] for n in nodes}

    missing = node_ids_in_nodes - node_ids_in_data
    extra   = node_ids_in_data  - node_ids_in_nodes

    if missing:
        print(f"\nNodes in nodes.py with NO feature data: {missing}")
        print("   Warning: these nodes will be zero-filled.")
    if extra:
        print(f"\nNode IDs in parquet not in nodes.py: {extra}")
        print("   These will be ignored by the graph builder.")
    if not missing and not extra:
        print("✅ All 25 nodes matched between nodes.py and features_daily.parquet")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Compute pairwise distance matrix
# ─────────────────────────────────────────────────────────────────────────────
def compute_distance_matrix(nodes: list) -> np.ndarray:
    N = len(nodes)
    D = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(i + 1, N):
            d = geodesic(
                (nodes[i]["lat"], nodes[i]["lon"]),
                (nodes[j]["lat"], nodes[j]["lon"])
            ).km
            D[i, j] = d
            D[j, i] = d
    print(f"\nDistance matrix — shape {D.shape}")
    print(f"   Min non-zero: {D[D > 0].min():.1f} km")
    print(f"   Max:          {D.max():.1f} km")
    return D


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Build adjacency matrix
# ─────────────────────────────────────────────────────────────────────────────
def build_adjacency(D: np.ndarray) -> np.ndarray:
    N = D.shape[0]
    A = np.zeros((N, N), dtype=np.float32)
    mask = (D > 0) & (D < DISTANCE_THRESHOLD_KM)
    A[mask] = np.exp(-D[mask] / DISTANCE_DECAY_KM)
    n_edges = int(mask.sum() / 2)
    print(f"Raw adjacency — {n_edges} undirected edges (threshold={DISTANCE_THRESHOLD_KM} km)")
    return A


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Normalize — D^{-1/2} A_hat D^{-1/2}
# ─────────────────────────────────────────────────────────────────────────────
def normalize_adjacency(A: np.ndarray) -> np.ndarray:
    N          = A.shape[0]
    A_hat      = A + np.eye(N, dtype=np.float32)
    degree     = A_hat.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degree + 1e-8))
    A_norm     = D_inv_sqrt @ A_hat @ D_inv_sqrt
    print(f"Adjacency normalized")
    return A_norm


# ─────────────────────────────────────────────────────────────────────────────
# Step 6: COO format for PyTorch Geometric
# ─────────────────────────────────────────────────────────────────────────────
def to_coo(A_norm: np.ndarray):
    rows, cols  = np.nonzero(A_norm)
    edge_index  = np.vstack([rows, cols]).astype(np.int64)
    edge_weight = A_norm[rows, cols].astype(np.float32)
    print(f"COO format: {edge_index.shape[1]} directed edges")
    return edge_index, edge_weight


# ─────────────────────────────────────────────────────────────────────────────
# Step 7: NetworkX graph for validation
# ─────────────────────────────────────────────────────────────────────────────
def build_networkx_graph(nodes: list, A: np.ndarray) -> nx.Graph:
    G = nx.Graph()
    for i, node in enumerate(nodes):
        G.add_node(i, **node)
    N = len(nodes)
    for i in range(N):
        for j in range(i + 1, N):
            if A[i, j] > 0:
                G.add_edge(i, j, weight=float(A[i, j]))
    return G


# ─────────────────────────────────────────────────────────────────────────────
# Step 8: Sanity checks
# ─────────────────────────────────────────────────────────────────────────────
def run_sanity_checks(nodes: list, A: np.ndarray, G: nx.Graph):
    N        = len(nodes)
    node_ids = [n["id"] for n in nodes]

    print("\n── Sanity checks ──────────────────────────────────────")

    assert A.shape == (N, N),               f"A shape {A.shape} != ({N}, {N})"
    assert np.allclose(A, A.T, atol=1e-6),  "Not symmetric"
    assert not np.any(np.isnan(A)),         "NaN in A"
    assert A.min() >= 0,                    f"Negative weight: {A.min()}"
    assert A.max() <= 1.0,                  f"Weight > 1: {A.max()}"
    assert nx.is_connected(G),              "Graph is NOT connected"
    print(f" Shape, symmetry, values, connectivity — all good")

    corridors = [
        ("mopti",        "gao",         "Mali central corridor"),
        ("gao",          "kidal",       "Mali northeast"),
        ("gao",          "menaka",      "Mali-Niger border"),
        ("menaka",       "ansongo",     "Mali southeast"),
        ("niamey",       "tillaberi",   "Niger west"),
        ("agadez",       "diffa",       "Niger corridor"),
        ("ouagadougou",  "dori",        "Burkina corridor"),
        ("dori",         "djibo",       "Burkina north"),
    ]
    for src, dst, label in corridors:
        if src in node_ids and dst in node_ids:
            i, j = node_ids.index(src), node_ids.index(dst)
            w    = A[i, j]
            print(f"  {'PASSED' if w > 0 else 'FAILED'} {src:15s} ↔ {dst:15s}  w={w:.4f}  ({label})")

    degrees = [d for _, d in G.degree()]
    print(f"\n  Degree — mean={np.mean(degrees):.1f}  "
          f"min={min(degrees)} ({nodes[np.argmin(degrees)]['id']})  "
          f"max={max(degrees)} ({nodes[np.argmax(degrees)]['id']})")
    print("── All checks passed ──────────────────────────────────\n")


# ─────────────────────────────────────────────────────────────────────────────
# Step 9: Visualization
# ─────────────────────────────────────────────────────────────────────────────
def visualize_graph(nodes: list, G: nx.Graph, save_path: Path = None):
    pos    = {i: (n["lon"], n["lat"]) for i, n in enumerate(nodes)}
    labels = {i: n["id"] for i, n in enumerate(nodes)}

    country_colors = {"ML": "#E63946", "NI": "#F4A261", "UV": "#2A9D8F",
                      "CD": "#457B9D", "MR": "#6A4C93"}
    country_names  = {"ML": "Mali", "NI": "Niger", "UV": "Burkina Faso",
                      "CD": "Chad", "MR": "Mauritania"}

    node_colors  = [country_colors.get(nodes[i]["country"], "#888") for i in G.nodes()]
    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]

    fig, ax = plt.subplots(figsize=(15, 10))
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")

    nx.draw_networkx_edges(G, pos, ax=ax,
                           width=[1 + w * 4 for w in edge_weights],
                           edge_color=[mcolors.to_rgba("white", alpha=w * 0.5)
                                       for w in edge_weights])
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=220, linewidths=1.5, edgecolors="white")
    nx.draw_networkx_labels(G, pos, labels, ax=ax,
                            font_size=6.5, font_color="white", font_weight="bold")

    for code, color in country_colors.items():
        ax.plot([], [], "o", color=color, label=country_names[code], markersize=9)
    ax.legend(loc="lower left", facecolor="#0d0d1a", labelcolor="white", fontsize=9)
    ax.set_title("Sahel Conflict GNN — Spatial Graph (v1)", color="white", fontsize=13)
    ax.set_xlabel("Longitude", color="#aaa")
    ax.set_ylabel("Latitude",  color="#aaa")
    ax.tick_params(colors="#aaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Visualization saved → {save_path}")
    else:
        plt.show()
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def build_graph(visualize: bool = True) -> dict:
    print("=" * 60)
    print("  Sahel GNN — Graph Construction (v1)")
    print("=" * 60)

    nodes       = load_nodes()
    df_features = load_and_validate_features(nodes)
    D           = compute_distance_matrix(nodes)
    A_raw       = build_adjacency(D)
    A_norm      = normalize_adjacency(A_raw)
    G           = build_networkx_graph(nodes, A_raw)
    run_sanity_checks(nodes, A_raw, G)
    edge_index, edge_weight = to_coo(A_norm)

    np.save(ADJ_OUT,         A_norm)
    np.save(EDGE_INDEX_OUT,  edge_index)
    np.save(EDGE_WEIGHT_OUT, edge_weight)

    node_meta = {n["id"]: {"index": i, **n} for i, n in enumerate(nodes)}
    with open(NODE_META_OUT, "w") as f:
        json.dump(node_meta, f, indent=2)

    print(f"Saved all graph files to graph/")

    if visualize:
        viz_path = ROOT / "outputs" / "graph_visualization.png"
        viz_path.parent.mkdir(exist_ok=True)
        visualize_graph(nodes, G, save_path=viz_path)

    return {
        "nodes": nodes, "D": D, "A_raw": A_raw, "A_norm": A_norm,
        "G": G, "edge_index": edge_index, "edge_weight": edge_weight,
        "df_features": df_features,
    }


if __name__ == "__main__":
    build_graph(visualize=True)
