"""
graph/build_graph_v2.py
"""

import sys
import json
import importlib.util
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from geopy.distance import geodesic
from pathlib import Path

ROOT       = Path(__file__).parent.parent
NODES_PY   = ROOT / "data" / "nodes.py"
GRAPH_DIR  = ROOT / "graph"
OUTPUTS    = ROOT / "outputs"
GRAPH_DIR.mkdir(exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)

DISTANCE_THRESHOLD_KM = 300
DISTANCE_DECAY_KM     = 200

# Known major market hubs that drive conflict diffusion
MARKET_HUBS = {
    "mopti":    (14.49, -4.20),
    "agadez":   (16.97,  7.99),
    "zinder":   (13.80,  8.99),
    "gao":      (16.27, -0.04),
    "ndjamena": (12.11, 15.04),
    "ouagadougou": (12.37, -1.53),
}


def load_nodes() -> list:
    spec = importlib.util.spec_from_file_location("nodes", NODES_PY)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    nodes = mod.NODES
    print(f"✅ Loaded {len(nodes)} nodes")
    return nodes


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
    print(f"✅ Distance matrix: shape={D.shape}  "
          f"min={D[D>0].min():.1f}km  max={D.max():.1f}km")
    return D


def try_build_road_graph(nodes: list):
    """
    Attempt to download OSMnx road network for the Sahel bounding box.
    Returns (road_graph, node_to_osm_map) or (None, None) on failure.
    """
    try:
        import osmnx as ox
        print("\nDownloading OSMnx road network for Sahel...")
        print("(This may take 2-5 minutes on first run)")

        road_graph = ox.graph_from_bbox(
            north=23, south=10, east=16, west=-18,
            network_type="drive",
            custom_filter='["highway"~"trunk|primary|secondary"]',
        )
        print(f"✅ Road network: {len(road_graph.nodes)} nodes, "
              f"{len(road_graph.edges)} edges")

        # Snap each city node to nearest OSM node
        node_to_osm = {}
        for i, node in enumerate(nodes):
            osm_id = ox.nearest_nodes(
                road_graph, node["lon"], node["lat"]
            )
            node_to_osm[i] = osm_id

        return road_graph, node_to_osm

    except ImportError:
        print("⚠️  osmnx not installed. Run: pip install osmnx")
        print("   Falling back to distance-only weights.")
        return None, None
    except Exception as e:
        print(f"⚠️  OSMnx download failed: {e}")
        print("   Falling back to distance-only weights.")
        return None, None


def compute_road_distances(
    nodes: list,
    road_graph,
    node_to_osm: dict,
    D: np.ndarray,
) -> np.ndarray:
    """
    Compute road travel distance (km) between connected node pairs.
    Returns (N, N) matrix. Falls back to geographic distance if no road path.
    """
    N = len(nodes)
    R = np.full((N, N), np.inf, dtype=np.float32)
    np.fill_diagonal(R, 0)

    pairs_tried = 0
    pairs_found = 0

    for i in range(N):
        for j in range(i + 1, N):
            if D[i, j] >= DISTANCE_THRESHOLD_KM:
                continue
            try:
                road_km = nx.shortest_path_length(
                    road_graph,
                    node_to_osm[i],
                    node_to_osm[j],
                    weight="length"
                ) / 1000
                R[i, j] = road_km
                R[j, i] = road_km
                pairs_found += 1
            except nx.NetworkXNoPath:
                # No road path — use geographic distance as fallback
                R[i, j] = D[i, j] * 1.5   # penalize: road detour factor
                R[j, i] = D[i, j] * 1.5
            pairs_tried += 1

    print(f"✅ Road distances: {pairs_found}/{pairs_tried} pairs "
          f"found via road network")
    return R


def market_overlap_score(nodes: list, i: int, j: int) -> float:
    """
    Returns a score for how much two nodes share market/trade hub proximity.
    Nodes that both lie within 200km of the same major market
    have elevated conflict diffusion risk.
    """
    score = 0.0
    for hub, coords in MARKET_HUBS.items():
        di = geodesic((nodes[i]["lat"], nodes[i]["lon"]), coords).km
        dj = geodesic((nodes[j]["lat"], nodes[j]["lon"]), coords).km
        if di < 200 and dj < 200:
            score += 1.0 / (1 + (di + dj) / 200)
    return score


def build_adjacency_v2(
    nodes: list,
    D: np.ndarray,
    R: np.ndarray = None,
) -> np.ndarray:
    """
    Build composite edge weight matrix.

    W(i,j) = 0.40 * distance_score
           + 0.40 * road_score        (or distance_score if no road data)
           + 0.20 * market_score

    All components normalized to [0,1].
    """
    N = len(nodes)
    A = np.zeros((N, N), dtype=np.float32)

    has_road = R is not None

    for i in range(N):
        for j in range(i + 1, N):
            if D[i, j] >= DISTANCE_THRESHOLD_KM:
                continue

            dist_score   = float(np.exp(-D[i, j] / DISTANCE_DECAY_KM))
            market_score = float(market_overlap_score(nodes, i, j))
            market_score = min(market_score, 1.0)   # cap at 1

            if has_road and R[i, j] < np.inf:
                road_score = float(np.exp(-R[i, j] / (DISTANCE_DECAY_KM * 1.5)))
            else:
                road_score = dist_score   # fallback

            w = 0.40 * dist_score + 0.40 * road_score + 0.20 * market_score
            A[i, j] = w
            A[j, i] = w

    n_edges = int((A > 0).sum() / 2)
    print(f"✅ v2 adjacency: {n_edges} edges  "
          f"weight range [{A[A>0].min():.4f}, {A[A>0].max():.4f}]")
    return A


def normalize_adjacency(A: np.ndarray) -> np.ndarray:
    N          = A.shape[0]
    A_hat      = A + np.eye(N, dtype=np.float32)
    degree     = A_hat.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degree + 1e-8))
    return D_inv_sqrt @ A_hat @ D_inv_sqrt


def to_coo(A_norm: np.ndarray):
    rows, cols  = np.nonzero(A_norm)
    edge_index  = np.vstack([rows, cols]).astype(np.int64)
    edge_weight = A_norm[rows, cols].astype(np.float32)
    print(f"✅ COO: {edge_index.shape[1]} directed edges")
    return edge_index, edge_weight


def compare_v1_v2(nodes, A_v1, A_v2):
    """Print which edges changed most between v1 and v2."""
    N = len(nodes)
    records = []
    for i in range(N):
        for j in range(i + 1, N):
            if A_v1[i, j] > 0 or A_v2[i, j] > 0:
                records.append({
                    "node_i":   nodes[i]["id"],
                    "node_j":   nodes[j]["id"],
                    "w_v1":     round(float(A_v1[i, j]), 4),
                    "w_v2":     round(float(A_v2[i, j]), 4),
                    "delta":    round(float(A_v2[i, j] - A_v1[i, j]), 4),
                })

    df = pd.DataFrame(records).sort_values("delta", key=abs, ascending=False)

    print(f"\n  Top 10 edges changed most (v1 → v2):")
    print(f"  {'Edge':35s} {'v1':>8} {'v2':>8} {'delta':>8}")
    print(f"  {'─'*62}")
    for _, row in df.head(10).iterrows():
        edge = f"{row['node_i']} ↔ {row['node_j']}"
        direction = "⬆️ " if row["delta"] > 0 else "⬇️ "
        print(f"  {edge:35s} {row['w_v1']:8.4f} {row['w_v2']:8.4f} "
              f"{direction}{row['delta']:+.4f}")

    out_path = OUTPUTS / "graph_v1_vs_v2_comparison.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  ✅ Comparison saved → {out_path}")
    return df


def visualize_v2(nodes, A_v2, save_path=None):
    G = nx.Graph()
    for i, node in enumerate(nodes):
        G.add_node(i, **node)
    N = len(nodes)
    for i in range(N):
        for j in range(i + 1, N):
            if A_v2[i, j] > 0:
                G.add_edge(i, j, weight=float(A_v2[i, j]))

    pos    = {i: (n["lon"], n["lat"]) for i, n in enumerate(nodes)}
    labels = {i: n["id"] for i, n in enumerate(nodes)}
    country_colors = {
        "ML": "#E63946", "NI": "#F4A261", "UV": "#2A9D8F",
        "CD": "#457B9D", "MR": "#6A4C93",
    }
    country_names = {
        "ML": "Mali", "NI": "Niger", "UV": "Burkina Faso",
        "CD": "Chad",  "MR": "Mauritania",
    }
    node_colors  = [country_colors.get(nodes[i]["country"], "#888") for i in G.nodes()]
    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]

    fig, ax = plt.subplots(figsize=(15, 10))
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")
    nx.draw_networkx_edges(G, pos, ax=ax,
                           width=[1 + w * 5 for w in edge_weights],
                           edge_color=[mcolors.to_rgba("white", alpha=w * 0.6)
                                       for w in edge_weights])
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=220, linewidths=1.5, edgecolors="white")
    nx.draw_networkx_labels(G, pos, labels, ax=ax,
                            font_size=6.5, font_color="white", font_weight="bold")
    for code, color in country_colors.items():
        ax.plot([], [], "o", color=color, label=country_names[code], markersize=9)
    ax.legend(loc="lower left", facecolor="#0d0d1a", labelcolor="white", fontsize=9)
    ax.set_title("Sahel GNN — Road-weighted Graph (v2)", color="white", fontsize=13)
    ax.set_xlabel("Longitude", color="#aaa")
    ax.set_ylabel("Latitude",  color="#aaa")
    ax.tick_params(colors="#aaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✅ v2 graph visualization → {save_path}")
    else:
        plt.show()
    plt.close()


def build_graph_v2():
    print("=" * 60)
    print("  Sahel GNN — Graph Construction v2 (road-weighted)")
    print("=" * 60)

    nodes = load_nodes()
    D     = compute_distance_matrix(nodes)

    # Try road network
    road_graph, node_to_osm = try_build_road_graph(nodes)
    if road_graph is not None:
        R = compute_road_distances(nodes, road_graph, node_to_osm, D)
    else:
        R = None
        print("  Using distance-only weights (OSMnx unavailable)")

    # Build v2 adjacency
    A_v2   = build_adjacency_v2(nodes, D, R)
    A_norm = normalize_adjacency(A_v2)
    edge_index, edge_weight = to_coo(A_norm)

    # Load v1 for comparison
    v1_path = GRAPH_DIR / "adjacency_matrix.npy"
    if v1_path.exists():
        A_v1 = np.load(v1_path)
        compare_v1_v2(nodes, A_v1, A_v2)
    else:
        print("⚠️  v1 adjacency_matrix.npy not found — skipping comparison")

    # Save v2 files
    np.save(GRAPH_DIR / "adjacency_matrix_v2.npy",  A_norm)
    np.save(GRAPH_DIR / "edge_index_v2.npy",         edge_index)
    np.save(GRAPH_DIR / "edge_weight_v2.npy",        edge_weight)
    print(f"\n✅ Saved: adjacency_matrix_v2.npy, edge_index_v2.npy, edge_weight_v2.npy")

    viz_path = OUTPUTS / "graph_v2_visualization.png"
    visualize_v2(nodes, A_v2, save_path=viz_path)

    return A_v2, edge_index, edge_weight


if __name__ == "__main__":
    build_graph_v2()
