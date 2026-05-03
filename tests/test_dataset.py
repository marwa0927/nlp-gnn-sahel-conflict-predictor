"""
tests/test_dataset.py
=====================
Pytest suite for graph, dataset, and model validation.
Works with real data from dataset_daily.parquet + nodes.py.

Run from project root:
    pytest tests/ -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import importlib.util
import numpy as np
import torch

ROOT       = Path(__file__).parent.parent
NODES_PY   = ROOT / "data" / "nodes.py"
GRAPH_DIR  = ROOT / "graph"
FEATURES_PATH = ROOT / "data" / "processed" / "dataset_daily.parquet"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def import_nodes():
    spec = importlib.util.spec_from_file_location("nodes", NODES_PY)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.NODES


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def nodes():
    return import_nodes()


@pytest.fixture(scope="session")
def graph_files():
    """Ensure graph files exist — run build_graph.py first if missing."""
    adj  = GRAPH_DIR / "adjacency_matrix.npy"
    ei   = GRAPH_DIR / "edge_index.npy"
    ew   = GRAPH_DIR / "edge_weight.npy"
    meta = GRAPH_DIR / "node_metadata.json"
    if not adj.exists():
        pytest.skip("Graph files not found. Run: python graph/build_graph.py")
    return {"adj": adj, "edge_index": ei, "edge_weight": ew, "meta": meta}


@pytest.fixture(scope="session")
def dataset(nodes, graph_files):
    if not FEATURES_PATH.exists():
        pytest.skip("dataset_daily.parquet not found.")
    from model.dataset import SahelConflictDataset
    return SahelConflictDataset(split=None)


@pytest.fixture(scope="session")
def train_dataset(nodes, graph_files):
    if not FEATURES_PATH.exists():
        pytest.skip("dataset_daily.parquet not found.")
    from model.dataset import SahelConflictDataset
    return SahelConflictDataset(split="train")


@pytest.fixture(scope="session")
def model(dataset):
    from model.gcn import build_model
    return build_model(
        window=dataset.window,
        n_features=dataset.F,
    )


# ─────────────────────────────────────────────────────────────────────────────
# nodes.py tests
# ─────────────────────────────────────────────────────────────────────────────
class TestNodesFile:
    def test_nodes_loaded(self, nodes):
        assert len(nodes) == 25, f"Expected 25 nodes, got {len(nodes)}"

    def test_nodes_have_required_fields(self, nodes):
        for n in nodes:
            assert "id"      in n, f"Node missing 'id': {n}"
            assert "lat"     in n, f"Node missing 'lat': {n}"
            assert "lon"     in n, f"Node missing 'lon': {n}"
            assert "country" in n, f"Node missing 'country': {n}"

    def test_node_ids_unique(self, nodes):
        ids = [n["id"] for n in nodes]
        assert len(ids) == len(set(ids)), "Duplicate node IDs found"

    def test_lat_lon_in_sahel_bounds(self, nodes):
        for n in nodes:
            assert 10 <= n["lat"] <= 25, \
                f"{n['id']} lat={n['lat']} outside Sahel bounds"
            assert -16 <= n["lon"] <= 24, \
                f"{n['id']} lon={n['lon']} outside Sahel bounds"

    def test_countries_valid(self, nodes):
        valid = {"ML", "NI", "UV", "CD", "MR"}
        for n in nodes:
            assert n["country"] in valid, \
                f"{n['id']} has unknown country code: {n['country']}"

    def test_expected_cities_present(self, nodes):
        ids = {n["id"] for n in nodes}
        required = {"bamako", "mopti", "gao", "kidal", "niamey",
                    "agadez", "ouagadougou", "ndjamena", "nouakchott"}
        missing = required - ids
        assert not missing, f"Missing expected cities: {missing}"


# ─────────────────────────────────────────────────────────────────────────────
# Graph / Adjacency matrix tests
# ─────────────────────────────────────────────────────────────────────────────
class TestAdjacencyMatrix:
    def test_shape(self, nodes, graph_files):
        A = np.load(graph_files["adj"])
        N = len(nodes)
        assert A.shape == (N, N), f"A.shape={A.shape}, expected ({N},{N})"

    def test_no_nans(self, graph_files):
        A = np.load(graph_files["adj"])
        assert not np.any(np.isnan(A)), "NaN in adjacency matrix"

    def test_non_negative(self, graph_files):
        A = np.load(graph_files["adj"])
        assert A.min() >= 0, f"Negative weight: {A.min()}"

    def test_edge_index_shape(self, graph_files):
        ei = np.load(graph_files["edge_index"])
        assert ei.shape[0] == 2, "edge_index must have 2 rows"

    def test_edge_weight_matches_index(self, graph_files):
        ei = np.load(graph_files["edge_index"])
        ew = np.load(graph_files["edge_weight"])
        assert ei.shape[1] == ew.shape[0], \
            f"edge_index has {ei.shape[1]} entries, edge_weight has {ew.shape[0]}"

    def test_key_corridors_connected(self, nodes, graph_files):
        A        = np.load(graph_files["adj"])
        node_ids = [n["id"] for n in nodes]
        corridors = [
            ("mopti",       "gao"),
            ("gao",         "menaka"),
            ("niamey",      "tillaberi"),
            ("ouagadougou", "dori"),
        ]
        for src, dst in corridors:
            if src in node_ids and dst in node_ids:
                i, j = node_ids.index(src), node_ids.index(dst)
                assert A[i, j] > 0, \
                    f"Expected edge {src}↔{dst} but weight is 0"


# ─────────────────────────────────────────────────────────────────────────────
# Dataset tests
# ─────────────────────────────────────────────────────────────────────────────
class TestSahelConflictDataset:
    def test_dataset_length_positive(self, dataset):
        assert len(dataset) > 0, "Dataset has 0 samples"

    def test_sample_x_shape(self, dataset):
        s = dataset.get(0)
        expected = (dataset.N, dataset.window * dataset.F)
        assert s.x.shape == expected, \
            f"x.shape={tuple(s.x.shape)}, expected {expected}"

    def test_sample_y_shape(self, dataset):
        s = dataset.get(0)
        assert s.y.shape == (dataset.N, 2), \
            f"y.shape={tuple(s.y.shape)}, expected ({dataset.N}, 2)"

    def test_edge_index_dtype(self, dataset):
        s = dataset.get(0)
        assert s.edge_index.dtype == torch.long, "edge_index must be long"

    def test_no_nan_in_x(self, dataset):
        s = dataset.get(0)
        assert not torch.any(torch.isnan(s.x)), "NaN in sample x"

    def test_no_nan_in_y(self, dataset):
        s = dataset.get(0)
        assert not torch.any(torch.isnan(s.y)), "NaN in sample y"

    def test_y_values_binary(self, dataset):
        s = dataset.get(0)
        assert s.y.min() >= 0.0 and s.y.max() <= 1.0, \
            f"y out of [0,1]: min={s.y.min()}, max={s.y.max()}"

    def test_temporal_splits_no_overlap(self, nodes, graph_files):
        if not FEATURES_PATH.exists():
            pytest.skip("Parquet not available")
        from model.dataset import SahelConflictDataset
        train_ds = SahelConflictDataset(split="train")
        val_ds   = SahelConflictDataset(split="val")
        test_ds  = SahelConflictDataset(split="test")
        assert not set(train_ds.sample_indices) & set(val_ds.sample_indices),  "Train/val overlap"
        assert not set(val_ds.sample_indices)   & set(test_ds.sample_indices), "Val/test overlap"
        assert not set(train_ds.sample_indices) & set(test_ds.sample_indices), "Train/test overlap"

    def test_consecutive_samples_differ(self, dataset):
        if len(dataset) < 2:
            pytest.skip("Not enough samples")
        s0 = dataset.get(0)
        s1 = dataset.get(1)
        assert not torch.equal(s0.x, s1.x), \
            "Consecutive samples have identical x — sliding window bug"

    def test_node_count_matches_nodes_py(self, dataset, nodes):
        assert dataset.N == len(nodes), \
            f"dataset.N={dataset.N} but nodes.py has {len(nodes)} nodes"


# ─────────────────────────────────────────────────────────────────────────────
# Model tests
# ─────────────────────────────────────────────────────────────────────────────
class TestConflictGCN:
    def test_output_shape(self, model, dataset):
        s = dataset.get(0)
        model.eval()
        with torch.no_grad():
            out = model(s.x, s.edge_index, s.edge_attr)
        assert out.shape == (dataset.N, 2), \
            f"Output shape {out.shape}, expected ({dataset.N}, 2)"

    def test_output_in_0_1(self, model, dataset):
        s = dataset.get(0)
        model.eval()
        with torch.no_grad():
            out = model(s.x, s.edge_index, s.edge_attr)
        assert out.min() >= 0.0 and out.max() <= 1.0, \
            f"Output out of [0,1]"

    def test_no_nan_output(self, model, dataset):
        s = dataset.get(0)
        model.eval()
        with torch.no_grad():
            out = model(s.x, s.edge_index, s.edge_attr)
        assert not torch.any(torch.isnan(out)), "NaN in model output"

    def test_backward_pass(self, model, dataset):
        s         = dataset.get(0)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.BCEWithLogitsLoss()
        model.train()
        logits = model.forward_logits(s.x, s.edge_index, s.edge_attr)
        loss   = criterion(logits, s.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        assert loss.item() > 0,          "Loss is 0"
        assert not torch.isnan(loss),    "Loss is NaN"

    def test_parameter_count(self, model):
        n = model.count_parameters()
        assert 1_000 < n < 1_000_000, f"Unexpected param count: {n}"
