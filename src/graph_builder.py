"""
Graph construction for the Memory Forensics GNN project.

CIC-MalMem-2022 gives one AGGREGATE feature row per memory dump (i.e. per
whole-system snapshot), not per-process. To get real graph structure without
raw memory images, we build one graph PER SAMPLE where:

  - each Volatility plugin group (pslist, dlllist, handles, ldrmodules,
    malfind, psxview, modules, svcscan, callbacks) becomes an "artifact node"
  - node features = that group's columns for this sample (z-scored)
  - edges connect artifact nodes that are semantically related in a real
    memory-forensics investigation graph, e.g.:
        pslist  <-> dlllist     (process loads DLLs)
        pslist  <-> handles     (process owns handles/threads)
        dlllist <-> ldrmodules  (loader-list consistency check -> hiding detector)
        pslist  <-> malfind     (injected/RWX memory in a process)
        pslist  <-> psxview     (cross-view process hiding detector)
        svcscan <-> pslist      (services spawn processes)
        callbacks <-> modules   (kernel callbacks registered by modules)
  - the whole graph gets ONE label (benign/malicious) -> graph classification

If/when you move to RAW memory dumps + Volatility JSON output instead of the
CIC-MalMem-2022 CSV, swap build_graph_from_row() for a real per-process graph
(process nodes with parent/child + injection edges) using the same GNN model
below -- that is the stronger version of this project and worth doing if you
have time, since it captures actual structural relationships instead of
per-sample summary correlations.
"""

import torch
import numpy as np
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler

from data_loader import FEATURE_GROUPS

NODE_TYPES = list(FEATURE_GROUPS.keys())  # fixed node ordering
NODE_TYPE_IDX = {name: i for i, name in enumerate(NODE_TYPES)}

# Undirected artifact-relationship edges (investigation-motivated, see docstring)
EDGE_TEMPLATE = [
    ("pslist", "dlllist"),
    ("pslist", "handles"),
    ("pslist", "malfind"),
    ("pslist", "psxview"),
    ("dlllist", "ldrmodules"),
    ("handles", "ldrmodules"),
    ("svcscan", "pslist"),
    ("svcscan", "modules"),
    ("callbacks", "modules"),
    ("malfind", "ldrmodules"),
    ("psxview", "ldrmodules"),
]


def _max_group_len():
    return max(len(v) for v in FEATURE_GROUPS.values())


def fit_scaler(df):
    """Fit one StandardScaler over all Volatility feature columns (train split only)."""
    all_cols = [c for group in FEATURE_GROUPS.values() for c in group]
    scaler = StandardScaler()
    scaler.fit(df[all_cols].values)
    return scaler, all_cols


def save_scaler(scaler, cols, path):
    """Persist a fitted scaler's stats so inference-time code (e.g. a deployed
    app without access to the full training set) can reproduce the EXACT
    scaling the model was trained with, instead of re-fitting on whatever
    smaller sample happens to be available -- a silent mismatch that would
    corrupt every prediction without erroring."""
    import json
    with open(path, "w") as f:
        json.dump({"columns": cols, "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}, f)


def load_scaler(path):
    """Load a scaler persisted by save_scaler(). Returns (scaler, cols)."""
    import json
    with open(path) as f:
        data = json.load(f)
    scaler = StandardScaler()
    scaler.mean_ = np.array(data["mean"])
    scaler.scale_ = np.array(data["scale"])
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(data["columns"])
    return scaler, data["columns"]


def build_edge_index():
    src, dst = [], []
    for a, b in EDGE_TEMPLATE:
        i, j = NODE_TYPE_IDX[a], NODE_TYPE_IDX[b]
        src += [i, j]
        dst += [j, i]  # undirected
    return torch.tensor([src, dst], dtype=torch.long)


_EDGE_INDEX_CACHE = None


def build_graph_from_scaled_array(scaled_row, label, pad_len, col_to_idx):
    """Build one PyG Data object from an already-scaled numpy row."""
    global _EDGE_INDEX_CACHE
    node_feats = []
    for ntype in NODE_TYPES:
        cols = FEATURE_GROUPS[ntype]
        idxs = [col_to_idx[c] for c in cols]
        vec = scaled_row[idxs].astype(np.float32)
        if len(vec) < pad_len:
            vec = np.pad(vec, (0, pad_len - len(vec)), constant_values=0.0)
        node_feats.append(vec)

    x = torch.tensor(np.stack(node_feats), dtype=torch.float)
    if _EDGE_INDEX_CACHE is None:
        _EDGE_INDEX_CACHE = build_edge_index()
    y = torch.tensor([int(label)], dtype=torch.long)
    return Data(x=x, edge_index=_EDGE_INDEX_CACHE, y=y)


def build_graph_from_row(row, scaler, all_cols, pad_len=None):
    """Single-row convenience wrapper (kept for backward compatibility / demos)."""
    if pad_len is None:
        pad_len = _max_group_len()
    scaled_row = scaler.transform([row[all_cols].values])[0]
    col_to_idx = {c: i for i, c in enumerate(all_cols)}
    return build_graph_from_scaled_array(scaled_row, row["binary_label"], pad_len, col_to_idx)


def build_dataset(df, scaler=None, all_cols=None, label_col="binary_label"):
    """Convert an entire dataframe into a list of PyG graphs (vectorized).

    label_col defaults to the binary Benign/Malicious target; pass
    label_col="family_label" (see data_loader.FAMILY_LABEL_MAP) for the
    multiclass Trojan/Spyware/Ransomware/Benign task.
    """
    if scaler is None:
        scaler, all_cols = fit_scaler(df)

    pad_len = _max_group_len()
    col_to_idx = {c: i for i, c in enumerate(all_cols)}

    # Scale ALL rows at once (fast), then slice per-row (still python loop,
    # but no repeated per-row scaler.transform() calls -- that was the
    # bottleneck on 58k rows)
    scaled_matrix = scaler.transform(df[all_cols].values)
    labels = df[label_col].values

    graphs = [
        build_graph_from_scaled_array(scaled_matrix[i], labels[i], pad_len, col_to_idx)
        for i in range(len(df))
    ]
    return graphs, scaler, all_cols


if __name__ == "__main__":
    from data_loader import load_dataset

    df = load_dataset()
    graphs, scaler, cols = build_dataset(df.head(50))
    g = graphs[0]
    print("Nodes:", g.num_nodes, "| Node feat dim:", g.x.shape[1], "| Edges:", g.edge_index.shape[1])
    print("Label:", g.y.item())
    print("Node type order:", NODE_TYPES)
