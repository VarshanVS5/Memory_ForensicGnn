"""
FastAPI backend for the fancy React/Three.js frontend.

Wraps the exact same model/scaler/dataset logic app.py (Streamlit) uses --
this is a second frontend on the same backend, not a reimplementation. Keeps
to the core "pick a sample, get a verdict + attention graph" flow; CSV
upload and live memory-dump scanning stay on the Streamlit app for now.

Run with:
    uvicorn api:app --reload --port 8000
"""

import json
import os
import random

import pandas as pd
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from data_loader import load_dataset, FEATURE_GROUPS
from graph_builder import fit_scaler, load_scaler, build_graph_from_row, NODE_TYPES, EDGE_TEMPLATE, _max_group_len
from model import build_model

_REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
OUTPUT_DIR = os.path.join(_REPO_ROOT, "outputs")
PROCESSED_DIR = os.path.join(_REPO_ROOT, "data", "processed")
ALL_FEATURES = [c for group in FEATURE_GROUPS.values() for c in group]

NODE_LABELS = {
    "pslist": "Running processes",
    "dlllist": "Loaded DLLs",
    "handles": "Open handles",
    "ldrmodules": "Module loader records",
    "malfind": "Injected memory",
    "psxview": "Process visibility",
    "modules": "Kernel modules",
    "svcscan": "Windows services",
    "callbacks": "Kernel callbacks",
}

app = FastAPI(title="Memory Forensics GNN API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local dev / demo only -- tighten before any real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------ loading

def _load_splits():
    train_path = os.path.join(PROCESSED_DIR, "train.csv")
    test_path = os.path.join(PROCESSED_DIR, "test.csv")
    if os.path.exists(train_path) and os.path.exists(test_path):
        return pd.read_csv(train_path), pd.read_csv(test_path)
    df = load_dataset()
    return train_test_split(df, test_size=0.2, stratify=df["binary_label"], random_state=42)


def _load_scaler():
    scaler_path = os.path.join(OUTPUT_DIR, "scaler.json")
    if os.path.exists(scaler_path):
        return load_scaler(scaler_path)
    train_df, _ = _load_splits()
    return fit_scaler(train_df)


def _load_gnn(name):
    scaler, cols = _load_scaler()
    m = build_model(name, _max_group_len(), hidden_dim=64)
    ckpt = os.path.join(OUTPUT_DIR, f"{name}_model.pt")
    if not os.path.exists(ckpt):
        return None
    m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    m.eval()
    return m


# Loaded once at process startup -- this is a small demo API, not a
# production service with per-request model loading concerns.
_TRAIN_DF, _TEST_DF = _load_splits()
_FULL_DF = load_dataset()
_SCALER, _COLS = _load_scaler()
_MODELS = {"sage": _load_gnn("sage"), "gat": _load_gnn("gat")}
_RF = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
_RF.fit(_TRAIN_DF[ALL_FEATURES], _TRAIN_DF["binary_label"])


def _nan_to_none(obj):
    """Standard JSON has no NaN literal -- the generalization summary has a
    few (fpr/auc are undefined when the held-out set has no benign rows).
    Recursively swap them for null so the response is valid JSON."""
    if isinstance(obj, float) and obj != obj:  # NaN != NaN
        return None
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_none(v) for v in obj]
    return obj


def _generalization_summary():
    path = os.path.join(OUTPUT_DIR, "generalization_summary.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return _nan_to_none(json.load(f))


def _predict_row(row, model_name):
    model = _MODELS.get(model_name)
    if model is None:
        raise HTTPException(404, f"No trained {model_name} checkpoint. Run train.py --model {model_name} first.")

    if "binary_label" not in row.index:
        row = row.copy()
        row["binary_label"] = -1  # placeholder; unused for prediction itself
    graph = build_graph_from_row(row, _SCALER, _COLS)
    batch = torch.zeros(graph.num_nodes, dtype=torch.long)
    is_gat = model_name == "gat"
    with torch.no_grad():
        if is_gat:
            out, attn_layers = model(graph.x, graph.edge_index, batch, return_attention=True)
        else:
            out = model(graph.x, graph.edge_index, batch)
            attn_layers = None
        prob = F.softmax(out, dim=1)[0, 1].item()

    edges = []
    if attn_layers is not None:
        ei, alpha = attn_layers[-1]
        alpha = alpha.mean(dim=1)
        scores = {}
        for k in range(ei.shape[1]):
            src, dst = ei[0, k].item(), ei[1, k].item()
            if src == dst:
                continue
            key = tuple(sorted((NODE_TYPES[src], NODE_TYPES[dst])))
            scores[key] = max(scores.get(key, 0.0), alpha[k].item())
        edges = [{"source": a, "target": b, "weight": w} for (a, b), w in scores.items()]

    rf_prob = _RF.predict_proba(row[ALL_FEATURES].to_frame().T)[0, 1]

    return {
        "prediction": "MALICIOUS" if prob >= 0.5 else "BENIGN",
        "confidence": prob,
        "baseline": {
            "prediction": "MALICIOUS" if rf_prob >= 0.5 else "BENIGN",
            "confidence": float(rf_prob),
        },
        "attentionEdges": edges,
        "features": {k: float(row[k]) for k in ALL_FEATURES},
    }


# ------------------------------------------------------------------- schema

class PredictRequest(BaseModel):
    model: str = "gat"
    features: dict


# ---------------------------------------------------------------- endpoints

@app.get("/api/graph-schema")
def graph_schema():
    """The fixed 9-node artifact graph structure -- static, used to lay out the 3D scene."""
    return {
        "nodes": [{"id": n, "label": NODE_LABELS[n]} for n in NODE_TYPES],
        "edges": [{"source": a, "target": b} for a, b in EDGE_TEMPLATE],
    }


@app.get("/api/dataset-stats")
def dataset_stats():
    return {
        "totalSamples": len(_FULL_DF),
        "benign": int((_FULL_DF["binary_label"] == 0).sum()),
        "malicious": int((_FULL_DF["binary_label"] == 1).sum()),
        "families": int(_FULL_DF.loc[_FULL_DF["binary_label"] == 1, "family"].nunique()),
    }


@app.get("/api/generalization")
def generalization():
    summary = _generalization_summary()
    if summary is None:
        raise HTTPException(404, "Run generalization_test.py first.")
    return summary


@app.get("/api/sample/random")
def random_sample(model: str = "gat"):
    idx = random.randrange(len(_TEST_DF))
    row = _TEST_DF.iloc[idx]
    result = _predict_row(row, model)
    result["index"] = idx
    result["trueLabel"] = "MALICIOUS" if int(row["binary_label"]) == 1 else "BENIGN"
    return result


@app.get("/api/sample/{index}")
def sample_by_index(index: int, model: str = "gat"):
    if index < 0 or index >= len(_TEST_DF):
        raise HTTPException(400, f"index must be between 0 and {len(_TEST_DF) - 1}")
    row = _TEST_DF.iloc[index]
    result = _predict_row(row, model)
    result["index"] = index
    result["trueLabel"] = "MALICIOUS" if int(row["binary_label"]) == 1 else "BENIGN"
    return result


@app.post("/api/predict")
def predict(req: PredictRequest):
    missing = [c for c in ALL_FEATURES if c not in req.features]
    if missing:
        raise HTTPException(400, f"Missing features: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    row = pd.Series(req.features)
    return _predict_row(row, req.model)
