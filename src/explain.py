"""
Interpretability: loads the GAT model trained by train.py (--model gat),
then for a handful of test samples shows which artifact-node edges got the
most attention -- i.e. "the model flagged this sample mainly because of
malfind <-> ldrmodules disagreement", etc. Good for a mentor demo / report
figure.

Uses the exact same held-out test split that train.py reported metrics on
(loaded from data/processed/test.csv if present), so the explanations line
up with the numbers you already reported -- not a freshly-shuffled split.

Requires a saved checkpoint: run `python train.py --model gat` first.
"""

import os
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from data_loader import load_dataset
from graph_builder import build_dataset, fit_scaler, NODE_TYPES
from model import GATClassifier

_REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
OUTPUT_DIR = os.path.join(_REPO_ROOT, "outputs")
PROCESSED_DIR = os.path.join(_REPO_ROOT, "data", "processed")
GAT_CHECKPOINT = os.path.join(OUTPUT_DIR, "gat_model.pt")


def explain_sample(model, graph, device="cpu"):
    model.eval()
    x = graph.x.to(device)
    edge_index = graph.edge_index.to(device)
    batch = torch.zeros(x.size(0), dtype=torch.long, device=device)

    with torch.no_grad():
        out, attn_layers = model(x, edge_index, batch, return_attention=True)
        pred = out.argmax(dim=1).item()
        prob = F.softmax(out, dim=1)[0, 1].item()

    # Use the LAST layer's attention (closest to the final prediction)
    ei, alpha = attn_layers[-1]
    alpha = alpha.mean(dim=1)  # average across attention heads

    edge_scores = []
    for k in range(ei.shape[1]):
        src, dst = ei[0, k].item(), ei[1, k].item()
        edge_scores.append((NODE_TYPES[src], NODE_TYPES[dst], alpha[k].item()))
    edge_scores.sort(key=lambda t: -t[2])

    return pred, prob, edge_scores


def _load_splits(seed):
    """Prefer the exact splits train.py persisted; fall back to re-splitting
    (deterministic with the same seed, but only exact if the CSV is unchanged)."""
    train_path = os.path.join(PROCESSED_DIR, "train.csv")
    test_path = os.path.join(PROCESSED_DIR, "test.csv")
    if os.path.exists(train_path) and os.path.exists(test_path):
        print(f"Loading persisted splits from {PROCESSED_DIR}")
        return pd.read_csv(train_path), pd.read_csv(test_path)

    print("[WARN] No persisted splits found -- re-splitting load_dataset() output. "
          "Run train.py first to guarantee this matches its reported metrics.")
    df = load_dataset()
    return train_test_split(df, test_size=0.2, stratify=df["binary_label"], random_state=seed)


def main(n_examples=5, seed=42):
    train_df, test_df = _load_splits(seed)
    scaler, cols = fit_scaler(train_df)  # fit stats only -- no need to build train graphs here
    test_graphs, _, _ = build_dataset(test_df, scaler, cols)

    in_dim = test_graphs[0].x.shape[1]
    model = GATClassifier(in_dim, hidden_dim=64)

    if not os.path.exists(GAT_CHECKPOINT):
        raise FileNotFoundError(
            f"No trained GAT checkpoint at {GAT_CHECKPOINT}.\n"
            "Run `python train.py --model gat` first -- explain.py now explains "
            "the actual validated model instead of training its own throwaway copy."
        )
    model.load_state_dict(torch.load(GAT_CHECKPOINT, map_location="cpu"))
    print(f"Loaded trained GAT weights from {GAT_CHECKPOINT}")

    print(f"\nExplaining {n_examples} test predictions via GAT attention:\n")
    malicious_samples = [g for g in test_graphs if g.y.item() == 1][:n_examples]

    for i, g in enumerate(malicious_samples):
        pred, prob, edges = explain_sample(model, g)
        label_str = "MALICIOUS" if pred == 1 else "benign"
        print(f"Sample {i+1} | true=malicious | predicted={label_str} (p={prob:.3f})")
        print("  Top artifact-relationship signals (by attention weight):")
        for src, dst, score in edges[:3]:
            print(f"    {src:12s} <-> {dst:12s}  attention={score:.4f}")
        print()


if __name__ == "__main__":
    main()
