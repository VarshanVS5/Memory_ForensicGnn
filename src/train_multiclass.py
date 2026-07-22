"""
Multiclass malware FAMILY classification (Benign / Trojan / Spyware / Ransomware).

train.py's binary Benign-vs-Malicious task is saturated -- every model tried
(GraphSAGE, GAT, RandomForest, GradientBoosting) lands at ~99.9%+ accuracy,
which means there's no headroom left to show whether the GNN's relationship
modeling actually buys anything over a flat tabular model. Family
classification is harder and not saturated, so it's the more honest place to
look for a real GNN-vs-baseline gap (see PROJECT_OVERVIEW.md, "What would
make this stronger").

Usage:
    python train_multiclass.py --model sage --epochs 40
    python train_multiclass.py --model gat  --epochs 40
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report
)

from data_loader import load_dataset, FAMILY_NAMES, FEATURE_GROUPS
from graph_builder import build_dataset, fit_scaler, save_scaler
from model import build_model
from train import run_epoch, set_seed

_REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
OUTPUT_DIR = os.path.join(_REPO_ROOT, "outputs")
PROCESSED_DIR = os.path.join(_REPO_ROOT, "data", "processed")
ALL_FEATURES = [c for group in FEATURE_GROUPS.values() for c in group]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["sage", "gat"], default="sage")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    df = load_dataset()
    print("Family distribution:\n", df["family"].value_counts())

    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["family_label"], random_state=args.seed
    )
    train_df, val_df = train_test_split(
        train_df, test_size=0.15, stratify=train_df["family_label"], random_state=args.seed
    )

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    train_df.to_csv(os.path.join(PROCESSED_DIR, "train_multiclass.csv"), index=False)
    test_df.to_csv(os.path.join(PROCESSED_DIR, "test_multiclass.csv"), index=False)

    scaler, cols = fit_scaler(train_df)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_scaler(scaler, cols, os.path.join(OUTPUT_DIR, "scaler_multiclass.json"))
    train_graphs, _, _ = build_dataset(train_df, scaler, cols, label_col="family_label")
    val_graphs, _, _ = build_dataset(val_df, scaler, cols, label_col="family_label")
    test_graphs, _, _ = build_dataset(test_df, scaler, cols, label_col="family_label")

    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size)
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size)

    in_dim = train_graphs[0].x.shape[1]
    model = build_model(args.model, in_dim, hidden_dim=args.hidden_dim, num_classes=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

    best_val_f1, best_state = 0.0, None
    for epoch in range(1, args.epochs + 1):
        train_loss, *_ = run_epoch(model, train_loader, optimizer, device)
        val_loss, val_y, val_pred, _ = run_epoch(model, val_loader, None, device)
        val_f1 = f1_score(val_y, val_pred, average="macro", zero_division=0)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | val_macroF1 {val_f1:.4f}")

    model.load_state_dict(best_state)
    _, y_true, y_pred, _ = run_epoch(model, test_loader, None, device)

    print("\n=== GNN Test Set Results (family classification) ===")
    print(classification_report(y_true, y_pred, target_names=FAMILY_NAMES, zero_division=0))
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix (rows=true, cols=pred), order =", FAMILY_NAMES)
    print(cm)
    gnn_acc = accuracy_score(y_true, y_pred)
    gnn_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # RandomForest baseline on the same task, for the comparison that
    # actually matters here: does the GNN beat flat tabular features when
    # the task ISN'T saturated?
    rf = RandomForestClassifier(n_estimators=300, random_state=args.seed, n_jobs=-1)
    rf.fit(train_df[ALL_FEATURES], train_df["family_label"])
    rf_pred = rf.predict(test_df[ALL_FEATURES])
    rf_acc = accuracy_score(test_df["family_label"], rf_pred)
    rf_f1 = f1_score(test_df["family_label"], rf_pred, average="macro", zero_division=0)

    print("\n=== RandomForest baseline (family classification) ===")
    print(classification_report(test_df["family_label"], rf_pred, target_names=FAMILY_NAMES, zero_division=0))

    print("\n=== Summary: does the GNN beat RandomForest on the NON-saturated task? ===")
    print(f"{'Model':20s} {'Accuracy':>10s} {'Macro F1':>10s}")
    print(f"{args.model.upper() + ' (GNN)':20s} {gnn_acc:10.4f} {gnn_f1:10.4f}")
    print(f"{'RandomForest':20s} {rf_acc:10.4f} {rf_f1:10.4f}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model_path = os.path.join(OUTPUT_DIR, f"{args.model}_multiclass_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")

    with open(os.path.join(OUTPUT_DIR, "multiclass_summary.json"), "w") as f:
        json.dump({
            "families": FAMILY_NAMES,
            "gnn_model": args.model,
            "gnn_accuracy": gnn_acc,
            "gnn_macro_f1": gnn_f1,
            "rf_accuracy": rf_acc,
            "rf_macro_f1": rf_f1,
            "gnn_confusion_matrix": cm.tolist(),
        }, f, indent=2)


if __name__ == "__main__":
    main()
