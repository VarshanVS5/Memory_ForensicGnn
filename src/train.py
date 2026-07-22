"""
End-to-end train/eval for the Memory Forensics GNN project.

Usage:
    python train.py --model sage --epochs 40
    python train.py --model gat  --epochs 40

Reports: accuracy, precision, recall, F1, ROC-AUC, confusion matrix.
False-negative rate is called out explicitly since in malware detection
missed detections are the costly error, not false alarms.
"""

import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

from data_loader import load_dataset
from graph_builder import build_dataset, fit_scaler, save_scaler
from model import build_model

# Resolved relative to this file, not the caller's cwd, so `python train.py`
# works the same whether you run it from src/ or the repo root.
_REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
OUTPUT_DIR = os.path.join(_REPO_ROOT, "outputs")
PROCESSED_DIR = os.path.join(_REPO_ROOT, "data", "processed")


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_epoch(model, loader, optimizer=None, device="cpu"):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, all_preds, all_labels, all_probs = 0.0, [], [], []

    for batch in loader:
        batch = batch.to(device)
        if training:
            optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = F.cross_entropy(out, batch.y)

        if training:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        probs = F.softmax(out, dim=1)[:, 1].detach().cpu().numpy()
        preds = out.argmax(dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_probs.extend(probs)
        all_labels.extend(batch.y.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, all_labels, all_preds, all_probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["sage", "gat"], default="sage")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 1. Load data (real CIC-MalMem-2022 if present, else synthetic dev data)
    df = load_dataset()

    # 2. Split BEFORE fitting the scaler (avoid leakage)
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["binary_label"], random_state=args.seed
    )
    train_df, val_df = train_test_split(
        train_df, test_size=0.15, stratify=train_df["binary_label"], random_state=args.seed
    )

    # Persist the exact splits used for this run so train/val/test data can
    # be inspected directly instead of only seeing the aggregate metrics.
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    train_df.to_csv(os.path.join(PROCESSED_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(PROCESSED_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(PROCESSED_DIR, "test.csv"), index=False)
    print(f"Saved splits to {PROCESSED_DIR}: "
          f"train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    scaler, cols = fit_scaler(train_df)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_scaler(scaler, cols, os.path.join(OUTPUT_DIR, "scaler.json"))
    train_graphs, _, _ = build_dataset(train_df, scaler, cols)
    val_graphs, _, _ = build_dataset(val_df, scaler, cols)
    test_graphs, _, _ = build_dataset(test_df, scaler, cols)

    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size)
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size)

    in_dim = train_graphs[0].x.shape[1]
    model = build_model(args.model, in_dim, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

    best_val_f1, best_state = 0.0, None
    for epoch in range(1, args.epochs + 1):
        train_loss, *_ = run_epoch(model, train_loader, optimizer, device)
        val_loss, val_y, val_pred, _ = run_epoch(model, val_loader, None, device)
        val_f1 = f1_score(val_y, val_pred)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | val_F1 {val_f1:.4f}")

    model.load_state_dict(best_state)
    test_loss, y_true, y_pred, y_prob = run_epoch(model, test_loader, None, device)

    print("\n=== Test Set Results ===")
    print(classification_report(y_true, y_pred, target_names=["Benign", "Malicious"]))
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print("Confusion matrix [ [TN FP] [FN TP] ]:\n", cm)
    print(f"False Negative Rate (missed malware): {fn / (fn + tp):.4f}")
    print(f"False Positive Rate (false alarms):   {fp / (fp + tn):.4f}")
    try:
        auc = roc_auc_score(y_true, y_prob)
        print(f"ROC-AUC: {auc:.4f}")
    except ValueError:
        pass

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model_path = os.path.join(OUTPUT_DIR, f"{args.model}_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    main()
