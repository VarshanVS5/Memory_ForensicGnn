"""
Cross-family generalization test.

train.py / baseline.py report accuracy on a held-out **random** slice of
CIC-MalMem-2022 -- rows from the same malware subfamilies the model trained
on. That's the easy version of the test: it mostly proves the model can
interpolate within families it has already seen.

This script asks a harder, more honest question: can the model detect
malware subfamilies it has NEVER seen at all during training?

CIC-MalMem-2022's `Category` column encodes subfamily (e.g.
"Ransomware-Pysa-<hash>-3.raw" -> subfamily "Ransomware-Pysa"). Each of the
3 malicious families (Trojan, Spyware, Ransomware) ships 5 subfamilies. We
pick one subfamily per family, remove it ENTIRELY from training/validation,
train on the rest, then evaluate on:
  (a) an in-distribution test set (same subfamilies as training, held-out rows)
  (b) the held-out subfamilies (100% unseen malware variants)

The gap between (a) and (b) is the real generalization number -- and it's
the number worth reporting to a mentor, since near-100% accuracy on (a) is
not surprising given the whole field is saturated on this benchmark (see
PROJECT_OVERVIEW.md limitations).

Usage:
    python generalization_test.py --epochs 15
"""

import argparse
import json
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
from graph_builder import build_dataset, fit_scaler
from model import build_model
from train import run_epoch, set_seed

_REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
OUTPUT_DIR = os.path.join(_REPO_ROOT, "outputs")
PROCESSED_DIR = os.path.join(_REPO_ROOT, "data", "processed")

# One subfamily per malware family held out entirely from training. Picked
# to be the smallest subfamily in each family so the training pool still
# has plenty of examples of that family's OTHER variants to learn from.
HELD_OUT_SUBFAMILIES = ["Ransomware-Pysa", "Spyware-TIBS", "Trojan-Reconyc"]


def _subfamily(category: str) -> str:
    if str(category).lower().startswith("benign"):
        return "Benign"
    return "-".join(str(category).split("-")[:2])


def evaluate(name, y_true, y_pred, y_prob):
    print(f"\n=== {name} ===")
    print(classification_report(y_true, y_pred, target_names=["Benign", "Malicious"], zero_division=0))
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print("Confusion matrix [ [TN FP] [FN TP] ]:\n", cm)
    fnr = fn / (fn + tp) if (fn + tp) else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    print(f"False Negative Rate (missed malware): {fnr:.4f}")
    print(f"False Positive Rate (false alarms):   {fpr:.4f}")
    result = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "fnr": fnr,
        "fpr": fpr,
    }
    try:
        result["auc"] = roc_auc_score(y_true, y_prob)
        print(f"ROC-AUC: {result['auc']:.4f}")
    except ValueError:
        result["auc"] = float("nan")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["sage", "gat"], default="sage")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    df = load_dataset()
    df["subfamily"] = df["category"].apply(_subfamily)

    held_out_mask = df["subfamily"].isin(HELD_OUT_SUBFAMILIES)
    gen_test_df = df[held_out_mask].reset_index(drop=True)
    train_pool_df = df[~held_out_mask].reset_index(drop=True)

    print(f"Held-out (never trained on) subfamilies: {HELD_OUT_SUBFAMILIES}")
    print(f"Held-out generalization test rows: {len(gen_test_df)}")
    print(f"Training pool (everything else):   {len(train_pool_df)}")

    # In-distribution split, carved only from the training pool so the
    # held-out subfamilies never leak into train/val/in-dist-test.
    train_df, indist_test_df = train_test_split(
        train_pool_df, test_size=0.2, stratify=train_pool_df["binary_label"], random_state=args.seed
    )
    train_df, val_df = train_test_split(
        train_df, test_size=0.15, stratify=train_df["binary_label"], random_state=args.seed
    )

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    gen_test_df.to_csv(os.path.join(PROCESSED_DIR, "generalization_test.csv"), index=False)

    scaler, cols = fit_scaler(train_df)
    train_graphs, _, _ = build_dataset(train_df, scaler, cols)
    val_graphs, _, _ = build_dataset(val_df, scaler, cols)
    indist_test_graphs, _, _ = build_dataset(indist_test_df, scaler, cols)
    gen_test_graphs, _, _ = build_dataset(gen_test_df, scaler, cols)

    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size)
    indist_test_loader = DataLoader(indist_test_graphs, batch_size=args.batch_size)
    gen_test_loader = DataLoader(gen_test_graphs, batch_size=args.batch_size)

    in_dim = train_graphs[0].x.shape[1]
    model = build_model(args.model, in_dim, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

    best_val_f1, best_state = 0.0, None
    for epoch in range(1, args.epochs + 1):
        train_loss, *_ = run_epoch(model, train_loader, optimizer, device)
        val_loss, val_y, val_pred, _ = run_epoch(model, val_loader, None, device)
        val_f1 = f1_score(val_y, val_pred, zero_division=0)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | val_F1 {val_f1:.4f}")

    model.load_state_dict(best_state)

    _, y_true_id, y_pred_id, y_prob_id = run_epoch(model, indist_test_loader, None, device)
    indist_result = evaluate("In-distribution test (seen subfamilies, held-out rows)", y_true_id, y_pred_id, y_prob_id)

    _, y_true_gen, y_pred_gen, y_prob_gen = run_epoch(model, gen_test_loader, None, device)
    gen_result = evaluate(f"Generalization test (UNSEEN subfamilies: {HELD_OUT_SUBFAMILIES})", y_true_gen, y_pred_gen, y_prob_gen)

    print("\n=== Summary: does the model generalize to malware variants it never trained on? ===")
    print(f"{'Metric':12s} {'In-distribution':>16s} {'Unseen families':>18s} {'Gap':>8s}")
    for k in ["accuracy", "f1", "auc", "fnr", "fpr"]:
        a, b = indist_result[k], gen_result[k]
        print(f"{k:12s} {a:16.4f} {b:18.4f} {a - b:8.4f}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model_path = os.path.join(OUTPUT_DIR, f"{args.model}_generalization_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")

    summary_path = os.path.join(OUTPUT_DIR, "generalization_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "held_out_subfamilies": HELD_OUT_SUBFAMILIES,
            "n_train": len(train_df),
            "n_indist_test": len(indist_test_df),
            "n_generalization_test": len(gen_test_df),
            "in_distribution": indist_result,
            "unseen_families": gen_result,
        }, f, indent=2)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
