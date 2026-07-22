"""
Try the trained model on ONE sample and see its verdict -- this is the
"enter an input, get an output" demo that train.py/baseline.py don't give you
(those only report aggregate accuracy over thousands of rows at once).

Two ways to use it:

  python predict.py                  # picks a random row from the held-out
                                      # test set and shows the model's guess
                                      # vs. the real answer
  python predict.py --index 42       # a specific row from the test set
  python predict.py --csv myrow.csv  # score YOUR OWN row -- must have the
                                      # same ~55 feature columns as
                                      # data/CIC-MalMem-2022.csv (copy the
                                      # header from data/processed/test.csv
                                      # if you want a template)
  python predict.py --dump scan.raw  # score a REAL raw memory image via
                                      # live Volatility3 extraction (see
                                      # live_extract.py -- some features are
                                      # imputed, reported features tell you
                                      # which ones)

Requires a trained model first: run train.py (RUN_ME.bat does this).
"""

import argparse
import os
import random

import pandas as pd
import torch

from graph_builder import fit_scaler, load_scaler, build_graph_from_row, _max_group_len
from model import build_model

_REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
OUTPUT_DIR = os.path.join(_REPO_ROOT, "outputs")
PROCESSED_DIR = os.path.join(_REPO_ROOT, "data", "processed")


def load_trained_model(name="sage"):
    checkpoint = os.path.join(OUTPUT_DIR, f"{name}_model.pt")
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(
            f"No trained model at {checkpoint}.\nRun train.py --model {name} first (RUN_ME.bat does this)."
        )
    scaler_path = os.path.join(OUTPUT_DIR, "scaler.json")
    if os.path.exists(scaler_path):
        # The exact scaler persisted at training time -- always correct,
        # regardless of whether the original train.csv split is present.
        scaler, cols = load_scaler(scaler_path)
    else:
        train_path = os.path.join(PROCESSED_DIR, "train.csv")
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"No {scaler_path} or {train_path}. Run train.py first -- it creates these.")
        # Re-fit the same scaler train.py used (it's deterministic given the
        # same persisted training rows, so this reproduces the exact scaling).
        train_df = pd.read_csv(train_path)
        scaler, cols = fit_scaler(train_df)

    model = build_model(name, _max_group_len())
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()
    return model, scaler, cols


def predict_row(model, scaler, cols, row):
    if "binary_label" not in row.index:
        row = row.copy()
        row["binary_label"] = -1  # placeholder; unused for prediction itself
    graph = build_graph_from_row(row, scaler, cols)
    batch = torch.zeros(graph.num_nodes, dtype=torch.long)
    with torch.no_grad():
        out = model(graph.x, graph.edge_index, batch)
        prob_malicious = torch.softmax(out, dim=1)[0, 1].item()
    pred = int(prob_malicious >= 0.5)
    return pred, prob_malicious


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["sage", "gat"], default="sage")
    ap.add_argument("--index", type=int, default=None,
                     help="Row number in data/processed/test.csv to test (default: random)")
    ap.add_argument("--csv", type=str, default=None,
                     help="Path to your own single-row CSV with the same feature columns")
    ap.add_argument("--dump", type=str, default=None,
                     help="Path to a REAL raw memory image -- runs live Volatility3 extraction")
    args = ap.parse_args()

    model, scaler, cols = load_trained_model(args.model)

    true_label = None
    if args.dump:
        from live_extract import extract_features_from_dump
        row, report = extract_features_from_dump(args.dump)
        source = args.dump
        print(f"\nLive extraction: {len(report.real)}/{len(report.real) + len(report.imputed)} features "
              f"measured directly from this dump; {len(report.imputed)} imputed from the training-set mean.")
        if report.imputed:
            print("Imputed:", ", ".join(report.imputed))
        if report.extreme and len(report.real) - len(report.extreme) < 6:
            print(f"WARNING: {len(report.extreme)} real feature(s) are outliers vs. training "
                  f"({', '.join(report.extreme)}) combined with very few other real features -- "
                  "this specific combination produced a confirmed false positive during hardware "
                  "testing. Do not trust the verdict below at face value.")
    elif args.csv:
        row = pd.read_csv(args.csv).iloc[0]
        source = args.csv
    else:
        test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))
        idx = args.index if args.index is not None else random.randrange(len(test_df))
        row = test_df.iloc[idx]
        true_label = int(row["binary_label"])
        source = f"test.csv, row #{idx}"

    pred, prob = predict_row(model, scaler, cols, row)
    label_str = "MALICIOUS" if pred == 1 else "BENIGN"

    print(f"\nSample source: {source}")
    print(f"Model prediction: {label_str}   (confidence: {prob * 100:.1f}% malicious)")
    if true_label is not None:
        true_str = "MALICIOUS" if true_label == 1 else "BENIGN"
        verdict = "CORRECT" if pred == true_label else "WRONG"
        print(f"Actual answer:    {true_str}")
        print(f"--> Model was {verdict}")
    print()


if __name__ == "__main__":
    main()
