"""
Baseline tabular classifiers (Random Forest + XGBoost/GradientBoosting) on
the same CIC-MalMem-2022 features, for direct comparison against the GNN.

This answers the inevitable question: "why use a GNN instead of just
XGBoost on the flat features?" — run this alongside train.py and compare.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

from data_loader import load_dataset, FEATURE_GROUPS

ALL_FEATURES = [c for group in FEATURE_GROUPS.values() for c in group]


def evaluate(name, y_true, y_pred, y_prob):
    print(f"\n=== {name} ===")
    print(classification_report(y_true, y_pred, target_names=["Benign", "Malicious"]))
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print("Confusion matrix [ [TN FP] [FN TP] ]:\n", cm)
    print(f"False Negative Rate: {fn / (fn + tp):.4f}")
    print(f"False Positive Rate: {fp / (fp + tn):.4f}")
    try:
        print(f"ROC-AUC: {roc_auc_score(y_true, y_prob):.4f}")
    except ValueError:
        pass
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }


def main(seed=42):
    df = load_dataset()
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["binary_label"], random_state=seed
    )

    scaler = StandardScaler().fit(train_df[ALL_FEATURES])
    X_train = scaler.transform(train_df[ALL_FEATURES])
    X_test = scaler.transform(test_df[ALL_FEATURES])
    y_train = train_df["binary_label"].values
    y_test = test_df["binary_label"].values

    results = {}

    rf = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=seed, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_prob = rf.predict_proba(X_test)[:, 1]
    results["RandomForest"] = evaluate("Random Forest", y_test, rf_pred, rf_prob)

    # GradientBoosting as XGBoost stand-in (avoids needing xgboost install;
    # swap for xgboost.XGBClassifier if you have it installed -- same API)
    gb = GradientBoostingClassifier(n_estimators=300, max_depth=3, random_state=seed)
    gb.fit(X_train, y_train)
    gb_pred = gb.predict(X_test)
    gb_prob = gb.predict_proba(X_test)[:, 1]
    results["GradientBoosting"] = evaluate("Gradient Boosting (XGBoost-equivalent)", y_test, gb_pred, gb_prob)

    # Feature importance -- useful to sanity-check against which node types
    # the GAT attention weights highlight later
    importances = sorted(zip(ALL_FEATURES, rf.feature_importances_), key=lambda x: -x[1])
    print("\nTop 10 most important features (Random Forest):")
    for feat, imp in importances[:10]:
        print(f"  {feat:35s} {imp:.4f}")

    print("\n=== Summary (compare these numbers against train.py's GNN output) ===")
    for name, r in results.items():
        print(f"{name:20s} Acc={r['accuracy']:.4f} F1={r['f1']:.4f} Precision={r['precision']:.4f} Recall={r['recall']:.4f}")

    return results


if __name__ == "__main__":
    main()
