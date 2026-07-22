"""
Interactive frontend for the Memory Forensics GNN project.

Lets you feed the trained GraphSAGE/GAT model a memory-scan sample (a random
held-out row, a specific row, or your own CSV), see the Benign/Malicious
verdict next to the RandomForest baseline, and -- for the GAT model -- see
which artifact-relationship the model paid the most attention to, rendered
as the actual 9-node investigation graph.

Run with:
    streamlit run app.py
"""

import json
import os
import random
import tempfile

import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from data_loader import load_dataset, FEATURE_GROUPS, FAMILY_NAMES
from graph_builder import fit_scaler, load_scaler, build_graph_from_row, NODE_TYPES, EDGE_TEMPLATE, _max_group_len
from model import build_model, GATClassifier

_REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
OUTPUT_DIR = os.path.join(_REPO_ROOT, "outputs")
PROCESSED_DIR = os.path.join(_REPO_ROOT, "data", "processed")
ALL_FEATURES = [c for group in FEATURE_GROUPS.values() for c in group]

try:
    import shutil as _shutil
    LIVE_DUMP_AVAILABLE = _shutil.which("volmemlyzer") is not None
except Exception:
    LIVE_DUMP_AVAILABLE = False

st.set_page_config(
    page_title="Memory forensics GNN",
    page_icon=":material/memory_alt:",
    layout="wide",
)


# ---------------------------------------------------------------- data/model

@st.cache_data
def get_splits():
    train_path = os.path.join(PROCESSED_DIR, "train.csv")
    test_path = os.path.join(PROCESSED_DIR, "test.csv")
    if os.path.exists(train_path) and os.path.exists(test_path):
        return pd.read_csv(train_path), pd.read_csv(test_path)
    df = load_dataset()
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["binary_label"], random_state=42
    )
    return train_df, test_df


@st.cache_resource
def get_scaler():
    # Prefer the scaler persisted at training time -- reproduces the EXACT
    # scaling the checkpoint was trained with, which matters a lot when only
    # a small bundled demo sample is available (re-fitting from that sample
    # instead would silently mis-scale every prediction). Falls back to
    # fitting fresh only if a model was never actually trained here.
    scaler_path = os.path.join(OUTPUT_DIR, "scaler.json")
    if os.path.exists(scaler_path):
        return load_scaler(scaler_path)
    train_df, _ = get_splits()
    return fit_scaler(train_df)


@st.cache_resource
def get_gnn_model(name):
    scaler, cols = get_scaler()
    model = build_model(name, _max_group_len(), hidden_dim=64)
    ckpt = os.path.join(OUTPUT_DIR, f"{name}_model.pt")
    if not os.path.exists(ckpt):
        return None
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    return model


@st.cache_resource
def get_multiclass_scaler():
    # Same idea as get_scaler(), but for the family (Trojan/Spyware/Ransomware)
    # model, which was fit on a different train/val/test split (stratified on
    # family_label, see train_multiclass.py). Prefer the persisted scaler;
    # fall back to reproducing train_multiclass.py's exact split (same seed)
    # so older checkpoints trained before scaler_multiclass.json existed
    # still score correctly instead of silently using mismatched scaling.
    scaler_path = os.path.join(OUTPUT_DIR, "scaler_multiclass.json")
    if os.path.exists(scaler_path):
        return load_scaler(scaler_path)
    df = load_dataset()
    train_df, _ = train_test_split(df, test_size=0.2, stratify=df["family_label"], random_state=42)
    train_df, _ = train_test_split(train_df, test_size=0.15, stratify=train_df["family_label"], random_state=42)
    return fit_scaler(train_df)


@st.cache_resource
def get_multiclass_model(name="sage"):
    scaler, cols = get_multiclass_scaler()
    model = build_model(name, _max_group_len(), hidden_dim=64, num_classes=4)
    ckpt = os.path.join(OUTPUT_DIR, f"{name}_multiclass_model.pt")
    if not os.path.exists(ckpt):
        return None
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    return model


def predict_family(row):
    """Runs the family (Benign/Trojan/Spyware/Ransomware) model on one row.
    Returns None if no multiclass checkpoint has been trained yet."""
    model = get_multiclass_model("sage")
    if model is None:
        return None
    scaler, cols = get_multiclass_scaler()
    graph = build_graph_from_row(row, scaler, cols)
    batch = torch.zeros(graph.num_nodes, dtype=torch.long)
    with torch.no_grad():
        out = model(graph.x, graph.edge_index, batch)
        probs = F.softmax(out, dim=1)[0].tolist()
    return dict(zip(FAMILY_NAMES, probs))


# Plain-language explanation of each malware family -- so the family label
# means something to someone who isn't a security analyst.
FAMILY_DESCRIPTIONS = {
    "Benign": ("No malware detected", ":material/check_circle:", "green",
               "This scan looks like normal, everyday system activity."),
    "Trojan": ("Trojan", ":material/security:", "orange",
               "Malware disguised as (or hidden inside) legitimate-looking software — "
               "often used to open a hidden backdoor for an attacker."),
    "Spyware": ("Spyware", ":material/visibility:", "orange",
                "Malware built to quietly watch and steal data — keystrokes, passwords, "
                "screen activity — without the user noticing."),
    "Ransomware": ("Ransomware", ":material/lock:", "red",
                   "Malware that locks or encrypts files and demands payment to release them."),
}


@st.cache_resource
def get_baseline_model():
    train_df, _ = get_splits()
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(train_df[ALL_FEATURES], train_df["binary_label"])
    return rf


@st.cache_data
def get_generalization_summary():
    path = os.path.join(OUTPUT_DIR, "generalization_summary.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data
def get_multiclass_summary():
    path = os.path.join(OUTPUT_DIR, "multiclass_summary.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def predict_gnn(model, scaler, cols, row, is_gat):
    graph = build_graph_from_row(row, scaler, cols)
    batch = torch.zeros(graph.num_nodes, dtype=torch.long)
    with torch.no_grad():
        if is_gat:
            out, attn_layers = model(graph.x, graph.edge_index, batch, return_attention=True)
        else:
            out = model(graph.x, graph.edge_index, batch)
            attn_layers = None
        prob = F.softmax(out, dim=1)[0, 1].item()

    edge_scores = None
    if attn_layers is not None:
        ei, alpha = attn_layers[-1]
        alpha = alpha.mean(dim=1)
        edge_scores = {}
        for k in range(ei.shape[1]):
            src, dst = ei[0, k].item(), ei[1, k].item()
            if src == dst:
                continue
            key = tuple(sorted((NODE_TYPES[src], NODE_TYPES[dst])))
            edge_scores[key] = max(edge_scores.get(key, 0.0), alpha[k].item())
    return prob, edge_scores


# Plain-language labels for the 9 artifact types -- the raw Volatility
# plugin names (pslist, ldrmodules, ...) mean nothing to a first-time
# visitor, so every place these show up in the UI uses this instead.
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


def render_artifact_graph(edge_scores=None):
    """Graphviz rendering of the 9-node artifact-relationship graph. Edge
    thickness/color encodes GAT attention when available."""
    lines = [
        "graph G {",
        "  layout=neato;",
        "  overlap=false;",
        "  bgcolor=transparent;",
        '  node [shape=box style="rounded,filled" fontname="Helvetica" fontsize=11 '
        'color="#5b6270" fillcolor="#1c2128" fontcolor="#e6e6e6" margin=0.15];',
        '  edge [fontname="Helvetica" fontsize=9 fontcolor="#9aa4b2"];',
    ]
    for n in NODE_TYPES:
        lines.append(f'  "{n}" [label="{NODE_LABELS[n]}"];')

    max_score = max(edge_scores.values()) if edge_scores else 1.0
    for a, b in EDGE_TEMPLATE:
        key = tuple(sorted((a, b)))
        score = edge_scores.get(key) if edge_scores else None
        if score is not None:
            norm = score / max_score if max_score > 0 else 0
            width = 1 + norm * 5
            color = "#ff6b6b" if norm > 0.6 else ("#ffb347" if norm > 0.3 else "#5b6270")
            lines.append(f'  "{a}" -- "{b}" [penwidth={width:.2f} color="{color}"];')
        else:
            lines.append(f'  "{a}" -- "{b}" [penwidth=1.2 color="#3a4150"];')
    lines.append("}")
    st.graphviz_chart("\n".join(lines), width="stretch")


# --------------------------------------------------------------------- UI

train_df, test_df = get_splits()
scaler, cols = get_scaler()
full_df = load_dataset()
gen_summary = get_generalization_summary()

st.title("Memory forensics GNN")
st.markdown(
    "##### Catching malware that never touches the disk — using memory scans instead of files"
)

tab_overview, tab_try = st.tabs([":material/menu_book: What this is", ":material/science: Try it live"])

# ------------------------------------------------------------- Overview tab

with tab_overview:
    st.markdown(
        "Normal antivirus scans **files on disk**. Some modern malware never "
        "writes a file — it runs entirely in a computer's RAM (\"fileless\" "
        "malware), so file-based scanning never sees it. The only way to "
        "catch it is to look at a **memory scan**: a snapshot of everything "
        "running in RAM at one moment. This app takes that scan and predicts "
        "**Benign or Malicious** using a graph neural network (GNN), then "
        "shows you *why* it made that call."
    )

    with st.container(horizontal=True):
        with st.container(border=True, horizontal_alignment="center"):
            st.markdown(":material/looks_one: **Take a scan**")
            st.caption("~55 measurements from one memory snapshot — process count, injected code, hidden modules, etc.")
        with st.container(border=True, horizontal_alignment="center"):
            st.markdown(":material/looks_two: **Build a graph**")
            st.caption("The 9 measurement groups become nodes; edges connect ones a real analyst would cross-check.")
        with st.container(border=True, horizontal_alignment="center"):
            st.markdown(":material/looks_3: **Get a verdict**")
            st.caption("The model classifies the whole graph, and can point to which relationship drove its answer.")

    st.markdown("#### The evidence, in plain numbers")
    st.caption("Trained and tested on CIC-MalMem-2022 — 58,596 real memory scans, half clean, half infected.")

    with st.container(horizontal=True):
        st.metric("Memory scans in the dataset", f"{len(full_df):,}", border=True)
        st.metric("Clean (benign)", f"{(full_df['binary_label'] == 0).sum():,}", border=True)
        st.metric("Infected (malicious)", f"{(full_df['binary_label'] == 1).sum():,}", border=True)
        st.metric("Malware families covered", f"{full_df.loc[full_df['binary_label'] == 1, 'family'].nunique()}", border=True)

    if gen_summary:
        with st.container(border=True):
            st.markdown("**The harder test: malware variants it never trained on**")
            st.caption(
                "Same-distribution accuracy is easy to inflate. This holds out entire malware "
                "subfamilies (" + ", ".join(gen_summary["held_out_subfamilies"]) + ") from "
                "training entirely, then scores the model on those unseen variants."
            )
            id_acc = gen_summary["in_distribution"]["accuracy"]
            gen_acc = gen_summary["unseen_families"]["accuracy"]
            id_fnr = gen_summary["in_distribution"]["fnr"]
            gen_fnr = gen_summary["unseen_families"]["fnr"]
            with st.container(horizontal=True):
                st.metric("Accuracy, malware it's seen before", f"{id_acc * 100:.2f}%", border=True)
                st.metric(
                    "Accuracy, malware it's NEVER seen", f"{gen_acc * 100:.2f}%",
                    f"{(gen_acc - id_acc) * 100:.2f} pts", border=True,
                )
                st.metric("Missed malware, seen", f"{id_fnr * 100:.2f}%", border=True)
                st.metric(
                    "Missed malware, unseen", f"{gen_fnr * 100:.2f}%",
                    f"{(gen_fnr - id_fnr) * 100:.2f} pts", delta_color="inverse", border=True,
                )
    else:
        st.info(
            "Run `python generalization_test.py` to unlock the unseen-malware-family "
            "generalization comparison here.",
            icon=":material/info:",
        )

    st.markdown("#### Being honest about the limits")
    st.caption(
        "A model that only reports its wins isn't trustworthy. Two things worth knowing before you trust this one:"
    )
    with st.container(horizontal=True):
        with st.container(border=True):
            st.markdown(":material/balance: **A simpler model sometimes wins**")
            mc_summary = get_multiclass_summary()
            if mc_summary:
                st.caption(
                    "On the harder Trojan/Spyware/Ransomware classification task, a plain RandomForest "
                    f"({mc_summary['rf_accuracy'] * 100:.1f}% accuracy) currently beats this GNN "
                    f"({mc_summary['gnn_accuracy'] * 100:.1f}%). Reported as-is, not hidden."
                )
            else:
                st.caption(
                    "On the harder Trojan/Spyware/Ransomware classification task, a plain RandomForest "
                    "currently beats this GNN. Run `python train_multiclass.py` to see the exact numbers. "
                    "Reported as-is, not hidden."
                )
        with st.container(border=True):
            st.markdown(":material/warning: **Live scanning isn't reliable yet**")
            st.caption(
                "Scanning a REAL live memory dump works end-to-end, but most of the underlying "
                "Volatility3 plugins hang on large real machines today — see \"Try it live\" for the honest breakdown."
            )

    st.markdown("Now go try it yourself in the **Try it live** tab above.")

# -------------------------------------------------------------------- Sidebar
# (controls live in the sidebar regardless of which tab is open; results
# only render inside the "Try it live" tab below)

with st.sidebar:
    st.subheader("Step 1: give it a scan")
    st.caption(
        "No idea what a \"memory scan\" file looks like? Start with the first option below — "
        "it's a real scan, no file needed. Then check the \"Try it live\" tab for the verdict."
    )
    model_choice = st.segmented_control(
        "GNN variant", options=["sage", "gat"], default="gat",
        help="GAT exposes attention weights for the graph explanation below.",
    )
    sample_sources = [
        "Demo scan (no file needed)",
        "Pick a specific demo scan",
        "Upload a scan CSV",
    ]
    if LIVE_DUMP_AVAILABLE:
        sample_sources.append("Scan a real memory dump file")
    source = st.radio("Pick a sample", sample_sources, label_visibility="collapsed")
    if not LIVE_DUMP_AVAILABLE:
        st.caption(
            "Scanning a real memory-dump file isn't available in this deployment "
            "(needs Volatility3 installed locally with admin rights — see README)."
        )

    row = None
    true_label = None
    live_report = None

    if source == "Demo scan (no file needed)":
        if st.button("Give me a new scan", icon=":material/casino:", width="stretch"):
            st.session_state["row_idx"] = random.randrange(len(test_df))
        idx = st.session_state.get("row_idx", random.randrange(len(test_df)))
        row = test_df.iloc[idx]
        true_label = int(row["binary_label"])
        st.caption(f"Real memory scan from the test set (row #{idx}) — the model has never seen this exact scan during training.")

    elif source == "Pick a specific demo scan":
        idx = st.number_input("Row index", min_value=0, max_value=len(test_df) - 1, value=0, step=1)
        row = test_df.iloc[idx]
        true_label = int(row["binary_label"])

    elif source == "Upload a scan CSV":
        st.caption(
            "For people who already have their own memory-forensics measurements "
            "(e.g. Volatility/VolMemLyzer output) as a CSV."
        )
        uploaded = st.file_uploader("CSV with the same ~55 feature columns", type=["csv"])
        if uploaded is not None:
            up_df = pd.read_csv(uploaded)
            row = up_df.iloc[0]
        else:
            st.caption("Copy the header from data/processed/test.csv as a template.")

    else:
        st.caption(
            "Runs live Volatility3 extraction on a real memory image. "
            "Some features (psxview cross-view checks) have no Volatility3 "
            "equivalent and are imputed from the training-set mean -- the "
            "report below shows exactly which."
        )
        dump = st.file_uploader("Raw memory image (.raw/.vmem/.dmp/.mem)", type=None)
        if dump is not None:
            dump_path = os.path.join(tempfile.gettempdir(), dump.name)
            with open(dump_path, "wb") as f:
                f.write(dump.getbuffer())
            with st.spinner("Running Volatility3 plugins on the image (can take a few minutes)..."):
                try:
                    from live_extract import extract_features_from_dump
                    row, live_report = extract_features_from_dump(dump_path)
                except Exception as e:
                    st.error(f"Extraction failed: {e}", icon=":material/error:")

    st.divider()
    st.caption("Model checkpoints load from outputs/*_model.pt")

with tab_try:
    if row is None:
        st.info(
            "Look left → click **\"Give me a new scan\"** in the sidebar. That's it — "
            "no file, no setup. You'll get a Benign/Malicious verdict here, and if it's "
            "malicious, which type (Ransomware/Spyware/Trojan) it most likely is.",
            icon=":material/arrow_back:",
        )

    model = get_gnn_model(model_choice) if row is not None else None
    if row is not None and model is None:
        st.error(
            f"No trained {model_choice.upper()} checkpoint at outputs/{model_choice}_model.pt — "
            f"run `python train.py --model {model_choice}` first.",
            icon=":material/error:",
        )

    if row is not None and model is not None:
        is_gat = model_choice == "gat"
        prob, edge_scores = predict_gnn(model, scaler, cols, row, is_gat)
        pred = int(prob >= 0.5)

        rf = get_baseline_model()
        rf_prob = rf.predict_proba(row[ALL_FEATURES].to_frame().T)[0, 1]
        rf_pred = int(rf_prob >= 0.5)

        MIN_REAL_FEATURES_FOR_TRUST = 6  # below this, don't trust the verdict at all -- see README's live-scan findings

        live_verdict_untrustworthy = False
        if live_report is not None:
            n_real = len(live_report.real)
            n_total = n_real + len(live_report.imputed)
            if n_real == 0:
                st.error(
                    "0/55 features were measured from this image — Volatility3 couldn't identify "
                    "a supported OS/profile in it, so this verdict is meaningless (100% imputed).",
                    icon=":material/error:",
                )
                live_verdict_untrustworthy = True
            elif n_real < MIN_REAL_FEATURES_FOR_TRUST:
                st.error(
                    f"Only {n_real}/{n_total} features were measured directly from this dump — too few to "
                    "trust the verdict below. On real (non-toy) images, most Volatility3 plugins can hang "
                    "indefinitely, so this is often as good as it gets right now; a confident-looking verdict "
                    "built mostly from imputed averages plus one or two real outliers has been observed to be "
                    "a false positive (see README's \"Real-time / live memory dump scanning\" section). Treat "
                    "this as a demonstration that the pipeline runs end-to-end, not as a real detection.",
                    icon=":material/error:",
                )
                live_verdict_untrustworthy = True
            else:
                st.warning(
                    f"Live extraction: {n_real}/{n_total} features measured directly from this dump, "
                    f"{n_total - n_real} imputed from the training-set mean (psxview has no Volatility3 equivalent).",
                    icon=":material/info:",
                )
                n_extreme = len(live_report.extreme)
                if n_extreme and (n_real - n_extreme) < MIN_REAL_FEATURES_FOR_TRUST:
                    st.error(
                        f"{n_extreme} of the real feature(s) ({', '.join(live_report.extreme)}) fall "
                        "outside anything seen in training, and there are few other real features to "
                        "balance them out. This is the exact combination that produced a confirmed "
                        "false positive during hardware testing — treat the verdict below with real "
                        "skepticism even though it technically cleared the minimum real-feature bar.",
                        icon=":material/error:",
                    )
                    live_verdict_untrustworthy = True
                with st.expander("Which features were imputed?"):
                    st.write(", ".join(live_report.imputed))
                if live_report.rejected:
                    st.caption(
                        f"{len(live_report.rejected)} value(s) were measured but rejected as physically "
                        f"implausible (likely an extractor bug) and imputed instead: {', '.join(live_report.rejected)}"
                    )

        col_verdict, col_graph = st.columns([1, 1.4])

        with col_verdict:
            with st.container(border=True):
                st.markdown("**GNN verdict**")
                if live_verdict_untrustworthy:
                    st.badge("NOT RELIABLE", icon=":material/block:", color="gray")
                    st.caption("Too little real data was measured to trust this verdict — see the error above.")
                elif pred == 1:
                    st.badge("MALICIOUS", icon=":material/warning:", color="red")
                else:
                    st.badge("BENIGN", icon=":material/check_circle:", color="green")
                st.metric("Confidence (malicious probability)", f"{prob * 100:.1f}%", border=True)

                if true_label is not None:
                    correct = pred == true_label
                    true_str = "MALICIOUS" if true_label == 1 else "BENIGN"
                    if correct:
                        st.success(f"Correct — ground truth is {true_str}", icon=":material/check:")
                    else:
                        st.error(f"Wrong — ground truth is {true_str}", icon=":material/close:")

            if pred == 1 and not live_verdict_untrustworthy:
                with st.container(border=True):
                    st.markdown("**What kind of malware is it?**")
                    family_probs = predict_family(row)
                    if family_probs is None:
                        st.caption(
                            "No family model trained yet — run `python train_multiclass.py` "
                            "to unlock a Ransomware/Spyware/Trojan breakdown here."
                        )
                    else:
                        top_family = max(family_probs, key=family_probs.get)
                        name, icon, color, desc = FAMILY_DESCRIPTIONS[top_family]
                        if top_family == "Benign":
                            # The binary model says malicious but the family model's
                            # top guess is Benign -- the two disagree; say so plainly
                            # instead of showing a confusing "Benign" badge here.
                            st.warning(
                                "The binary model flagged this as malicious, but the family "
                                "model isn't confident which malware type it is.",
                                icon=":material/help:",
                            )
                        else:
                            st.badge(name.upper(), icon=icon, color=color)
                            st.caption(desc)
                        st.caption("Most likely family, by probability:")
                        for fam in sorted(family_probs, key=family_probs.get, reverse=True):
                            st.progress(family_probs[fam], text=f"{fam} — {family_probs[fam] * 100:.1f}%")

            with st.container(border=True):
                st.markdown("**Baseline comparison (RandomForest)**")
                with st.container(horizontal=True):
                    st.metric(
                        "RandomForest verdict",
                        "MALICIOUS" if rf_pred == 1 else "BENIGN",
                        border=True,
                    )
                    st.metric("RandomForest confidence", f"{rf_prob * 100:.1f}%", border=True)
                if rf_pred != pred:
                    st.warning("Models disagree on this sample.", icon=":material/warning:")

        with col_graph:
            with st.container(border=True):
                st.markdown("**Which evidence drove this call**")
                if is_gat and edge_scores:
                    st.caption("Thicker/redder lines = the model paid more attention to that relationship for this sample.")
                else:
                    st.caption("Switch to the GAT model in the sidebar to see attention-weighted evidence.")
                render_artifact_graph(edge_scores if is_gat else None)

                if is_gat and edge_scores:
                    top_edges = sorted(edge_scores.items(), key=lambda kv: -kv[1])[:3]
                    st.markdown("**Top signals driving this prediction:**")
                    for (a, b), score in top_edges:
                        st.markdown(f"- **{NODE_LABELS[a]}** ↔ **{NODE_LABELS[b]}** — attention {score:.3f}")

        with st.expander("Raw feature values for this sample", icon=":material/table_chart:"):
            feature_tabs = st.tabs([NODE_LABELS[g] for g in FEATURE_GROUPS.keys()])
            for tab, group_name in zip(feature_tabs, FEATURE_GROUPS.keys()):
                with tab:
                    group_cols = FEATURE_GROUPS[group_name]
                    st.dataframe(
                        row[group_cols].to_frame(name="value"),
                        width="stretch",
                    )
