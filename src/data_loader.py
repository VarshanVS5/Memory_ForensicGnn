"""
Data loading for the Memory Forensics / Fileless Malware GNN project.

Primary dataset: CIC-MalMem-2022 (Canadian Institute for Cybersecurity)
  Download from: https://www.unb.ca/cic/datasets/malmem-2022.html
  or: https://www.kaggle.com/datasets/dhoogla/ciccicmalmem2022
  -> place the CSV as data/CIC-MalMem-2022.csv

The CSV contains ~55 Volatility-derived features per memory-dump "record"
(pslist, dlllist, handles, ldrmodules, malfind, svcscan, callbacks, ...)
with a Category column like:
    Benign
    Trojan-<family>
    Spyware-<family>
    Ransomware-<family>

This loader also ships a synthetic generator that mimics the same schema,
so the rest of the pipeline (graph construction + GNN) can be built and
sanity-tested without the real file present.
"""

import os
import numpy as np
import pandas as pd

RAW_CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "CIC-MalMem-2022.csv")

# The real dataset's feature groups (Volatility plugin -> feature prefix).
# Used both for synthetic generation and for grouping features into
# "node types" when we build the graph in graph_builder.py
FEATURE_GROUPS = {
    "pslist": ["pslist.nproc", "pslist.nppid", "pslist.avg_threads",
               "pslist.nprocs64bit", "pslist.avg_handlers"],
    "dlllist": ["dlllist.ndlls", "dlllist.avg_dlls_per_proc"],
    "handles": ["handles.nhandles", "handles.avg_handles_per_proc",
                "handles.nport", "handles.nfile", "handles.nevent",
                "handles.ndesktop", "handles.nkey", "handles.nthread",
                "handles.ndirectory", "handles.nsemaphore", "handles.ntimer",
                "handles.nsection", "handles.nmutant"],
    "ldrmodules": ["ldrmodules.not_in_load", "ldrmodules.not_in_init",
                   "ldrmodules.not_in_mem", "ldrmodules.not_in_load_avg",
                   "ldrmodules.not_in_init_avg", "ldrmodules.not_in_mem_avg"],
    "malfind": ["malfind.ninjections", "malfind.commitCharge",
                "malfind.protection", "malfind.uniqueInjections"],
    "psxview": ["psxview.not_in_pslist", "psxview.not_in_eprocess_pool",
                "psxview.not_in_ethread_pool", "psxview.not_in_pspcid_list",
                "psxview.not_in_csrss_handles", "psxview.not_in_session",
                "psxview.not_in_deskthrd",
                "psxview.not_in_pslist_false_avg", "psxview.not_in_eprocess_pool_false_avg",
                "psxview.not_in_ethread_pool_false_avg", "psxview.not_in_pspcid_list_false_avg",
                "psxview.not_in_csrss_handles_false_avg", "psxview.not_in_session_false_avg",
                "psxview.not_in_deskthrd_false_avg"],
    "modules": ["modules.nmodules"],
    "svcscan": ["svcscan.nservices", "svcscan.kernel_drivers",
                "svcscan.fs_drivers", "svcscan.process_services",
                "svcscan.shared_process_services", "svcscan.interactive_process_services",
                "svcscan.nactive"],
    "callbacks": ["callbacks.ncallbacks", "callbacks.nanonymous",
                  "callbacks.ngeneric"],
}

ALL_FEATURES = [f for group in FEATURE_GROUPS.values() for f in group]

# Multiclass family target (Benign vs. the 3 malware families), for the
# harder, non-saturated classification task -- see generalization_test.py's
# module docstring and PROJECT_OVERVIEW.md for why binary accuracy alone
# doesn't distinguish the GNN from a flat tabular model on this dataset.
FAMILY_LABEL_MAP = {"Benign": 0, "Trojan": 1, "Spyware": 2, "Ransomware": 3}
FAMILY_NAMES = list(FAMILY_LABEL_MAP.keys())


def load_real_dataset(path: str = RAW_CSV_PATH) -> pd.DataFrame:
    """Load the actual CIC-MalMem-2022 CSV. Raises a clear error if missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find {path}.\n"
            "Download CIC-MalMem-2022.csv from:\n"
            "  https://www.unb.ca/cic/datasets/malmem-2022.html\n"
            "  or https://www.kaggle.com/datasets/dhoogla/ciccicmalmem2022\n"
            "and place it at data/CIC-MalMem-2022.csv"
        )
    df = pd.read_csv(path)
    return _postprocess(df)


def _postprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize labels into binary + multiclass targets.

    Real CIC-MalMem-2022 schema: 'Category' holds detailed strings like
    'Ransomware-Ako-<hash>-1.raw' or 'Benign'; 'Class' holds the clean
    binary label ('Benign' / 'Malware'). The synthetic generator only
    produces a 'category' column, so both paths are handled here.
    """
    if "Class" in df.columns:
        df = df.rename(columns={"Category": "category_raw", "Class": "class_raw"})
        df["binary_label"] = (df["class_raw"].str.lower() != "benign").astype(int)
        df["family"] = df["category_raw"].apply(
            lambda x: "Benign" if str(x).lower().startswith("benign") else str(x).split("-")[0]
        )
        df["category"] = df["category_raw"]
    else:
        # synthetic path: only a 'category' column exists
        df["binary_label"] = (~df["category"].str.lower().str.startswith("benign")).astype(int)
        df["family"] = df["category"].apply(
            lambda x: "Benign" if str(x).lower().startswith("benign") else str(x).split("-")[0]
        )
    df["family_label"] = df["family"].map(FAMILY_LABEL_MAP)
    return df


def generate_synthetic_dataset(n_samples: int = 4000, seed: int = 42) -> pd.DataFrame:
    """
    Synthetic stand-in with the same column schema and plausible correlation
    structure, so graph_builder.py / model.py / train.py can be developed
    and unit-tested before you drop in the real CSV.

    NOT for reporting real results — swap in load_real_dataset() for that.
    """
    rng = np.random.default_rng(seed)
    n_benign = n_samples // 2
    n_malicious = n_samples - n_benign

    families = ["Trojan", "Spyware", "Ransomware"]
    rows = []

    def sample_group(base_mean, base_std, malicious, mal_shift):
        mean = base_mean + (mal_shift if malicious else 0)
        return max(0, rng.normal(mean, base_std))

    for i in range(n_samples):
        malicious = i >= n_benign
        row = {}
        # Benign processes: modest handle/DLL counts, near-zero injections
        row["pslist.nproc"] = sample_group(45, 8, malicious, 15)
        row["pslist.nppid"] = sample_group(40, 7, malicious, 10)
        row["pslist.avg_threads"] = sample_group(9, 2, malicious, 3)
        row["pslist.nprocs64bit"] = sample_group(40, 6, malicious, 5)
        row["pslist.avg_handlers"] = sample_group(150, 30, malicious, 40)

        row["dlllist.ndlls"] = sample_group(900, 100, malicious, 150)
        row["dlllist.avg_dlls_per_proc"] = sample_group(20, 4, malicious, -3)

        row["handles.nhandles"] = sample_group(9000, 1200, malicious, 2500)
        row["handles.avg_handles_per_proc"] = sample_group(200, 30, malicious, 60)
        row["handles.nport"] = sample_group(5, 2, malicious, 4)
        row["handles.nfile"] = sample_group(300, 50, malicious, 30)
        row["handles.nevent"] = sample_group(1200, 150, malicious, 100)
        row["handles.ndesktop"] = sample_group(3, 1, malicious, 0)
        row["handles.nkey"] = sample_group(400, 60, malicious, 50)
        row["handles.nthread"] = sample_group(600, 80, malicious, 120)
        row["handles.ndirectory"] = sample_group(60, 10, malicious, 5)
        row["handles.nsemaphore"] = sample_group(200, 30, malicious, 20)
        row["handles.ntimer"] = sample_group(40, 8, malicious, 10)
        row["handles.nsection"] = sample_group(500, 70, malicious, 150)
        row["handles.nmutant"] = sample_group(150, 25, malicious, 30)

        # Fileless-malware fingerprint: unlinked / hidden modules
        row["ldrmodules.not_in_load"] = sample_group(0.5, 0.5, malicious, 3.0)
        row["ldrmodules.not_in_init"] = sample_group(0.5, 0.5, malicious, 3.5)
        row["ldrmodules.not_in_mem"] = sample_group(0.3, 0.4, malicious, 4.0)
        row["ldrmodules.not_in_load_avg"] = sample_group(0.01, 0.01, malicious, 0.08)
        row["ldrmodules.not_in_init_avg"] = sample_group(0.01, 0.01, malicious, 0.09)
        row["ldrmodules.not_in_mem_avg"] = sample_group(0.01, 0.01, malicious, 0.10)

        # malfind: RWX / injected pages -> strong fileless indicator
        row["malfind.ninjections"] = sample_group(0.2, 0.4, malicious, 3.5)
        row["malfind.commitCharge"] = sample_group(500, 100, malicious, 800)
        row["malfind.protection"] = sample_group(1, 0.5, malicious, 4)
        row["malfind.uniqueInjections"] = sample_group(0.1, 0.3, malicious, 2.5)

        for f in FEATURE_GROUPS["psxview"]:
            row[f] = sample_group(0.2, 0.3, malicious, 1.5)

        row["modules.nmodules"] = sample_group(140, 20, malicious, -20)

        row["svcscan.nservices"] = sample_group(220, 25, malicious, 15)
        row["svcscan.kernel_drivers"] = sample_group(180, 20, malicious, 5)
        row["svcscan.fs_drivers"] = sample_group(25, 5, malicious, 2)
        row["svcscan.process_services"] = sample_group(15, 4, malicious, 5)
        row["svcscan.shared_process_services"] = sample_group(60, 10, malicious, 5)
        row["svcscan.interactive_process_services"] = sample_group(1, 1, malicious, 1)
        row["svcscan.nactive"] = sample_group(150, 20, malicious, 10)

        row["callbacks.ncallbacks"] = sample_group(90, 15, malicious, 10)
        row["callbacks.nanonymous"] = sample_group(2, 1, malicious, 8)
        row["callbacks.ngeneric"] = sample_group(85, 12, malicious, 5)

        if malicious:
            fam = families[(i - n_benign) % len(families)]
            row["category"] = f"{fam}-Synthetic{(i % 5) + 1}"
        else:
            row["category"] = "Benign"

        rows.append(row)

    df = pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return _postprocess(df)


DEMO_SAMPLE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "demo_sample.csv")


def load_dataset(prefer_real: bool = True) -> pd.DataFrame:
    """Try the real CIC-MalMem-2022 CSV first.

    If it's missing (e.g. a public deployment that doesn't bundle the full
    58,596-row dataset -- see README for why), fall back to a small bundled
    sample of REAL, already-labeled rows (data/demo_sample.csv, 300 rows
    drawn from the same dataset) so the app still works end-to-end on real
    data. Only falls back to fully synthetic data if even that is missing.
    """
    if prefer_real and os.path.exists(RAW_CSV_PATH):
        return load_real_dataset()
    if os.path.exists(DEMO_SAMPLE_PATH):
        print(
            "[INFO] Full CIC-MalMem-2022.csv not found -- using the bundled "
            "300-row demo sample (real data, not synthetic) from data/demo_sample.csv. "
            "Download the full dataset (see README) to train/evaluate on all 58,596 rows."
        )
        df = pd.read_csv(DEMO_SAMPLE_PATH)
        # demo_sample.csv is already post-processed (it's a slice of a
        # processed split), but re-run it defensively in case its columns
        # ever drift from _postprocess's expected output.
        if "binary_label" not in df.columns:
            df = _postprocess(df)
        return df
    print(
        "[WARN] Real CIC-MalMem-2022.csv not found at data/, and no bundled demo sample either. "
        "Using synthetic data for pipeline development only.\n"
        "       Download the real dataset before reporting any results."
    )
    return generate_synthetic_dataset()


if __name__ == "__main__":
    df = load_dataset()
    print(df.shape)
    print(df["category"].value_counts())
    print(df[["binary_label", "family"]].head())
