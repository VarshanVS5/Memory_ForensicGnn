"""
Live memory-dump feature extraction.

Runs VolMemLyzer (a Volatility 3 wrapper, `pip install volmemlyzer`) against
a REAL raw memory image (.raw/.vmem/.dmp/...) and maps its output onto the
same 55-feature schema CIC-MalMem-2022 uses, so the trained GNN/RandomForest
can score an actual live/fresh memory dump instead of only pre-extracted CSV
rows.

Honesty note -- read before trusting a live verdict:
Volatility 3's psxview plugin does not preserve the multi-cross-view detail
Volatility 2 had (the 7 "is this process visible via pslist / eprocess_pool /
ethread_pool / pspcid_list / csrss_handles / session / deskthrd" checks and
their _false_avg companions -- 14 of our 55 features total). There is no live
equivalent for those today, short of reimplementing that plugin from scratch.
They are imputed from the TRAINING SET MEAN and every extraction reports
exactly which features are real vs. imputed, so nobody downstream mistakes
an approximate verdict for a fully-measured one.

Everything else (pslist, dlllist, handles, ldrmodules, malfind, modules,
most of svcscan/callbacks -- 41 of 55 features) maps onto a real,
freshly-computed Volatility 3 plugin result.
"""

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from data_loader import FEATURE_GROUPS

ALL_FEATURES = [c for group in FEATURE_GROUPS.values() for c in group]

_REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
PROCESSED_DIR = os.path.join(_REPO_ROOT, "data", "processed")

# IMPORTANT, hard-won finding from testing against a real ~16GB live capture
# of an actively-used machine (not a small curated test image): almost every
# Volatility3 plugin except windows.pslist hung indefinitely at some point
# across repeated runs -- callbacks, malfind, ldrmodules, and even handles
# each hung on at least one run (0 CPU for 10-15+ min while other plugins on
# the SAME image completed in under a minute), and which plugin hung was NOT
# consistent run to run. VolMemLyzer's own --timeout flag did not reliably
# abort them. This looks like resource contention (repeatedly scanning a
# 15.6GB file that doesn't comfortably fit alongside everything else running
# on the machine) rather than one specific broken plugin.
#
# Root cause of the original bug: all plugins were requested from VolMemLyzer
# in a SINGLE subprocess call sharing one timeout. Since VolMemLyzer's
# internal --timeout was unreliable, a hang in any one plugin meant killing
# the whole call -- discarding results from plugins that had already
# finished. That's why PLUGINS_NEEDED used to be hardcoded down to just
# "pslist": it was cheaper to skip 41 features than risk losing all of them.
#
# The actual fix: run every plugin as its OWN subprocess with its OWN hard
# OS-level timeout (see _run_volmemlyzer below). A hang in malfind no longer
# costs pslist/dlllist/handles/ldrmodules/etc -- each plugin either succeeds
# or is individually marked failed-and-imputed. This does not make
# Volatility3 itself faster or hang-proof (that's an upstream problem), but
# it means a live extraction now actually attempts -- and usually gets --
# real values for every plugin instead of forfeiting all but one by default.
PLUGINS_NEEDED = [
    "pslist", "dlllist", "handles", "ldrmodules",
    "malfind", "modules", "svcscan", "callbacks",
]
PLUGIN_TIMEOUT_SECONDS = 600

# our_schema_name -> volmemlyzer_name, for features with a direct 1:1 (or
# near-identical-definition) equivalent in VolMemLyzer's Volatility-3-based
# extractors. See src/live_extract.py module docstring for what's excluded.
DIRECT_FEATURE_MAP = {
    "pslist.nproc": "pslist.nproc",
    "pslist.nppid": "pslist.nppid",
    "pslist.avg_threads": "pslist.avg_threads",
    "pslist.nprocs64bit": "pslist.nprocs64bit",
    "pslist.avg_handlers": "pslist.avg_handlers",

    "dlllist.ndlls": "dlllist.ndlls",
    "dlllist.avg_dlls_per_proc": "dlllist.avg_dllPerProc",

    "handles.nhandles": "handles.nHandles",
    "handles.avg_handles_per_proc": "handles.avgHandles_per_proc",
    "handles.nport": "handles.nTypePort",
    "handles.nfile": "handles.nTypeFile",
    "handles.nevent": "handles.nTypeEvent",
    "handles.ndesktop": "handles.nTypeDesk",
    "handles.nkey": "handles.nTypeKey",
    "handles.nthread": "handles.nTypeThread",
    "handles.ndirectory": "handles.nTypeDir",
    "handles.nsemaphore": "handles.nTypeSemaph",
    "handles.ntimer": "handles.nTypeTimer",
    "handles.nsection": "handles.nTypeSec",
    "handles.nmutant": "handles.nTypeMutant",

    "ldrmodules.not_in_load": "ldrmodules.not_in_load",
    "ldrmodules.not_in_init": "ldrmodules.not_in_init",
    "ldrmodules.not_in_mem": "ldrmodules.not_in_mem",
    "ldrmodules.not_in_load_avg": "ldrmodules.not_in_load_avg",
    "ldrmodules.not_in_init_avg": "ldrmodules.not_in_init_avg",
    "ldrmodules.not_in_mem_avg": "ldrmodules.not_in_mem_avg",

    "malfind.ninjections": "malfind.ninjections",
    "malfind.commitCharge": "malfind.commitCharge",
    "malfind.protection": "malfind.protection",
    "malfind.uniqueInjections": "malfind.uniqueInjections",

    "modules.nmodules": "modules.nModules",

    "svcscan.nservices": "svcscan.nServices",
    "svcscan.nactive": "svcscan.State_Run",
    "svcscan.kernel_drivers": "svcscan.Type_Kernel_Driver",
    "svcscan.fs_drivers": "svcscan.Type_FileSys_Driver",
    "svcscan.process_services": "svcscan.Type_Own",

    "callbacks.ncallbacks": "callbacks.ncallbacks",
}

# Features computed from a small SUM of VolMemLyzer columns (no 1:1 rename).
# Each entry: (source column names it depends on, combining function).
COMPUTED_FEATURES = {
    "svcscan.shared_process_services": (
        ["svcscan.Type_Own_Share", "svcscan.Type_Share"],
        lambda f: f.get("svcscan.Type_Own_Share", 0) + f.get("svcscan.Type_Share", 0),
    ),
    "svcscan.interactive_process_services": (
        ["svcscan.Type_Own_Interactive", "svcscan.Type_Share_Interactive"],
        lambda f: f.get("svcscan.Type_Own_Interactive", 0) + f.get("svcscan.Type_Share_Interactive", 0),
    ),
    "callbacks.nanonymous": (["callbacks.noSymbol"], lambda f: f.get("callbacks.noSymbol", 0)),
    "callbacks.ngeneric": (["callbacks.genericKernel"], lambda f: f.get("callbacks.genericKernel", 0)),
}

# No live equivalent -- Volatility 3's psxview plugin dropped the per-view
# granularity these depend on. Imputed from the training set mean.
IMPUTED_FEATURES = FEATURE_GROUPS["psxview"]


@dataclass
class ExtractionReport:
    real: list = field(default_factory=list)
    imputed: list = field(default_factory=list)
    rejected: list = field(default_factory=list)  # "real" but implausible -> imputed instead
    extreme: list = field(default_factory=list)  # real AND accepted, but outside the training min/max
    raw_volmemlyzer_features: dict = field(default_factory=dict)


def _training_stats() -> pd.DataFrame:
    train_path = os.path.join(PROCESSED_DIR, "train.csv")
    if os.path.exists(train_path):
        df = pd.read_csv(train_path)
    else:
        from data_loader import load_dataset
        df = load_dataset()
    return df[ALL_FEATURES].agg(["mean", "min", "max"])


def _is_plausible(name: str, value: float, stats: pd.DataFrame) -> bool:
    """Reject values that are extractor bugs, not real signal.

    VolMemLyzer has been observed to occasionally return nonsensical values
    on real (non-toy) images -- e.g. pslist.avg_threads coming back as
    millions, a physical impossibility (training data tops out around 17).
    A live-measured value is only trusted if it's within a generous 10x
    margin of the training set's observed range -- wide enough to allow a
    real busy machine to legitimately exceed the training VMs (e.g. more
    processes than any training sample), but tight enough to catch
    obviously-broken extractor output.
    """
    if value is None or not np.isfinite(value):
        return False
    lo, hi = stats.loc["min", name], stats.loc["max", name]
    margin = max(abs(hi), 1.0) * 10
    return (min(0, lo * 10) - margin) <= value <= (hi * 10 + margin)


def _is_outside_training_range(name: str, value: float, stats: pd.DataFrame) -> bool:
    """A NARROWER check than _is_plausible: is this accepted real value simply
    outside anything seen in training (even though it passed the generous 10x
    plausibility check)? This is exactly the pattern that produced the real
    false positive during hardware testing -- pslist.nproc=484 on a machine
    with training max=240 was "plausible" (not an extractor bug) but still an
    outlier the model had never seen, and combined with mostly-imputed
    features it drove a confident wrong verdict. Flagging this separately
    from rejected/imputed lets callers warn about that specific combination
    even when the value is real and correctly accepted."""
    lo, hi = stats.loc["min", name], stats.loc["max", name]
    return value < lo or value > hi


def _run_one_plugin(exe: str, dump_path: str, workdir: str, plugin: str) -> dict:
    """Run a single VolMemLyzer plugin in its own subprocess with its own
    hard timeout, so a hang here can't take down the other plugins' results.
    Returns {} (not raises) on timeout or failure -- the caller treats a
    missing feature the same as any other unavailable one: impute + report."""
    plugin_dir = os.path.join(workdir, plugin)
    os.makedirs(plugin_dir, exist_ok=True)
    cmd = [
        exe,
        "--jobs", "1",
        "--timeout", str(PLUGIN_TIMEOUT_SECONDS),
        "extract", "-i", dump_path, "-o", plugin_dir, "-f", "json",
        "--plugins", plugin,
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=PLUGIN_TIMEOUT_SECONDS + 60,
        )
    except subprocess.TimeoutExpired:
        print(f"[live_extract] plugin '{plugin}' timed out after "
              f"{PLUGIN_TIMEOUT_SECONDS + 60}s -- skipping, will impute its features.")
        return {}

    if result.returncode != 0:
        print(f"[live_extract] plugin '{plugin}' failed (exit {result.returncode}) "
              f"-- skipping, will impute its features.\nstderr: {result.stderr[-500:]}")
        return {}

    feature_dir = os.path.join(plugin_dir, "features")
    if not os.path.isdir(feature_dir):
        print(f"[live_extract] plugin '{plugin}' produced no output -- skipping.")
        return {}
    files = [f for f in os.listdir(feature_dir) if f.endswith(".json")]
    if not files:
        print(f"[live_extract] plugin '{plugin}' produced no feature file -- skipping.")
        return {}
    with open(os.path.join(feature_dir, files[0])) as f:
        payload = json.load(f)
    return payload.get("features", {})


def _run_volmemlyzer(dump_path: str, workdir: str) -> dict:
    import shutil
    exe = shutil.which("volmemlyzer")
    if exe is None:
        raise RuntimeError("volmemlyzer executable not found on PATH. Install with: pip install volmemlyzer")

    merged: dict = {}
    for plugin in PLUGINS_NEEDED:
        merged.update(_run_one_plugin(exe, dump_path, workdir, plugin))

    if not merged:
        raise RuntimeError(
            "Every Volatility3 plugin failed or timed out on this image -- "
            "no live features were extracted (all will be imputed)."
        )
    return merged


def extract_features_from_dump(dump_path: str) -> tuple[pd.Series, ExtractionReport]:
    """Run Volatility3 (via VolMemLyzer) on a real memory dump and return a
    row matching data_loader.ALL_FEATURES, plus a report of which features
    are real measurements vs. imputed from the training-set mean."""
    if not os.path.exists(dump_path):
        raise FileNotFoundError(dump_path)

    with tempfile.TemporaryDirectory(prefix="volmemlyzer_") as workdir:
        raw = _run_volmemlyzer(dump_path, workdir)

    stats = _training_stats()
    means = stats.loc["mean"]
    row = {}
    report = ExtractionReport(raw_volmemlyzer_features=raw)

    def _accept_or_impute(our_name, candidate_value):
        if candidate_value is not None and _is_plausible(our_name, candidate_value, stats):
            row[our_name] = float(candidate_value)
            report.real.append(our_name)
            if _is_outside_training_range(our_name, candidate_value, stats):
                report.extreme.append(our_name)
        else:
            row[our_name] = float(means[our_name])
            if candidate_value is not None:
                report.rejected.append(our_name)
            report.imputed.append(our_name)

    for our_name, vml_name in DIRECT_FEATURE_MAP.items():
        candidate = raw.get(vml_name)
        _accept_or_impute(our_name, float(candidate) if candidate is not None else None)

    for our_name, (source_cols, fn) in COMPUTED_FEATURES.items():
        candidate = fn(raw) if any(c in raw and raw[c] is not None for c in source_cols) else None
        _accept_or_impute(our_name, candidate)

    for our_name in IMPUTED_FEATURES:
        row[our_name] = float(means[our_name])
        report.imputed.append(our_name)

    # Anything in our schema this mapping forgot -- fall back to the mean
    # rather than KeyError, and make sure it shows up in the report.
    for our_name in ALL_FEATURES:
        if our_name not in row:
            row[our_name] = float(means[our_name])
            report.imputed.append(our_name)

    return pd.Series(row)[ALL_FEATURES], report


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Extract CIC-MalMem-2022-schema features from a real memory dump.")
    ap.add_argument("dump", help="Path to a raw memory image (.raw/.vmem/.dmp/...)")
    args = ap.parse_args()

    row, report = extract_features_from_dump(args.dump)
    print(f"\nReal (live-measured) features: {len(report.real)}/{len(ALL_FEATURES)}")
    print(f"Imputed (training-mean) features: {len(report.imputed)}/{len(ALL_FEATURES)}")
    if report.rejected:
        print(f"Rejected as implausible (extractor bug, imputed instead): {len(report.rejected)}")
        print("Rejected:", ", ".join(report.rejected))
    if report.extreme:
        print(f"\nWARNING: {len(report.extreme)} real feature(s) are outside anything seen in "
              f"training: {', '.join(report.extreme)}")
        if len(report.real) - len(report.extreme) < 6:
            print("This is the exact combination (few real features, one an outlier) that "
                  "produced a confirmed false positive during hardware testing -- treat this "
                  "verdict with real skepticism regardless of the confidence number shown.")
    print("Imputed:", ", ".join(report.imputed))
    print("\nFeature row:")
    print(row)
