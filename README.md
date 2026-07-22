# Memory Forensics for Fileless Malware Detection using GNNs

## Status
Full pipeline built and tested end-to-end (data loading -> graph construction
-> GraphSAGE/GAT training -> evaluation), running on the **real
CIC-MalMem-2022 dataset** (58,596 rows) already checked into `data/`. A
cross-family generalization test and an interactive Streamlit frontend are
included as well.

## Step 1 — Install dependencies
```bash
pip install torch torch_geometric pandas numpy scikit-learn networkx streamlit
```

## Step 2 — Run
```bash
cd src
python train.py --model sage --epochs 40   # GraphSAGE (recommended default)
python train.py --model gat  --epochs 40   # GAT (attention, interpretable)
python baseline.py                          # Random Forest + Gradient Boosting comparison
python explain.py                           # GAT attention-based explanations for malicious samples
python generalization_test.py               # Accuracy on malware SUBFAMILIES never seen in training
python train_multiclass.py --model sage     # Harder Benign/Trojan/Spyware/Ransomware task, GNN vs RandomForest
python predict.py --dump path/to/image.raw  # Score a REAL memory dump via live Volatility3 extraction
streamlit run app.py                        # Interactive frontend (verdicts, attention graph, baseline compare, live dumps)
```

## What's implemented

| File | Purpose |
|---|---|
| `src/data_loader.py` | Loads CIC-MalMem-2022, or generates schema-matched synthetic data for pipeline dev |
| `src/graph_builder.py` | Converts each tabular sample into a 9-node artifact graph (pslist, dlllist, handles, ldrmodules, malfind, psxview, modules, svcscan, callbacks) with investigation-motivated edges |
| `src/model.py` | `SAGEClassifier` and `GATClassifier` — graph-level classification via message passing + mean/max pooling |
| `src/train.py` | Train/val/test split (leak-free scaling), training loop, full metrics incl. false-negative rate |
| `src/baseline.py` | Random Forest + Gradient Boosting on the same features — run this to show whether the GNN actually beats flat tabular models |
| `src/explain.py` | Trains a GAT and prints, per malicious sample, which artifact-node relationships (e.g. `malfind <-> ldrmodules`) got the most attention — your "why did the model flag this" evidence |
| `src/generalization_test.py` | Holds out one malware subfamily per family (e.g. `Ransomware-Pysa`) entirely from training, then reports accuracy on those never-seen variants vs. an in-distribution test set — the honest generalization number, not just same-distribution accuracy |
| `src/train_multiclass.py` | Trains on the harder, non-saturated Benign/Trojan/Spyware/Ransomware task (vs. train.py's binary task, which every model hits ~99.9%+ on) and compares GNN vs. RandomForest here instead — this is the more honest place to look for a real GNN advantage |
| `src/live_extract.py` | Runs [VolMemLyzer](https://github.com/ahlashkari/VolMemLyzer) (Volatility 3 under the hood) against a **real raw memory dump** and maps its output onto our 55-feature schema — 41/55 features map directly, the other 14 (`psxview.*`) have no Volatility 3 equivalent and are imputed from the training mean, clearly flagged in the output. This is the real-time / live-scan path. |
| `src/predict.py` | CLI: score a single sample (random, by index, your own CSV row, or `--dump path/to/image.raw` for a real live memory dump) |
| `src/app.py` | Streamlit frontend: pick/upload a sample (or a real memory dump), see the GNN verdict vs. RandomForest baseline, and the live GAT attention-weighted artifact graph |

## Real-time / live memory dump scanning

`src/live_extract.py` + the "Live memory dump" option in `app.py` / `predict.py --dump` let you point the trained model at an actual `.raw`/`.vmem`/`.dmp` memory image instead of a pre-extracted CSV row. It requires `pip install volatility3 volmemlyzer` (already in `requirements`-equivalent above).

**Known limitation, stated plainly:** Volatility 3's `psxview` plugin does not preserve the 7-way cross-view detection Volatility 2 had, so 14 of the 55 original CIC-MalMem-2022 features have no live equivalent today and are imputed from the training-set mean rather than measured. `live_extract.py` reports exactly how many of the 55 features were real vs. imputed for every extraction — treat a verdict with a low real-feature count as unreliable, which the app surfaces as a warning/error banner automatically.

**Real-sample validation: done, against a real live machine — and it surfaced a genuine, important limitation.**

Public malware sample mirrors are all dead (Volatility Foundation wiki, archive.org, jstrosch/malware-samples' Google Drive links — all checked, all 404/dead as of this session). Instead, `tools/winpmem.exe` ([official Velocidex/WinPmem release](https://github.com/Velocidex/WinPmem)) was used to capture a real ~15.6GB live RAM snapshot of an actual machine, run through `live_extract.py`. Findings:

1. **Volatility3 plugin reliability on real (non-toy) images is a real problem.** Every plugin except `windows.pslist` hung indefinitely at some point across repeated runs — `handles`, `ldrmodules`, `malfind`, `callbacks` — non-deterministically (which one hung varied run to run). This looks like resource contention against a 15.6GB file rather than one broken plugin. This is a genuine finding worth stating plainly in a report: **plugin performance on small curated test images does not represent real-host performance.**
   - **Root cause found and fixed:** the original extractor requested all plugins in one subprocess call sharing a single timeout, so any one hang killed results from plugins that had already finished — which is why it was temporarily hard-coded down to `pslist` only. `live_extract.py` now runs each of the 8 plugins as its own subprocess with its own hard timeout, so a hang in one no longer costs the others. This hasn't been re-tested against a real large image yet (needs a real capture + Volatility3 locally, not available in this environment) — the expectation is more real features and less imputation, but that needs confirming on real hardware before treating the false-positive below as resolved.
2. **A real extractor bug was caught and fixed.** VolMemLyzer returned `pslist.avg_threads = 4,477,164` on this real image — physically impossible (training data tops out at 16.8). `live_extract.py` now sanity-checks every live-measured value against the training distribution (10x margin) before trusting it; implausible values are rejected and imputed instead, and reported as such.
3. **The end-to-end verdict on this real (non-infected) machine was MALICIOUS at 99.5% confidence — a false positive**, and the reason is itself informative: only 2/55 features were real (`pslist.nproc=484`, `pslist.nppid=102`), and `nproc` alone was a strong outlier vs. training (max 240 in CIC-MalMem-2022). A model fed mostly-imputed-average features plus one strong real outlier produced a confident wrong answer rather than an uncertain one. **This is the honest state of live real-time scanning today: functional end-to-end, but not reliable enough to trust its verdict when fewer than ~10% of features are genuinely measured.** Closing this gap needs either fixing Volatility3 plugin reliability on real hosts, or gating the app's verdict display behind a minimum real-feature-count threshold rather than always showing a confident percentage.

## Why graphs instead of just feeding the CSV to XGBoost
CIC-MalMem-2022 ships one flat row of Volatility-plugin features per memory
dump. The graph construction here is the actual contribution: it encodes the
**relationships forensic analysts reason about** (e.g. "does dlllist agree
with ldrmodules? if not, something is hiding a module" — a classic
fileless-malware fingerprint) as explicit edges, rather than letting a flat
model implicitly guess at feature interactions. That's the story to tell
your mentor: not "I ran a GNN because the project says GNN" but "the
graph structure encodes real forensic reasoning steps."

## Honest limitations to flag in your report
1. **CIC-MalMem-2022 is one row per whole-system snapshot**, not per-process.
   The 9-node graph here is a semantic/plugin-relationship graph, not a true
   process-interaction graph. This is a reasonable v1 and defensible in a
   report, but...
2. **Stronger version (if time allows):** build actual per-process graphs
   from raw memory dumps using Volatility 3's JSON output directly — nodes =
   individual processes, edges = parent/child + `malfind`-flagged injection
   relationships. This is what papers like DeepMem (Song et al., 2018) do
   and would be a meaningful step up in novelty. Volatility Foundation
   publishes free sample memory images (Cridex, etc.) to build/test this on
   before you need real malware samples.
3. A 2024 systematic review of this field notes only ~23% of studies use
   diverse public benchmarks and generalization across datasets is weak —
   so don't over-claim from CIC-MalMem-2022 alone; mention this as a
   limitation, it makes your report stronger, not weaker.

## Next steps once real data is in
1. Run `python train.py --model sage` and `--model gat`, compare F1/AUC/FNR
2. Do a family-level (multiclass: Trojan/Spyware/Ransomware) variant using
   the `family` column already computed in `data_loader.py`
3. For the GAT run, pull attention weights (`return_attention=True` in
   `GATClassifier.forward`) to show your mentor *which artifact type* drove
   each malicious prediction — good demo material
4. If time allows, build the per-process raw-dump graph pipeline (see
   limitation #2 above) as your "stretch" contribution
