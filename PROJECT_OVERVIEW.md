# Memory Forensics GNN — Project Overview

*(Simple explanation you can use to describe this project to anyone — mentor, interviewer, resume reader.)*

---

## 1. Problem Statement

Traditional antivirus works by scanning **files on disk** for known malware signatures. But modern malware increasingly runs **"fileless"** — it lives entirely in a computer's RAM (memory) and never writes a file to disk. Since there's no file to scan, file-based antivirus simply doesn't see it.

The only way to catch fileless malware is to look at a **memory dump** — a snapshot of everything running in RAM at a moment in time — and look for behavioral fingerprints of malicious activity (hidden processes, injected code, mismatched module lists, etc.). Doing this manually is slow, requires an expert forensic analyst, and doesn't scale.

**The problem:** can we automatically classify a memory dump as infected or clean, using the kind of evidence a forensic analyst would look at?

## 2. The Solution

Build a machine learning model that takes memory-forensics measurements (extracted by a tool called Volatility) and predicts **Benign vs. Malicious** — and where possible, explains *why* it made that call, the way a human analyst would justify a finding.

Two things make this more than "just run a classifier":

1. **A Graph Neural Network (GNN), not a flat model.** Instead of treating the ~55 measurements as one flat list of numbers, they're grouped into 9 "artifact types" (process list, DLL list, handles, loaded modules, injected memory regions, etc.) and connected as a graph — edges represent real relationships a forensic analyst reasons about (e.g. *"if the DLL list and the loader-list disagree, something is hiding a module"*). This lets the model learn from relationships between artifacts, not just raw counts.
2. **A tabular baseline run alongside it (RandomForest, GradientBoosting).** This is the honesty check: does the GNN's extra complexity actually buy anything over a standard, much simpler model? (See Results — the honest answer is nuanced, and that's a feature of the project, not a weakness.)

## 3. How It Works (the pipeline)

```
CSV of memory-scan data
        |
        v
[1] Load + label data  (data_loader.py)
        |
        v
[2] Turn each row into a small graph  (graph_builder.py)
      9 nodes (artifact types) + 11 relationship edges
        |
        v
[3] GNN learns to classify the whole graph  (model.py + train.py)
      Benign  or  Malicious
        |
        v
[4] Compared against RandomForest / GradientBoosting  (baseline.py)
        |
        v
[5] (Optional) Explain which artifact relationship drove the decision  (explain.py)
```

## 4. Input & Output

| | Details |
|---|---|
| **Input** | CIC-MalMem-2022 dataset — 58,596 real memory-dump snapshots, ~55 forensic measurements each, pre-labeled Benign/Malicious by researchers |
| **Training output** | A trained model + accuracy/precision/recall/F1/false-negative-rate metrics |
| **Per-sample output** (`predict.py`) | For one memory scan: a **Benign/Malicious verdict** + a confidence percentage |
| **Explainability output** (`explain.py`) | For a flagged sample: which artifact-relationship (e.g. `malfind ↔ ldrmodules`) the model paid the most attention to |

## 5. Results (real numbers, on the real dataset)

| Model | Accuracy | Missed malware (False Negatives) | False Alarms (False Positives) |
|---|---|---|---|
| GraphSAGE (GNN) | 99.9% | 2–5 out of 5,860 | 9–10 out of 5,860 |
| Random Forest | 99.99% | 0 out of 5,860 | 1 out of 5,860 |
| Gradient Boosting | 99.98% | 1 out of 5,860 | 0 out of 5,860 |

**Family classification (Benign/Trojan/Spyware/Ransomware) — the non-saturated task:**

| Model | Accuracy | Macro F1 |
|---|---|---|
| GraphSAGE (GNN) | 77.0% | 0.65 |
| Random Forest | 87.1% | 0.81 |

**Cross-family generalization (accuracy on malware subfamilies never seen in training):**

| | Accuracy | Missed malware |
|---|---|---|
| Seen subfamilies (in-distribution) | 99.88% | 0.08% |
| Unseen subfamilies | 99.17% | 0.83% |

## 6. Pros

- **Real dataset, real results** — not synthetic/toy data, 58,596 genuine samples from a peer-reviewed cybersecurity research dataset.
- **Interpretable by design** — the graph structure encodes actual forensic reasoning (artifact relationships), not a black box; the GAT variant can point to *which* relationship drove a flag.
- **Honest benchmarking** — it doesn't just claim "GNN is better," it directly compares against simpler baselines and reports the true picture.
- **Extremely low miss rate** — in malware detection, missing real malware (false negative) is the costly mistake, and all models here miss almost nothing.
- **Fast to run and reproduce** — full pipeline trains in under a minute on a normal laptop, no GPU needed.

## 7. Cons / Limitations

- **The GNN doesn't clearly beat a plain RandomForest on this task.** Binary benign/malicious accuracy is saturated near 100% for every method tried — there isn't enough headroom left for the GNN's relationship-modeling to show an advantage. This is a known, documented property of this specific dataset, not a flaw in the approach — but it means the "GNN vs. baseline" story is currently a wash, not a win.
- **One row = one whole-system snapshot, not one process.** The graph is a *semantic* graph of artifact-type relationships, not a true process-interaction graph (which would need raw memory dumps + Volatility's per-process output, not this pre-aggregated CSV). It's a reasonable, defensible first version — not the strongest possible version of the idea.
- ~~Only binary classification is done so far~~ — addressed: `src/train_multiclass.py` trains on Benign/Trojan/Spyware/Ransomware. Real result, and it's not the hoped-for one: RandomForest clearly **beats** the GNN here (87.1% accuracy / 0.81 macro-F1 vs. the GNN's 77.0% / 0.65 macro-F1). The task is genuinely harder than binary (Trojan/Spyware/Ransomware get confused with each other) and not saturated, but with this architecture and training budget the GNN doesn't close the gap — worth reporting exactly as-is rather than only running the version that flatters the approach. Plausible next steps if you want to chase this further: more epochs, a deeper/wider GNN, or accept that per-sample "snapshot" graphs (see next point) may just not carry enough family-distinguishing structure for message passing to exploit.
- **Live real-time scanning works end-to-end; reliability on real hardware was tested, one root cause found and fixed, and re-validation is still needed.** `src/live_extract.py` was validated against a real ~15.6GB live RAM capture (via WinPmem) of an actual machine, not a toy image. What happened, and what's changed since:
  - **Volatility 3 plugin reliability on real (large, actively-used) images is a genuine open problem.** Every plugin except `windows.pslist` hung indefinitely at some point across repeated runs (`handles`, `ldrmodules`, `malfind`, `callbacks`) — non-deterministically, a different plugin each run — while `pslist` alone completed reliably every time. This is almost certainly resource contention against a 15+GB file, not one broken plugin.
  - **The original extraction bug: one shared timeout for all plugins.** All plugins were requested from VolMemLyzer in a single subprocess call, and its internal `--timeout` didn't reliably abort a hang — so one hanging plugin killed the whole call and discarded results from plugins that had already finished. That's why the extractor was temporarily hard-coded down to `pslist` only: it was cheaper to skip 41 features than risk losing all of them.
  - **Fixed:** `live_extract.py` now runs every plugin (`pslist`, `dlllist`, `handles`, `ldrmodules`, `malfind`, `modules`, `svcscan`, `callbacks`) as its **own subprocess with its own hard OS-level timeout**. A hang in `malfind` can no longer take down `pslist`/`dlllist`/`handles`/etc — each plugin independently succeeds or is marked failed-and-imputed. This doesn't make Volatility 3 itself faster (that's an upstream limitation), but it means an extraction now actually *attempts* real values for all 8 plugins instead of forfeiting 7 of them by default.
  - **A real extractor bug was also caught, separately.** VolMemLyzer returned `pslist.avg_threads = 4,477,164` on the real image — physically impossible (training data tops out at 16.8). `live_extract.py` sanity-checks every live value against the training distribution (10x margin) and rejects/imputes anything implausible, reporting it as such.
  - **The original end-to-end verdict (before the per-plugin timeout fix) was MALICIOUS at 99.5% confidence — a false positive** (the test machine was not actually infected), with only 2/55 features genuinely real. `app.py` shows "NOT RELIABLE" instead of a number whenever fewer than 6/55 features are real (`MIN_REAL_FEATURES_FOR_TRUST`), regardless of how many plugins succeed.
  - **Not yet re-tested:** the per-plugin-timeout fix has not been re-validated against a real large image since it was written — doing so requires a real memory capture and Volatility3/VolMemLyzer installed locally, which isn't available in this environment. The expectation is more of the 8 plugins now complete (more real features, less imputation), but this needs confirming on real hardware before claiming the false-positive is resolved.
- ~~No test of whether the model generalizes to malware it hasn't seen~~ — addressed: `src/generalization_test.py` holds out one subfamily per malware family entirely from training and reports accuracy on those unseen variants. Real result: accuracy drops from 99.88% (seen) to 99.17% (unseen), and the missed-malware rate roughly 10x's (0.08% -> 0.83%) — a small but genuine, honestly-reported generalization gap.

## 8. What Would Make This Stronger (future work)

1. ~~Multiclass family classification~~ — done, see above; the GNN did not win, which is itself a useful, reportable finding.
2. **True per-process graphs** from raw Volatility JSON output instead of the aggregated CSV — a bigger scope change, but the "real" version of this idea (similar to academic work like DeepMem), and plausibly what's needed to actually beat RandomForest on family classification.
3. ~~A live demo path~~ — built and validated end-to-end against a real machine's live memory (`src/live_extract.py`); works, but currently unreliable due to Volatility 3 plugin performance on large real images — see the limitation above for the honest breakdown and what would fix it.
4. ~~An interactive frontend~~ — done: `src/app.py` (Streamlit) lets you feed the model a sample (including a real memory dump), compare it against the RandomForest baseline, and see the GAT attention graph live.
