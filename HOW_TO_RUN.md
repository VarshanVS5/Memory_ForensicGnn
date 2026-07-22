# How This Project Works (Plain-English Guide)

## What this project actually does

It's a malware detector — like a spam filter, but for malware hiding in a computer's memory instead of email.

We give it a big spreadsheet of **58,596 already-solved examples**. Each row is one memory scan from a real computer, with ~55 measurements (things like "how many processes were running," "how many hidden/suspicious code injections were found"). Each row already has the correct answer attached: **Benign** (clean) or **Malicious** (infected).

We show the model most of these rows *with* the answers, so it learns the pattern. Then we hide the answers on the rest and ask it to guess. If it guesses right almost every time, the model works.

## The input

**One file, already sitting in the project — you don't touch it or type anything into it:**
```
data/CIC-MalMem-2022.csv
```
58,596 rows of memory-scan measurements, each labeled Benign or Malicious. That's the only input the training step needs.

## The output

After training, you get:
1. **A score printed in your terminal** — e.g. "99.9% accuracy," and exactly how many malware samples it missed out of ~5,860.
2. **A saved trained model** — `outputs/sage_model.pt`.
3. **Saved train/test spreadsheets** — `data/processed/train.csv` and `test.csv` — the exact rows it studied vs. was quizzed on, viewable in Excel.

## Step 1 — Train it and see the score

Open a terminal (PowerShell), go to the project folder, then run:

```bash
cd memforensics_gnn
.\RUN_ME.bat
```

**Important:** in PowerShell you must type `.\RUN_ME.bat`, not just `RUN_ME.bat`, or it won't be found.

Wait about a minute. It will:
- Train the GNN model (~30 seconds)
- Run two comparison models, RandomForest and GradientBoosting (~30-60 seconds)
- Print all the accuracy scores
- Say "ALL DONE" at the end

Nothing to type in — just watch it run.

## Step 2 — Test it on ONE sample yourself (this is where you enter input)

After Step 1 finishes, in the same terminal:

```bash
cd src
python predict.py
```

This picks a random row and shows you what the model thinks vs. the real answer:
```
Sample source: test.csv, row #42
Model prediction: MALICIOUS   (confidence: 98.7% malicious)
Actual answer:    MALICIOUS
--> Model was CORRECT
```

**To pick a specific one yourself** (this is your input — any number from 0 to 11719):
```bash
python predict.py --index 42
python predict.py --index 500
python predict.py --index 10000
```

Every different number = a different real memory scan = a fresh prediction.

## Step 3 — See it in the interactive app

This is the fun one: a web page where you pick a sample (or upload your own),
see the model's Benign/Malicious call next to a RandomForest's call, and
watch which artifact relationship (e.g. `malfind <-> ldrmodules`) the model
paid attention to — drawn as the actual investigation graph.

```bash
cd src
streamlit run app.py
```

It opens in your browser automatically. Use the sidebar to switch between
the GraphSAGE/GAT model, draw a random test sample, jump to a specific row,
or upload your own CSV row.

## Step 4 (optional) — Test it on malware it's never seen

Everything above is tested on rows held back from the *same* malware
subfamilies the model trained on — the easy version of the test. To see how
it does on malware **variants it has literally never trained on**:

```bash
cd src
python generalization_test.py
```

This holds out one subfamily per malware family (Ransomware-Pysa,
Spyware-TIBS, Trojan-Reconyc) completely from training, then reports
accuracy on those unseen variants next to the normal in-distribution
accuracy. Once this has run, the Streamlit app's top panel automatically
shows both numbers side by side.

## Step 5 (optional, advanced) — Scan a REAL memory dump, not a CSV row

Everything above scores a row from the pre-built spreadsheet. This step instead
takes an actual `.raw`/`.vmem` memory image (a real snapshot of a real
computer's RAM) and scores that.

```bash
pip install volatility3 volmemlyzer
cd src
python predict.py --dump path\to\your\image.raw
```

Or use the "Live memory dump" option in the Streamlit app's sidebar.

**Read this before trusting the result.** This was tested end-to-end against
a real ~15.6GB live RAM capture (not a toy file) and here's what happened,
honestly:

- Almost every Volatility3 plugin except the most basic one (`pslist`) hung
  indefinitely when run against a real, large, busy machine's memory — a
  real limitation of the underlying tool at this scale, not a bug in this
  project. So today, only 2 of the 55 measurements (`pslist.nproc`,
  `pslist.nppid`) actually come from the real image; everything else is
  filled in with the training data's average value.
- With that little real data, the model called a genuinely clean test
  machine "MALICIOUS" at 99.5% confidence — a false positive. One real,
  unusually-high number (a lot of running processes) combined with a pile of
  "average" filler values fooled it into a confident wrong answer.
- Because of this, the app now shows **"NOT RELIABLE"** instead of a
  confidence number whenever fewer than 6 of the 55 features are real
  measurements — so a live scan won't quietly show you a number you
  shouldn't trust.

**In short:** this feature proves the plumbing works (real memory dump in,
real model verdict out), but isn't accurate enough yet to actually trust for
detecting real malware. That's an honest, useful thing to say in a report —
just don't present a live-scan verdict as a working detector without this
caveat attached.

## Recap — the whole thing in 5 commands

```bash
cd memforensics_gnn
.\RUN_ME.bat
cd src
python predict.py
python generalization_test.py
streamlit run app.py
```

## If something goes wrong

- **"RUN_ME.bat is not recognized"** → you forgot the `.\` in front, or you're in the wrong folder. Run `cd memforensics_gnn` first.
- **"No trained model" error from predict.py** → you skipped Step 1. Run `.\RUN_ME.bat` first, it creates the model file predict.py needs.
- **Step 1 seems stuck** → it isn't. Watch for new "Epoch" lines appearing every few seconds — as long as those keep showing up, it's working.
