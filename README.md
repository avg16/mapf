# Learning When to Search for Multi Agent Path Finding

Project members: Abhivansh Gupta (23112004) and Shivang Joshi (23117132), IIT Roorkee.

This project implements a local imitation policy, collision shield, PIBT, policy-guided CS-PIBT, LaCAM and a learned logistic gate. It evaluates whether selectively invoking coordination search reduces search calls while preserving solution success and quality.

The saved experiments contain 140 distinct test instances at seven agent counts and 2,500 total method/parameter runs. The hybrid with a LaCAM fallback solves all 140 tested instances and uses approximately 23–57% fewer external CS-PIBT calls than always-search. Plain LaCAM is faster on these maps. These findings do not establish a general runtime advantage or a universal success guarantee.

## Setup

Use Python 3.12. From this directory:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m hybrid_mapf.smoke_check
python -m hybrid_mapf.demo
```

On Windows activate with `.venv\Scripts\activate` and use the installed Python executable. PyTorch uses Apple MPS when available and otherwise CPU; CUDA is not selected automatically. A separate ngrok account or internet connection is not required for local use. Open http://127.0.0.1:8765 and stop the server with Ctrl+C. To choose another port, add `--port 8766`.

## Submission contents

- `report/project_report.tex`: editable report source; the PDF is in the submission folder.
- `report/contributions.tex` and `CONTRIBUTIONS.md`: individual contribution allocations, shared ideation and member-specific preparation/review roles.
- `hybrid_mapf/`: implementation, training, evaluation, plots and demo backend.
- `demo/`: browser interface and demonstration instructions.
- `artifacts/policy.pt`, `gate.json`, `r_order.json`: trained policy and final controller settings.
- `artifacts/results.csv`, `summary.csv`, `sweep.csv`: original experimental evidence.
- `artifacts/*_log.txt`: retained training and experiment logs.
- `report/build_assets.py`: regenerate report tables, figures and metrics audit from saved results.

The code archive includes model weights so assessors can run the demo without retraining. It omits caches, operating-system files, third-party paper PDFs and intermediate TeX files.

## Reproduce the saved analysis

```sh
python report/build_assets.py
python -m hybrid_mapf.plots
python -m hybrid_mapf.tex_numbers
```

`build_assets.py` checks saved summaries against the raw runs before creating report assets. Rebuilding PDFs requires the optional Tectonic LaTeX compiler:

```sh
cd report
tectonic project_report.tex
tectonic contributions.tex
```

The report source is also usable with a normal LaTeX installation or Overleaf after uploading the `report/` directory. Run the asset generator first.

## Rerun evaluation with the supplied trained models

Work in a copy of this package, since the evaluator overwrites output CSVs. Before fresh evaluation, move the existing `artifacts/runs.jsonl` checkpoint out of that copy if present; otherwise results are resumed rather than recomputed.

```sh
python -m hybrid_mapf.experiments --procs 1 --n 20 --Ns 16 32 64 96 128 160 192
```

The full evaluation contains 1,400 main runs and 1,100 sweep runs and can take substantial time. For a smaller exploratory evaluation use `--n 2 --Ns 16 32`; this does not reproduce the report's statistical sample. CPU runtimes will differ from the recorded experiment values. Original logs do not retain a complete hardware/OS specification, so exact runtime replication cannot be claimed.

## Retrain from scratch

Use a working copy and preserve the submitted artifacts first. Keep `artifacts/r_order.json` at `{"R": 0.5, "sample": true}` to use the submitted sampled ranking:

```sh
python -m hybrid_mapf.train --stage policy --policy-inst 400 --epochs 10 --procs 1
python -m hybrid_mapf.train --stage gate --gate-train 120 --gate-val 40 --procs 1
python -m hybrid_mapf.train --stage select --procs 1
python -m hybrid_mapf.train --stage calibrate --procs 1
```

Policy training constructs the imitation dataset in memory and can require several GB of RAM. Full retraining may differ across hardware or library builds. Gate labels are regenerated rather than bundled as a separate dataset. The historical `tune_R.py` script overwrites `r_order.json` with an R-only object, removing `sample:true`; do not run it as part of the final sampled configuration without restoring the full configuration afterward. The retained tuning log describes an earlier deterministic-ordering experiment.

## Evaluation interpretation

The controller's online loop allows 256 steps. A successful fallback appends its entire remaining LaCAM plan; full LaCAM baselines have their own node/time budgets. These are not identical path-length limits. Quality is computed on a common solved subset, excluding shield and confidence-only methods. The random-control probability is set retrospectively from the learned gate's pooled call frequency per agent count, and realised rates can differ. Three repeats in the demo measure timing variation on one instance, not aggregate success over three different maps.

Algorithms are implemented from the cited literature; this is not the authors' official implementation. No new completeness or optimality guarantee is asserted for the budgeted implementation. See the report for references, numerical results and limitations.
