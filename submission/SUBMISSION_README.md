# Project submission

**Learning When to Search for Multi Agent Path Finding**

- Abhivansh Gupta — 23112004
- Shivang Joshi — 23117132
- IIT Roorkee

## Files to submit

1. `Project_Report.pdf`: five-page project report, including references.
2. `Project_Code.zip`: source, browser demo, trained policy, configurations, original result data, logs, editable LaTeX report and reproduction instructions.
3. `Member_Contributions.pdf`: two individual contribution pages, one per member. Separate copies are supplied as `Abhivansh_Contributions.pdf` and `Shivang_Contributions.pdf`.

`Complete_Submission.zip` bundles the three deliverables, this note and the source SHA-256 manifest. If the portal asks for separate uploads, use the individual files above.

## Before submitting

Both members should review their individual allocations and the report before submission. The contribution sheets divide methods/policy/search and gate/experiments/demo respectively, with shared project ideation. Any separately required disclosure form is outside these contribution sheets. The report uses the supplied full names and student IDs.

## Assessor quick start

Extract `Project_Code.zip`, change into `mapf_project/`, create a Python 3.12 virtual environment, install `requirements.txt`, then run:

```sh
python -m hybrid_mapf.smoke_check
python -m hybrid_mapf.demo
```

Open http://127.0.0.1:8765. No ngrok account is required. The supplied weights avoid retraining. Details on running full experiments and rebuilding the report are in the code package's README.

The report's results come from the original saved 2,500 runs. Packaging verifies those summaries and smoke-checks current inference/replay; it does not claim a new full retraining or benchmark run. Hardware-dependent demo timings can differ from the archived experimental values.
