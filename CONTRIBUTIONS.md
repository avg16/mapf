# Individual contribution allocation

Project: Learning When to Search for Multi Agent Path Finding  
Institution: IIT Roorkee

Both members share the project idea and central question of when learned MAPF policies should invoke search. The individual responsibilities below allocate preparation and technical review; they do not claim completed independent implementation.

## Abhivansh Gupta — 23112004

Allocated role: problem formulation, learned policy and search methods.

1. MAPF formulation, collision constraints, flowtime, makespan, random-grid environment and BFS distance fields.
2. Local observations, CNN policy, action masking and imitation-learning methodology.
3. Collision shield, PIBT, policy-guided CS-PIBT, sampled candidate ordering and LaCAM.
4. Method sections of the report, first presentation segment and walkthrough of grid.py, policy.py, pibt.py and lacam.py.
5. Method integration, model-loading/setup review and consistency with the experiment description.

## Shivang Joshi — 23117132

Allocated role: search gate, experimental analysis and visual demonstration.

1. Gate features, logistic score, counterfactual labels, threshold selection and stall-based fallback interpretation.
2. Test protocol, seeds, baselines, random-trigger controls and comparison limitations.
3. Success, search calls, runtime, flowtime, common-subset quality and negative-result analysis.
4. Demo walkthrough, live telemetry, three-run benchmarks and CSV export interpretation.
5. Results/limitations sections of the report, second presentation segment and review of gate.py, train.py, experiments.py and demo assets.

The full one-page-per-member descriptions are in report/contribution_pages.tex. The combined PDF and individual PDFs are included in the submission package.
