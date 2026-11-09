# Local MAPF simulator

From the repository root:

```sh
python3.12 -m hybrid_mapf.demo
```

Open http://127.0.0.1:8765. Use `--port 8766` if the default port is occupied. Stop with Ctrl+C. Python needs the project's NumPy and PyTorch dependencies; the Python 3.12 installation on this machine already has them. No frontend installation or internet connection is needed.

Choose a controller, robot count and map seed, then generate a simulation. Play/pause, step, scrub the timeline or change speed. Click a robot to inspect its goal and 11×11 observation window. Space toggles playback and the right arrow advances one step when focus is outside a form control.

Compare controllers using the same seed and count: the generated starts, goals and obstacles remain the same. Two checked examples for your presentation:

- Seed **42**, **64 robots**: shield alone reaches the step limit; the hybrid solves the instance using shield and CS-PIBT.
- Seed **43**, **128 robots**: the hybrid invokes LaCAM once and solves the instance. Watch the tier indicator change to LaCAM after the stall.

The backend uses the actual project algorithms, `artifacts/policy.pt` and `artifacts/gate.json`. Trajectories are computed first, checked for legal movement and collisions, and replayed in the browser. The gate score belongs to the decision that produced the displayed configuration. During a successful LaCAM fallback, the triggering score is retained as the plan is replayed; the gate is not re-evaluated along that plan. Summary metrics describe the whole run, while the goal count, remaining distance and chart follow the playback position.

The viewport is always the full 32×32 map. The shaded 11×11 area is the selected robot's neural-policy observation, not the map available to LaCAM. Playback speed is a presentation control and does not represent measured controller runtime. Unsuccessful runs may show a lower distance remaining without solving the instance. Full LaCAM is limited to 20,000 generated nodes and 10 seconds in this demo. Controller runs use the project's normal 256 online-step limit and fallback settings.

The server binds only to this computer's loopback address. Recent generated trajectories are cached in memory. All assets are local.

## Search, time and algorithm comparisons

The cards below the replay show cumulative one-step search calls, LaCAM calls, recorded compute elapsed and LaCAM nodes at the current playback position. Full-run time and completed-solution quality are also visible. Scrubbing backward resets the cumulative displays to that recorded point. For LaCAM plans, planning time is charged before their first displayed move, rather than spread artificially over the animation. A fallback's internal CS-PIBT calls are not added to the external one-step-call count; its nodes are reported separately.

Click **Compare all 6 controllers** to benchmark the selected seed and count. The table runs each controller three times with fresh uncached execution, after an untimed policy warm-up. It shows median and minimum/maximum controller wall-clock time, solved repeats, goal completion, external one-step search calls, LaCAM calls/nodes, search calls per replay step, flowtime, quality ratio and makespan. All controllers use the same generated instance and seeded random choices. A star marks any differing repeat outcomes; non-time table metrics describe the first repeat. View a row to replay its controller, or download the table as CSV.

Unsolved runs have no solution flowtime, makespan or quality score in the table. The static-distance lower bound ignores agent interactions; a ratio of 1.2 means 20% above this bound, not 20% above a proven optimum. The search-rate denominator includes fallback-plan replay steps, so the explicit label is “one-step search calls / replay steps.” A LaCAM call is much more work than one PIBT call; compare runtime and nodes alongside calls.

Timings exclude map generation, model loading, warm-up, HTTP transfer, collision verification and drawing. The online-controller timer includes lightweight trajectory tracing, feature extraction and inference. These are single-machine demo measurements on one instance. Three repeated runs of the same seed are not three independent test maps and do not establish aggregate success rates. Use the project's multi-map experimental results for broader claims.
