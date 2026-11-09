"""Evaluate all methods on held-out test instances (seeds 40000+) across agent counts.
Random gates are compute-matched per N to the learned gate's measured CS-PIBT rate. A second sweep
traces success vs search rate for the learned gate (each validation recall level) and random gates."""
import argparse
import json
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch

from .gate import ConfidenceGate, LogisticGate
from .grid import make_instance
from .policy import Policy
from .sim import Controller, run_lacam

OUT = os.path.join(os.path.dirname(__file__), "..", "artifacts")
_policy = _gdict = None


def _init():
    global _policy, _gdict
    torch.set_num_threads(1)
    _policy = Policy.load(os.path.join(OUT, "policy.pt"))
    _gdict = json.load(open(os.path.join(OUT, "gate.json")))


def _run(job):
    """job = (method, N, instance index, parameter); parameter is a rate (random) or tau (gate)."""
    method, N, k, x = job
    seed = 40000 + 1000 * N + k
    inst = make_instance(seed, N)
    hold = _gdict["hold"]
    if method == "lacam":
        r = run_lacam(inst, None, seed=seed)
    elif method == "lacam_policy":
        r = run_lacam(inst, _policy, seed=seed)
    elif method == "gate_conf":
        g = ConfidenceGate(_gdict["conf_thr"])
        g.tau1 = _gdict["conf_thr"]
        r = Controller(_policy, "gate", gate=g, hold=hold, seed=seed).run(inst)
    elif method in ("gate", "gate_lacam", "gate_sweep"):
        g = LogisticGate.from_dict(_gdict)
        if method == "gate_sweep":
            g.tau1 = x
        r = Controller(_policy, "gate", gate=g, hold=hold, use_lacam=method == "gate_lacam",
                       seed=seed).run(inst)
    elif method in ("random", "random_lacam", "random_sweep"):
        r = Controller(_policy, "random", rate=x, use_lacam=method == "random_lacam", seed=seed).run(inst)
    else:
        r = Controller(_policy, method, seed=seed).run(inst)
    return dict(method=method, N=N, inst=k, param=x, **r)


CKPT = os.path.join(OUT, "runs.jsonl")


def _key(job):
    return json.dumps(list(job))


def _map(jobs, procs, timeout=300, retries=3):
    """Run jobs in a pool, appending each result to a checkpoint file. A crashed worker loses its job
    silently in multiprocessing.Pool (the first run hung on this), so a stalled pool is restarted
    and only missing jobs rerun."""
    done = {}
    if os.path.exists(CKPT):
        for line in open(CKPT):
            r = json.loads(line)
            done[r.pop("_job")] = r
    for attempt in range(retries + 1):
        todo = [j for j in jobs if _key(j) not in done]
        if not todo:
            break
        print(f"  {len(todo)} jobs to run (attempt {attempt + 1})", flush=True)
        p = Pool(procs, initializer=_init)
        try:
            it = p.imap_unordered(_run_keyed, todo)
            with open(CKPT, "a") as f:
                for n in range(len(todo)):
                    k, r = it.next(timeout=timeout)
                    done[k] = r
                    f.write(json.dumps({"_job": k, **r}) + "\n")
                    f.flush()
                    if (n + 1) % 100 == 0:
                        print(f"  runs: {n + 1}/{len(todo)}", flush=True)
        except Exception as e:  # TimeoutError from a lost job, or a broken worker
            print(f"  pool stalled ({type(e).__name__}); restarting for missing jobs", flush=True)
        finally:
            p.terminate()
    missing = [j for j in jobs if _key(j) not in done]
    if missing:
        raise RuntimeError(f"{len(missing)} jobs never completed, e.g. {missing[:3]}")
    return [done[_key(j)] for j in jobs]


def _run_keyed(job):
    return _key(job), _run(job)


def wilson(x, z=1.96):
    n, p = len(x), x.mean()
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return f"{p:.2f} [{max(c - h, 0):.2f},{min(c + h, 1):.2f}]"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ns", type=int, nargs="+", default=[16, 32, 64, 96, 128, 160, 192])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    main = ["shield", "pibt", "cspibt", "gate", "gate_lacam", "gate_conf", "lacam", "lacam_policy"]
    rows = _map([(m, N, k, 0.0) for m in main for N in a.Ns for k in range(a.n)], a.procs)
    df = pd.DataFrame(rows)
    g = df[df.method == "gate"]
    rate = (g.groupby("N").n_cspibt.sum() / g.groupby("N").steps.sum()).to_dict()
    print("matched random rates:", {k: round(v, 3) for k, v in rate.items()})
    rows += _map([(m, N, k, float(rate[N])) for m in ["random", "random_lacam"]
                  for N in a.Ns for k in range(a.n)], a.procs)
    gd = json.load(open(os.path.join(OUT, "gate.json")))
    sweep_N = [N for N in a.Ns if N >= 64]
    rows += _map([("gate_sweep", N, k, t) for t in gd["taus"].values() for N in sweep_N
                  for k in range(a.n)], a.procs)
    rows += _map([("random_sweep", N, k, x) for x in [0.1, 0.2, 0.3, 0.45, 0.6, 0.8]
                  for N in sweep_N for k in range(a.n)], a.procs)
    df = pd.DataFrame(rows)
    df["search_rate"] = df.n_cspibt / df.steps.clip(lower=1)
    df.to_csv(os.path.join(OUT, "results.csv"), index=False)
    m = df[~df.method.str.endswith("_sweep")]
    summ = m.groupby(["method", "N"]).agg(
        success=("success", wilson), runtime=("runtime", "mean"), cspibt_calls=("n_cspibt", "mean"),
        search_rate=("search_rate", "mean"), lacam_calls=("n_lacam", "mean"),
        lacam_nodes=("lacam_nodes", "mean")).round(3)
    # suboptimality only on instances solved by every search-using method, so failures (cut at T_MAX)
    # do not bias it; shield and the confidence gate solve too few instances to be included
    ok = m.pivot_table(index=["N", "inst"], columns="method", values="success").astype(bool)
    cols = [c for c in ok.columns if c not in ("shield", "gate_conf")]
    common = ok[ok[cols].all(1)].index
    ft = m[m.method.isin(cols)].set_index(["N", "inst"]).loc[lambda d: d.index.isin(common)]
    summ = summ.join(ft.groupby(["method", "N"]).subopt.mean().round(3).rename("subopt_common"))
    summ.to_csv(os.path.join(OUT, "summary.csv"))
    pd.set_option("display.width", 200)
    print(summ.to_string())
    print(f"common-solved instances per N: {pd.Series([n for n, _ in common]).value_counts().sort_index().to_dict()}")
    sw = df[df.method.str.endswith("_sweep")].groupby(["method", "param", "N"]).agg(
        success=("success", "mean"), search_rate=("search_rate", "mean")).round(3)
    sw.to_csv(os.path.join(OUT, "sweep.csv"))
    print(sw.to_string())
