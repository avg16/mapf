"""Pick R for the CS-PIBT ordering key on validation instances (seeds 30000+), never on test."""
import json
import os
from multiprocessing import Pool

import numpy as np
import torch

from . import pibt
from .grid import make_instance
from .policy import Policy
from .sim import Controller

OUT = os.path.join(os.path.dirname(__file__), "..", "artifacts")
GRID = [None, 0.5, 1.0, 2.0, 4.0, 8.0]


def _run(job):
    R, N, k = job
    torch.set_num_threads(1)
    pibt.R_ORDER = R
    seed = 30000 + 1000 * N + k
    r = Controller(Policy.load(os.path.join(OUT, "policy.pt")), "cspibt", seed=seed).run(make_instance(seed, N))
    return R, r["success"], r["subopt"]


if __name__ == "__main__":
    jobs = [(R, N, k) for R in GRID for N in [64, 128, 192] for k in range(10)]
    with Pool(8) as p:
        res = p.map(_run, jobs)
    best = None
    for R in GRID:
        s = [x for x in res if x[0] == R]
        succ, sub = np.mean([x[1] for x in s]), np.mean([x[2] for x in s if x[1]] or [np.nan])
        print(f"R={R}: val success {succ:.3f}  subopt on solved {sub:.3f}")
        if best is None or (succ, -sub) > best[0]:
            best = ((succ, -sub), R)
    print("chosen R =", best[1])
    json.dump({"R": best[1]}, open(os.path.join(OUT, "r_order.json"), "w"))
