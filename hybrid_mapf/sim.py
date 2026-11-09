"""Closed-loop controllers. Every method executes one joint step per timestep (except full-horizon LaCAM)."""
import time

import numpy as np

from .gate import History, features
from .grid import sum_dist
from .lacam import lacam
from .pibt import pibt_step, shield_step, update_priorities
from .policy import ObsBuilder

T_MAX = 256


def flowtime(path, goals):
    P = np.stack(path)  # (T+1, N)
    off = P != goals[None]
    last = np.where(off.any(0), P.shape[0] - np.argmax(off[::-1], 0), 0)  # 1 + last time off goal
    return int(last.sum()), int(last.max())


def _result(inst, path, success, t0, **kw):
    ft, mk = flowtime(path, inst.goals)
    lb = int(inst.dist[np.arange(inst.N), inst.starts].sum())
    return dict(success=bool(success), flowtime=ft, makespan=mk, sum_lb=lb,
                subopt=ft / max(lb, 1), runtime=time.perf_counter() - t0, **kw)


def run_lacam(inst, policy=None, max_nodes=20000, time_limit=60.0, seed=0):
    """Full-horizon LaCAM; with a policy this is Paper 1's LaCAM + CS-PIBT configuration generator."""
    t0 = time.perf_counter()
    ob = ObsBuilder(inst) if policy else None
    pref_fn = (lambda Q: policy.probs(inst.grid, ob, Q)) if policy else None
    path, nodes = lacam(inst.grid, inst.starts, inst.goals, inst.dist, pref_fn, max_nodes,
                        time_limit, np.random.default_rng(seed))
    ok = path is not None
    return _result(inst, path if ok else [inst.starts], ok, t0, n_cspibt=0,
                   n_lacam=1, lacam_nodes=nodes, steps=len(path) - 1 if ok else -1)


class Controller:
    """mode:
      shield        policy argmax + collision shield (no search)
      pibt          heuristic PIBT every step (no learning)
      cspibt        policy + CS-PIBT every step (Paper 1, one-step correction)
      gate          tiered: q<tau1 shield, q>=tau1 CS-PIBT, q>=tau1 and stalled -> budgeted LaCAM
      random        CS-PIBT with probability `rate` per step (compute-matched control)
    """

    def __init__(self, policy, mode, gate=None, rate=0.0, use_lacam=False, hold=1, stall=16,
                 b_min=200, b_max=2000, lacam_cooldown=16, seed=0):
        self.policy, self.mode, self.gate, self.rate = policy, mode, gate, rate
        self.use_lacam, self.b_min, self.b_max, self.cooldown = use_lacam, b_min, b_max, lacam_cooldown
        # hold: once CS-PIBT is triggered keep it on for `hold` steps (jams take several steps to clear)
        # stall: LaCAM fallback fires when the best sum of cost-to-go has not improved for `stall` steps
        self.hold, self.stall = hold, stall
        self.rng = np.random.default_rng(seed)

    def run(self, inst, T_max=T_MAX, record=None, trace=None):
        t0 = time.perf_counter()
        g, N = inst.grid, inst.N
        ob = ObsBuilder(inst)
        Q = inst.starts.copy()
        prio = self.rng.random(N) * 1e-3
        path, hist = [Q.copy()], History()
        hist.push(Q, sum_dist(inst, Q))
        n_cs = n_lc = lc_nodes = 0
        next_lacam = hold_until = 0
        best, best_t = sum_dist(inst, Q), 0
        q = 0.5
        success = False

        def log_step(after, method, score=None):
            if trace is not None:
                trace.append(dict(positions=after.tolist(), method=method, q=score,
                                  arrived=int((after == inst.goals).sum()),
                                  remaining=sum_dist(inst, after),
                                  search_calls=n_cs, lacam_calls=n_lc,
                                  lacam_nodes=lc_nodes,
                                  compute_seconds=time.perf_counter() - t0))

        for t in range(T_max):
            if (Q == inst.goals).all():
                success = True
                break
            order = np.argsort(-prio)
            if self.mode == "pibt":
                Q_to = pibt_step(g, Q, inst.dist, order, rng=self.rng)
                n_cs += 1
                step_method = "PIBT"
            else:
                probs = self.policy.probs(g, ob, Q)
                acts = probs.argmax(1)
                tier = 0
                if self.mode == "cspibt":
                    tier = 1
                elif self.mode == "random":
                    tier = int(self.rng.random() < self.rate)
                elif self.mode == "gate":
                    phi = features(inst, Q, probs, acts, hist, t, T_max)
                    q = float(self.gate.q(phi))
                    if q >= self.gate.tau1:
                        hold_until = t + self.hold
                    tier = int(t < hold_until)
                    if record is not None:
                        record.append((phi, Q.copy(), prio.copy(), t))
                if self.use_lacam and t - best_t >= self.stall and t >= next_lacam:
                    tier = 2
                if tier == 2:
                    n_lc += 1
                    budget = int(self.b_min + (self.b_max - self.b_min) * q)
                    plan, nodes = lacam(g, Q, inst.goals, inst.dist,
                                        lambda Qc: self.policy.probs(g, ob, Qc), budget,
                                        rng=self.rng)
                    lc_nodes += nodes
                    if plan is not None:
                        for after in plan[1:]:
                            log_step(after, "LaCAM", q if self.mode == "gate" else None)
                        path.extend(plan[1:])
                        Q = plan[-1]
                        success = True
                        break
                    next_lacam = t + self.cooldown
                    tier = 1
                if tier == 1:
                    Q_to = pibt_step(g, Q, inst.dist, order, pref=probs, rng=self.rng)
                    n_cs += 1
                else:
                    Q_to = shield_step(g, Q, acts)
                step_method = "CS-PIBT" if tier == 1 else "Shield"
            log_step(Q_to, step_method, q if self.mode == "gate" else None)
            Q = Q_to
            prio = update_priorities(prio, Q, inst.goals)
            path.append(Q.copy())
            sd = sum_dist(inst, Q)
            hist.push(Q, sd)
            if sd < best:
                best, best_t = sd, t + 1
        success = success or bool((Q == inst.goals).all())
        return _result(inst, path, success, t0, n_cspibt=n_cs, n_lacam=n_lc,
                       lacam_nodes=lc_nodes, steps=len(path) - 1)


def counterfactual_gain(inst, policy, ob, Q, prio, H=8, rng=None):
    """Label for the gate: extra progress (drop in sum of cost-to-go over H steps) from running CS-PIBT
    for the next H steps instead of the shielded policy. (A one-step version, CS-PIBT now then shield,
    gave a gate that lost to a random gate closed-loop: it ignored jams that persist over several steps.)"""
    rng = rng if rng is not None else np.random.default_rng(0)
    g = inst.grid
    out = []
    for first in (0, 1):
        Qc, pc = Q.copy(), prio.copy()
        for h in range(H):
            probs = policy.probs(g, ob, Qc)
            if first == 1:
                Qc = pibt_step(g, Qc, inst.dist, np.argsort(-pc), pref=probs, rng=rng)
            else:
                Qc = shield_step(g, Qc, probs.argmax(1))
            pc = update_priorities(pc, Qc, inst.goals)
        out.append(sum_dist(inst, Q) - sum_dist(inst, Qc))
    return out[1] - out[0]
