"""One-step joint action generators: collision shield, (CS-)PIBT, conflict checks."""
import json
import os
import sys

import numpy as np

sys.setrecursionlimit(10000)


def proposal_conflicts(Q, tgt):
    """Per-agent flag: is agent i in a vertex or edge conflict in the proposed move Q -> tgt."""
    N = len(Q)
    flag = np.zeros(N, dtype=bool)
    _, inv, cnt = np.unique(tgt, return_inverse=True, return_counts=True)
    flag |= cnt[inv] > 1
    pos = {int(v): i for i, v in enumerate(Q)}
    for i in range(N):
        if tgt[i] != Q[i]:
            j = pos.get(int(tgt[i]))
            if j is not None and j != i and tgt[j] == Q[i]:
                flag[i] = flag[j] = True
    return flag


def shield_step(grid, Q, actions):
    """Naive collision shield: execute actions, freeze every agent in conflict until conflict-free."""
    tgt = grid.nbr[Q, actions]
    tgt = np.where(tgt < 0, Q, tgt)
    while True:
        bad = proposal_conflicts(Q, tgt) & (tgt != Q)
        if not bad.any():
            return tgt
        tgt = np.where(bad, Q, tgt)


def is_valid_step(Q, Q_to):
    if len(np.unique(Q_to)) != len(Q_to):
        return False
    return not (proposal_conflicts(Q, Q_to) & (Q_to != Q)).any()


# weight R in the CS-PIBT ordering key h(T(s,a)) + R[1 - p(a|O)] (Veerapaneni et al. 2024);
# None ranks purely by policy probability. Tuned on validation instances (see tune_R.py).
def _load_cfg():
    path = os.path.join(os.path.dirname(__file__), "..", "artifacts", "r_order.json")
    cfg = json.load(open(path)) if os.path.exists(path) else {}
    return cfg.get("R"), cfg.get("sample", False)


R_ORDER, SAMPLE = _load_cfg()


def candidate_order(grid, i, v, dist_i, pref_i, rng):
    """Order next vertices for agent i. pref_i: (5,) policy probabilities over actions or None."""
    acts = [a for a in range(5) if grid.nbr[v, a] >= 0]
    noise = rng.random(len(acts))
    if pref_i is None:
        key = [(dist_i[grid.nbr[v, a]], noise[k]) for k, a in enumerate(acts)]
    elif R_ORDER is None:
        key = [(-pref_i[a], dist_i[grid.nbr[v, a]], noise[k]) for k, a in enumerate(acts)]
    elif SAMPLE:
        # Gumbel-top-k: a random ranking drawn from the policy distribution, not its argmax
        # the sampled rank replaces p in the key, so it decides ties between equal-distance moves
        g = np.log(np.maximum([pref_i[a] for a in acts], 1e-12)) - np.log(-np.log(noise))
        rank = np.argsort(np.argsort(-g)) / 4.0
        key = [(dist_i[grid.nbr[v, a]] + R_ORDER * rank[k], noise[k]) for k, a in enumerate(acts)]
    else:
        key = [(dist_i[grid.nbr[v, a]] + R_ORDER * (1 - pref_i[a]), noise[k]) for k, a in enumerate(acts)]
    return [grid.nbr[v, acts[k]] for k in sorted(range(len(acts)), key=key.__getitem__)]


def pibt_step(grid, Q, dist, order, pref=None, fixed=None, rng=None):
    """PIBT (Okumura et al. 2022) one step. With `pref` from a learned policy this is CS-PIBT
    (Veerapaneni et al. 2024): the policy ranks candidates, PIBT guarantees collision-freeness.
    order: agents by decreasing priority. fixed: {agent: vertex} constraints (used by LaCAM).
    Returns Q_to or None if the constraints are infeasible."""
    rng = rng if rng is not None else np.random.default_rng()
    N = len(Q)
    nV = grid.H * grid.W
    occ_now = np.full(nV, -1, dtype=np.int64)
    occ_now[Q] = np.arange(N)
    occ_nxt = np.full(nV, -1, dtype=np.int64)
    Q_to = np.full(N, -1, dtype=np.int64)

    if fixed:
        for i, v in fixed.items():
            if occ_nxt[v] >= 0:
                return None
            j = occ_now[v]
            if j >= 0 and j != i and Q_to[j] == Q[i]:
                return None
            Q_to[i] = v
            occ_nxt[v] = i

    def func(i):
        for v in candidate_order(grid, i, Q[i], dist[i], None if pref is None else pref[i], rng):
            if occ_nxt[v] >= 0:
                continue
            j = occ_now[v]
            if j >= 0 and Q_to[j] == Q[i]:
                continue
            Q_to[i] = v
            occ_nxt[v] = i
            if j >= 0 and j != i and Q_to[j] < 0 and not func(j):
                continue
            return True
        Q_to[i] = Q[i]
        occ_nxt[Q[i]] = i
        return False

    for i in order:
        if Q_to[i] < 0:
            func(i)
    if fixed and not is_valid_step(Q, Q_to):
        return None
    return Q_to


def update_priorities(prio, Q, goals):
    at_goal = Q == goals
    return np.where(at_goal, prio - np.floor(prio), prio + 1)
