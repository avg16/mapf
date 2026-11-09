"""LaCAM (Okumura, AAAI 2023): lazy constraint addition search over joint configurations,
with PIBT / CS-PIBT as the configuration generator."""
import time
from collections import deque

import numpy as np

from .pibt import pibt_step


class _High:
    __slots__ = ("Q", "parent", "order", "pref", "tree")

    def __init__(self, Q, parent, order, pref):
        self.Q, self.parent, self.order, self.pref = Q, parent, order, pref
        self.tree = deque([((), ())])  # low-level constraints: (agents, vertices)


def lacam(grid, starts, goals, dist, pref_fn=None, max_nodes=5000, time_limit=60.0, rng=None):
    """Returns (list of configurations from starts to goals or None, #high-level nodes generated)."""
    rng = rng if rng is not None else np.random.default_rng(0)
    N = len(starts)
    goal_key = tuple(int(g) for g in goals)
    idx = np.arange(N)
    t0 = time.perf_counter()

    def make(Q, parent):
        Qa = np.array(Q)
        d = dist[idx, Qa]
        order = list(np.lexsort((-d, d == 0)))  # agents off-goal first, far ones first
        return _High(Q, parent, order, pref_fn(Qa) if pref_fn else None)

    root = make(tuple(int(v) for v in starts), None)
    open_, explored, nodes = [root], {root.Q: root}, 1
    while open_:
        if nodes >= max_nodes or time.perf_counter() - t0 > time_limit:
            return None, nodes
        H = open_[-1]
        if H.Q == goal_key:
            path = []
            while H is not None:
                path.append(np.array(H.Q))
                H = H.parent
            return path[::-1], nodes
        if not H.tree:
            open_.pop()
            continue
        who, where = H.tree.popleft()
        if len(who) < N:
            i = H.order[len(who)]
            v = H.Q[i]
            cands = [int(u) for u in grid.nbr[v] if u >= 0]
            rng.shuffle(cands)
            for u in cands:
                H.tree.append((who + (i,), where + (u,)))
        Q_to = pibt_step(grid, np.array(H.Q), dist, H.order, pref=H.pref,
                         fixed=dict(zip(who, where)), rng=rng)
        if Q_to is None:
            continue
        key = tuple(int(v) for v in Q_to)
        if key in explored:
            open_.append(explored[key])
            continue
        new = make(key, H)
        nodes += 1
        explored[key] = new
        open_.append(new)
    return None, nodes
