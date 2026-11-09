"""Grid maps, MAPF instances and BFS cost-to-go tables."""
from collections import deque
from dataclasses import dataclass

import numpy as np

# action order: stay, up, down, left, right
MOVES = np.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]])
INF = 10 ** 6


class Grid:
    def __init__(self, free: np.ndarray):
        self.free = free.astype(bool)
        self.H, self.W = free.shape
        n = self.H * self.W
        # nbr[v, a] = vertex reached from v with action a, -1 if blocked
        self.nbr = np.full((n, 5), -1, dtype=np.int64)
        for r in range(self.H):
            for c in range(self.W):
                if not self.free[r, c]:
                    continue
                for a, (dr, dc) in enumerate(MOVES):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < self.H and 0 <= cc < self.W and self.free[rr, cc]:
                        self.nbr[r * self.W + c, a] = rr * self.W + cc
        self.n_free = int(self.free.sum())

    def bfs(self, goal: int) -> np.ndarray:
        d = np.full(self.H * self.W, INF, dtype=np.int64)
        d[goal] = 0
        q = deque([goal])
        while q:
            v = q.popleft()
            for u in self.nbr[v, 1:]:
                if u >= 0 and d[u] == INF:
                    d[u] = d[v] + 1
                    q.append(u)
        return d


@dataclass
class Instance:
    grid: Grid
    starts: np.ndarray
    goals: np.ndarray
    dist: np.ndarray  # (N, H*W) cost-to-go to each agent's goal
    seed: int

    @property
    def N(self):
        return len(self.starts)


def random_free_map(H, W, obstacle_ratio, rng):
    free = rng.random((H, W)) >= obstacle_ratio
    # keep only the largest 4-connected component so every start/goal pair is reachable
    label = -np.ones((H, W), dtype=int)
    sizes = []
    for r in range(H):
        for c in range(W):
            if free[r, c] and label[r, c] < 0:
                k = len(sizes)
                label[r, c] = k
                q, cnt = deque([(r, c)]), 0
                while q:
                    a, b = q.popleft()
                    cnt += 1
                    for dr, dc in MOVES[1:]:
                        x, y = a + dr, b + dc
                        if 0 <= x < H and 0 <= y < W and free[x, y] and label[x, y] < 0:
                            label[x, y] = k
                            q.append((x, y))
                sizes.append(cnt)
    return label == int(np.argmax(sizes))


def make_instance(seed, N, H=32, W=32, obstacle_ratio=0.1):
    rng = np.random.default_rng(seed)
    grid = Grid(random_free_map(H, W, obstacle_ratio, rng))
    cells = np.flatnonzero(grid.free.ravel())
    starts = rng.choice(cells, N, replace=False)
    goals = rng.choice(cells, N, replace=False)
    dist = np.stack([grid.bfs(g) for g in goals])
    return Instance(grid, starts, goals, dist, seed)


def sum_dist(inst, Q):
    return int(inst.dist[np.arange(inst.N), Q].sum())
