"""Learned 1-step local policy: field-of-view CNN trained by imitation of LaCAM solutions."""
import numpy as np
import torch
import torch.nn as nn

from .grid import INF

R = 5  # field-of-view radius -> (2R+1)^2 window
C = 4  # channels: obstacles, other agents, own cost-to-go (relative to centre), own goal


class ObsBuilder:
    def __init__(self, inst):
        self.inst = inst
        g = inst.grid
        self.W = g.W
        self.obst = np.pad(~g.free, R, constant_values=True).astype(np.float32)
        d = inst.dist.reshape(inst.N, g.H, g.W).astype(np.float32)
        d[d >= INF] = np.nan
        self.dist = np.pad(d, ((0, 0), (R, R), (R, R)), constant_values=np.nan)
        off = np.arange(-R, R + 1)
        self.dr, self.dc = off[:, None], off[None, :]
        gr, gc = np.divmod(inst.goals, g.W)
        self.goal_rc = np.stack([gr, gc], 1)

    def __call__(self, Q):
        N = len(Q)
        r, c = np.divmod(Q, self.W)
        rows = r[:, None, None] + R + self.dr  # (N, 2R+1, 1)
        cols = c[:, None, None] + R + self.dc  # (N, 1, 2R+1)
        occ = np.zeros_like(self.obst)
        occ[r + R, c + R] = 1.0
        obs = np.zeros((N, C, 2 * R + 1, 2 * R + 1), dtype=np.float32)
        obs[:, 0] = self.obst[rows, cols]
        obs[:, 1] = occ[rows, cols]
        obs[:, 1, R, R] = 0.0
        ctg = self.dist[np.arange(N)[:, None, None], rows, cols]
        ctg = ctg - ctg[:, R:R + 1, R:R + 1]
        obs[:, 2] = np.nan_to_num(np.clip(ctg, -R - 1, R + 1) / (R + 1), nan=0.0)
        gr = self.goal_rc[:, 0] - r + R
        gc = self.goal_rc[:, 1] - c + R
        inview = (gr >= 0) & (gr <= 2 * R) & (gc >= 0) & (gc <= 2 * R)
        obs[np.flatnonzero(inview), 3, gr[inview], gc[inview]] = 1.0
        return obs


class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(C, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(64 * 6 * 6, 256), nn.ReLU(), nn.Linear(256, 5))

    def forward(self, x):
        return self.body(x)


class Policy:
    """Wraps the net for inference: masked softmax over valid actions."""

    def __init__(self, net, device=None):
        # MPS is ~14x faster than single-threaded CPU for a batch of ~128 agents
        self.dev = torch.device(device or ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.net = net.eval().to(self.dev)

    @torch.no_grad()
    def probs(self, grid, obs_builder, Q):
        logits = self.net(torch.from_numpy(obs_builder(Q)).to(self.dev)).cpu().numpy()
        logits[grid.nbr[Q] < 0] = -np.inf
        logits -= logits.max(1, keepdims=True)
        p = np.exp(logits)
        return p / p.sum(1, keepdims=True)

    @classmethod
    def load(cls, path):
        net = PolicyNet()
        net.load_state_dict(torch.load(path, map_location="cpu"))
        return cls(net)


def actions_from_path(grid, path):
    """Expert actions a_t[i] such that nbr[Q_t[i], a] == Q_{t+1}[i]."""
    out = []
    for Q, Qn in zip(path[:-1], path[1:]):
        out.append(np.argmax(grid.nbr[Q] == Qn[:, None], axis=1))
    return out
