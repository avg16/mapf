"""Search-trigger gate: per-timestep risk features phi_t and a logistic gate q_t = sigmoid(w.phi + b)."""
import numpy as np

FEATURES = [
    # confidence
    "mean_maxp", "min_maxp", "mean_entropy", "max_entropy", "mean_margin", "min_margin",
    # local risk
    "frac_conflict", "density", "mean_local_density", "max_local_density", "mean_free_nbrs",
    # temporal risk
    "frac_active", "frac_stuck", "repeat_count", "progress_rate", "time_frac",
]


class History:
    """Rolling window of recent configurations for temporal features."""

    def __init__(self, w=8):
        self.w = w
        self.configs, self.sums = [], []

    def push(self, Q, sd):
        self.configs.append(Q.copy())
        self.sums.append(sd)
        if len(self.configs) > self.w + 1:
            self.configs.pop(0)
            self.sums.pop(0)


def features(inst, Q, probs, acts, hist, t, T_max):
    from .pibt import proposal_conflicts
    g = inst.grid
    N = inst.N
    active = Q != inst.goals
    p = np.clip(probs, 1e-12, 1)
    srt = np.sort(probs, 1)
    maxp, margin = srt[:, -1], srt[:, -1] - srt[:, -2]
    ent = -(p * np.log(p)).sum(1)
    tgt = g.nbr[Q, acts]
    tgt = np.where(tgt < 0, Q, tgt)
    conf = proposal_conflicts(Q, tgt)
    # agents within the 5x5 neighbourhood
    r, c = np.divmod(Q, g.W)
    occ = np.zeros((g.H + 4, g.W + 4), dtype=np.float32)
    occ[r + 2, c + 2] = 1
    cs = occ.cumsum(0).cumsum(1)
    cs = np.pad(cs, ((1, 0), (1, 0)))
    loc = (cs[r + 5, c + 5] - cs[r, c + 5] - cs[r + 5, c] + cs[r, c] - 1) / 24.0
    free_nbrs = (g.nbr[Q, 1:] >= 0).sum(1)
    if len(hist.configs) > 1:
        old = hist.configs[0]
        stuck = ((Q == old) & active).mean()
        key = Q.tobytes()
        repeat = sum(q.tobytes() == key for q in hist.configs[:-1])
        prog = (hist.sums[0] - hist.sums[-1]) / (N * (len(hist.sums) - 1))
    else:
        stuck, repeat, prog = 0.0, 0, 1.0
    a = active if active.any() else np.ones(N, bool)
    return np.array([
        maxp[a].mean(), maxp[a].min(), ent[a].mean(), ent[a].max(), margin[a].mean(), margin[a].min(),
        conf.mean(), N / g.n_free, loc.mean(), loc.max(), free_nbrs.mean(),
        active.mean(), stuck, repeat, prog, t / T_max,
    ], dtype=np.float64)


class LogisticGate:
    def __init__(self, mean, std, w, b, tau1, tau2):
        self.mean, self.std, self.w, self.b = mean, std, w, b
        self.tau1, self.tau2 = tau1, tau2

    def q(self, phi):
        z = ((phi - self.mean) / self.std) @ self.w + self.b
        return 1.0 / (1.0 + np.exp(-z))

    def to_dict(self):
        return {k: np.asarray(v).tolist() for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d):
        return cls(*(np.asarray(d[k]) if isinstance(d[k], list) else d[k]
                     for k in ["mean", "std", "w", "b", "tau1", "tau2"]))


class ConfidenceGate:
    """Ablation: trigger on low mean max-probability only."""

    def __init__(self, thr):
        self.thr = thr

    def q(self, phi):
        return 1.0 - phi[FEATURES.index("mean_maxp")]
