"""Stage 1: imitation-learn the local policy from LaCAM solutions.
Stage 2: generate counterfactual labels, fit the logistic search gate, pick thresholds on validation.

Seed ranges are disjoint: policy-train 10000+, gate-train 20000+, gate-val 30000+, test 40000+."""
import argparse
import json
import os
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .grid import make_instance
from .lacam import lacam
from .policy import ObsBuilder, Policy, PolicyNet, actions_from_path
from .sim import Controller, counterfactual_gain

OUT = os.path.join(os.path.dirname(__file__), "..", "artifacts")


def _expert(args):
    seed, N = args
    torch.set_num_threads(1)
    inst = make_instance(seed, N)
    path, _ = lacam(inst.grid, inst.starts, inst.goals, inst.dist, max_nodes=20000,
                    time_limit=30, rng=np.random.default_rng(seed))
    if path is None:
        return None
    ob, rng = ObsBuilder(inst), np.random.default_rng(seed)
    X, Y = [], []
    for Q, a in zip(path[:-1], actions_from_path(inst.grid, path)):
        # drop most "wait at goal" samples, which otherwise dominate
        keep = (Q != inst.goals) | (rng.random(N) < 0.1)
        X.append(ob(Q)[keep])
        Y.append(a[keep])
    return np.concatenate(X), np.concatenate(Y)


def train_policy(n_inst, epochs, procs):
    rng = np.random.default_rng(0)
    jobs = [(10000 + k, int(rng.choice([16, 32, 64, 96, 128, 160]))) for k in range(n_inst)]
    with Pool(procs) as p:
        res = [r for r in p.map(_expert, jobs) if r is not None]
    print(f"expert solved {len(res)}/{n_inst} instances")
    X = torch.from_numpy(np.concatenate([r[0] for r in res]))
    Y = torch.from_numpy(np.concatenate([r[1] for r in res])).long()
    perm = torch.randperm(len(X), generator=torch.Generator().manual_seed(0))
    n_val = len(X) // 10
    vi, ti = perm[:n_val], perm[n_val:]
    print(f"samples train={len(ti)} val={n_val} action freq={np.bincount(Y.numpy(), minlength=5) / len(Y)}")
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"training on {dev}")
    torch.manual_seed(0)
    net = PolicyNet().to(dev)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    lossf = nn.CrossEntropyLoss()
    for ep in range(epochs):
        net.train()
        for b in torch.split(ti[torch.randperm(len(ti))], 1024):
            opt.zero_grad()
            lossf(net(X[b].to(dev)), Y[b].to(dev)).backward()
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            acc = np.mean([(net(X[b].to(dev)).argmax(1).cpu() == Y[b]).float().mean().item()
                           for b in torch.split(vi, 4096)])
        print(f"epoch {ep + 1}/{epochs} val acc {acc:.3f}")
    torch.save({k: v.cpu() for k, v in net.state_dict().items()}, os.path.join(OUT, "policy.pt"))


class _RandomTier:
    """Behaviour gate for data collection: CS-PIBT on a random half of the steps."""
    tau1, tau2 = 0.5, 2.0

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def q(self, phi):
        return self.rng.random()


def _gate_data(args):
    seed, N, stride = args
    torch.set_num_threads(1)
    policy = Policy.load(os.path.join(OUT, "policy.pt"))
    inst = make_instance(seed, N)
    rec = []
    Controller(policy, "gate", gate=_RandomTier(seed), seed=seed).run(inst, record=rec)
    ob, rng = ObsBuilder(inst), np.random.default_rng(seed)
    phis, gains = [], []
    for phi, Q, prio, t in rec[::stride]:
        if (Q == inst.goals).all():
            continue
        phis.append(phi)
        gains.append(counterfactual_gain(inst, policy, ob, Q, prio, rng=rng))
    return np.array(phis).reshape(-1, len(rec[0][0]) if rec else 0), np.array(gains)


def gate_dataset(base, n_inst, procs, stride=2):
    rng = np.random.default_rng(base)
    jobs = [(base + k, int(rng.choice([32, 64, 96, 128, 160, 192])), stride) for k in range(n_inst)]
    res = []
    with Pool(procs) as p:
        for k, r in enumerate(p.imap_unordered(_gate_data, jobs)):
            if len(r[1]):
                res.append(r)
            if (k + 1) % 20 == 0:
                print(f"  gate labels: {k + 1}/{n_inst} instances", flush=True)
    return np.concatenate([r[0] for r in res]), np.concatenate([r[1] for r in res])


RECALLS = [0.7, 0.8, 0.9, 0.95, 0.98]
HOLDS = [1, 4, 8]


def train_gate(n_train, n_val, procs, recall=0.9, gain_thr=2):
    Xt, gt = gate_dataset(20000, n_train, procs)
    Xv, gv = gate_dataset(30000, n_val, procs)
    yt, yv = (gt >= gain_thr).astype(int), (gv >= gain_thr).astype(int)
    print(f"gate data train={len(yt)} (pos {yt.mean():.3f})  val={len(yv)} (pos {yv.mean():.3f})")
    mean, std = Xt.mean(0), Xt.std(0) + 1e-8
    clf = LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000)
    clf.fit((Xt - mean) / std, yt)
    qv = clf.predict_proba((Xv - mean) / std)[:, 1]
    print(f"val AUC learned gate: {roc_auc_score(yv, qv):.3f}")
    from .gate import FEATURES
    conf_q = 1 - Xv[:, FEATURES.index("mean_maxp")]
    print(f"val AUC confidence-only (1 - mean max p): {roc_auc_score(yv, conf_q):.3f}")
    # tau1: smallest threshold whose recall on positives is >= `recall`  (Pr(miss | needed) <= 1 - recall)
    taus = {str(r): float(np.quantile(qv[yv == 1], 1 - r)) for r in RECALLS}
    tau1 = taus[str(recall)]
    tau2 = float(np.quantile(qv, 0.99))
    rate = float((qv >= tau1).mean())
    conf_thr = float(np.quantile(conf_q, 1 - rate))  # confidence gate matched to the same trigger rate
    print(f"tau1={tau1:.3f} (val trigger rate {rate:.3f}, recall {(qv[yv == 1] >= tau1).mean():.3f})"
          f"  tau2={tau2:.3f}  conf_thr={conf_thr:.3f}")
    w = clf.coef_[0]
    for k in np.argsort(-np.abs(w)):
        print(f"  {FEATURES[k]:>20s} {w[k]:+.3f}")
    gate = dict(mean=mean.tolist(), std=std.tolist(), w=w.tolist(), b=float(clf.intercept_[0]),
                tau1=tau1, tau2=tau2, conf_thr=conf_thr, val_rate=rate, taus=taus, hold=1,
                val_auc=float(roc_auc_score(yv, qv)), val_auc_conf=float(roc_auc_score(yv, conf_q)))
    with open(os.path.join(OUT, "gate.json"), "w") as f:
        json.dump(gate, f, indent=1)


def _closed_loop(job):
    """Closed-loop gate run on one validation instance -> (success, CS-PIBT calls, steps)."""
    tau, hold, seed, N = job
    torch.set_num_threads(1)
    from .gate import LogisticGate
    g = LogisticGate.from_dict(json.load(open(os.path.join(OUT, "gate.json"))))
    g.tau1 = tau
    r = Controller(Policy.load(os.path.join(OUT, "policy.pt")), "gate", gate=g, hold=hold,
                   seed=seed).run(make_instance(seed, N))
    return r["success"], r["n_cspibt"], r["steps"]


def select_operating_point(procs, n=8):
    """Choose (recall level -> tau1, hold) closed-loop on validation instances: the lowest search rate
    whose success is within 0.05 of the best. The test set is never used for this choice."""
    gd = json.load(open(os.path.join(OUT, "gate.json")))
    inst = [(30500 + 1000 * N + k, N) for N in [64, 128, 192] for k in range(n)]
    cands = []
    with Pool(procs) as p:
        for r in RECALLS:
            for h in HOLDS:
                res = p.map(_closed_loop, [(gd["taus"][str(r)], h, s, N) for s, N in inst])
                succ = np.mean([x[0] for x in res])
                rate = sum(x[1] for x in res) / sum(x[2] for x in res)
                print(f"  recall={r} hold={h}: val success {succ:.3f} search rate {rate:.3f}", flush=True)
                cands.append((succ, rate, r, h))
    # cheapest configuration within `tol` of the best validation success (rule fixed before testing)
    tol = 0.05
    top = max(c[0] for c in cands)
    _, _, r, h = min((c for c in cands if c[0] >= top - tol), key=lambda c: c[1])
    print(f"chosen: recall={r} hold={h}")
    gd.update(tau1=gd["taus"][str(r)], hold=h, recall=r)
    json.dump(gd, open(os.path.join(OUT, "gate.json"), "w"), indent=1)


def _closed_loop_rate(job):
    kind, thr, seed, N = job
    torch.set_num_threads(1)
    from .gate import ConfidenceGate, LogisticGate
    policy = Policy.load(os.path.join(OUT, "policy.pt"))
    gd = json.load(open(os.path.join(OUT, "gate.json")))
    if kind == "learned":
        g = LogisticGate.from_dict(gd)
    else:
        g = ConfidenceGate(thr)
        g.tau1, g.tau2 = thr, 2.0
    r = Controller(policy, "gate", gate=g, hold=gd["hold"], seed=seed).run(make_instance(seed, N))
    return r["n_cspibt"], r["steps"]


def calibrate_conf(procs, n=6):
    """Match the confidence gate's CS-PIBT rate to the learned gate's, both measured closed-loop
    on validation instances (the first run matched open-loop rates, which did not transfer)."""
    inst = [(30500 + 1000 * N + k, N) for N in [64, 128, 192] for k in range(n)]
    rate = lambda res: sum(a for a, _ in res) / max(sum(b for _, b in res), 1)
    with Pool(procs) as p:
        target = rate(p.map(_closed_loop_rate, [("learned", 0, s, N) for s, N in inst]))
        lo, hi = 0.0, 1.0  # threshold on 1 - mean max p; higher -> fewer triggers
        for _ in range(7):
            mid = (lo + hi) / 2
            r = rate(p.map(_closed_loop_rate, [("conf", mid, s, N) for s, N in inst]))
            print(f"  conf_thr={mid:.4f} closed-loop rate {r:.3f} (target {target:.3f})", flush=True)
            lo, hi = (mid, hi) if r > target else (lo, mid)
    gd = json.load(open(os.path.join(OUT, "gate.json")))
    gd.update(conf_thr=(lo + hi) / 2, closed_loop_rate=target)
    json.dump(gd, open(os.path.join(OUT, "gate.json"), "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["policy", "gate", "select", "calibrate", "all"], default="all")
    ap.add_argument("--policy-inst", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--gate-train", type=int, default=120)
    ap.add_argument("--gate-val", type=int, default=40)
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.stage in ("policy", "all"):
        train_policy(a.policy_inst, a.epochs, a.procs)
    if a.stage in ("gate", "all"):
        train_gate(a.gate_train, a.gate_val, a.procs)
    if a.stage in ("gate", "select", "all"):
        select_operating_point(a.procs)
    if a.stage in ("gate", "select", "calibrate", "all"):
        calibrate_conf(a.procs)
