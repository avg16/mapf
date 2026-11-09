"""Audit original runs and generate report evidence. Run from the repository root."""
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mapf-report-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "report/generated"
NAMES = {"shield": "Policy + shield", "pibt": "Plain PIBT", "cspibt": "CS-PIBT always",
         "gate": "Learned gate", "gate_lacam": "Gate + LaCAM", "gate_conf": "Confidence gate",
         "random": "Random gate", "random_lacam": "Random + LaCAM", "lacam": "Plain LaCAM",
         "lacam_policy": "Policy-guided LaCAM"}
ORDER = list(NAMES)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(ROOT / "artifacts/results.csv")
    summary = pd.read_csv(ROOT / "artifacts/summary.csv").set_index(["method", "N"])
    main_runs = raw[~raw.method.str.endswith("_sweep")]
    assert len(main_runs) == 1400 and len(raw) == 2500
    assert not raw.duplicated(["method", "N", "inst", "param"]).any()
    means = main_runs.groupby(["method", "N"]).agg(success=("success", "mean"),
        runtime=("runtime", "mean"), cspibt_calls=("n_cspibt", "mean"),
        search_rate=("search_rate", "mean"), lacam_calls=("n_lacam", "mean"))
    for col in ["runtime", "cspibt_calls", "search_rate", "lacam_calls"]:
        assert np.allclose(means[col], summary.loc[means.index, col], atol=0.000501), col
    ok = main_runs.pivot_table(index=["N", "inst"], columns="method", values="success").astype(bool)
    quality_methods = [m for m in ok if m not in ("shield", "gate_conf")]
    common = ok[ok[quality_methods].all(axis=1)].index
    quality = main_runs[main_runs.method.isin(quality_methods)].set_index(["N", "inst"])
    quality = quality[quality.index.isin(common)].reset_index().groupby(["method", "N"]).subopt.mean()
    assert np.allclose(quality, summary.loc[quality.index, "subopt_common"], atol=0.000501)
    counts = pd.Series(common.get_level_values("N")).value_counts().sort_index().to_dict()
    ns = sorted(main_runs.N.unique())
    lines = []
    for method in ORDER:
        vals = [f"{means.loc[(method,n),'success']*100:.0f}" for n in ns]
        lines.append(NAMES[method] + " & " + " & ".join(vals) + r" \\")
    (OUT / "success_rows.tex").write_text("\n".join(lines)+"\n"+r"\bottomrule"+"\n")
    lines = []
    for method in ORDER:
        r = summary.loc[(method,192)]
        q = "--" if pd.isna(r.subopt_common) else f"{r.subopt_common:.3f}"
        lines.append(f"{NAMES[method]} & {means.loc[(method,192),'success']*100:.0f} & "
            f"{r.cspibt_calls:.2f} & {r.lacam_calls:.2f} & {r.runtime:.3f} & {q}" + r" \\")
    (OUT / "dense_rows.tex").write_text("\n".join(lines)+"\n"+r"\bottomrule"+"\n")
    cs = means.loc["cspibt", "cspibt_calls"]
    hybrid = means.loc["gate_lacam", "cspibt_calls"]
    savings = 100*(1-hybrid/cs)
    lines = [f"{n} & {cs[n]:.2f} & {hybrid[n]:.2f} & {savings[n]:.1f} & {counts[n]}"+r" \\" for n in ns]
    (OUT / "calls_rows.tex").write_text("\n".join(lines)+"\n"+r"\bottomrule"+"\n")
    pooled = main_runs[main_runs.method=="gate"].groupby("N")
    pooled_rate = pooled.n_cspibt.sum()/pooled.steps.sum()
    random_prob = main_runs[main_runs.method=="random"].groupby("N").param.first()
    assert np.allclose(pooled_rate,random_prob)
    sweep = raw[raw.method.str.endswith("_sweep")].groupby(["method", "param"]).agg(
        success=("success", "mean"), rate=("search_rate", "mean")).reset_index()
    audit = dict(raw_runs=len(raw),main_runs=len(main_runs),unique_test_instances=len(ok),
        common_solved_counts={str(k):int(v) for k,v in counts.items()},
        call_savings_min=float(savings.min()),call_savings_max=float(savings.max()),
        hybrid_mean_lacam_calls=float(main_runs[main_runs.method=="gate_lacam"].n_lacam.mean()),
        random_probabilities={str(k):float(v) for k,v in random_prob.items()},
        summary_matches_raw=True,sweep=sweep.to_dict(orient="records"))
    (OUT/"metrics_audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,
        "axes.grid":True,"grid.alpha":.2,"savefig.dpi":180})
    colours={"cspibt":"#d57724","gate_lacam":"#245c98","random_lacam":"#389179","lacam":"#72777e"}
    fig, axes = plt.subplots(1,2,figsize=(10,3.6))
    for m,c in colours.items():
        d=summary.loc[m].sort_index()
        axes[0].plot(d.index,d.subopt_common,marker='o',markersize=3,label=NAMES[m],color=c)
        axes[1].plot(d.index,d.runtime,marker='o',markersize=3,label=NAMES[m],color=c)
    axes[0].set(ylabel="Flowtime / distance lower bound",xlabel="Agents")
    axes[1].set(ylabel="Mean recorded runtime (s)",xlabel="Agents")
    axes[0].legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'quality_runtime.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,3.5))
    for m,c,label in [('gate_sweep','#245c98','Learned gate'),('random_sweep','#d57724','Random gate')]:
        d=sweep[sweep.method==m].sort_values('rate')
        ax.plot(d.rate*100,d.success*100,'-o',color=c,label=label,markersize=4)
    ax.set(xlabel="Mean realised steps calling CS-PIBT (%)",ylabel="Success across 100 instances (%)",ylim=(0,100))
    ax.legend();fig.tight_layout();fig.savefig(OUT/'sweep.png');plt.close(fig)
    print('Audited 2,500 runs; saved tables, figures and metrics_audit.json to',OUT)


if __name__ == "__main__":
    main()
