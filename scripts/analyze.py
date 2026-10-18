"""Generate tables and publication figures only from saved experimental outputs."""
from pathlib import Path
import json
import hashlib
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]; R=ROOT/"results"; F=ROOT/"figures"
F.mkdir(exist_ok=True)
LABELS={"feedback":"Projected feedback", "exact":"Compiled exact",
        "delayed_hard":"Delayed + hard filter", "delayed_priority":"Delayed + priority filter",
        "filtered_exact":"Smoothed delayed exact", "uniform":"No priority", "no_projection":"No projection"}
COLORS={"feedback":"#0072B2","exact":"#333333","delayed_hard":"#D55E00",
        "delayed_priority":"#009E73","filtered_exact":"#CC79A7","uniform":"#E69F00","no_projection":"#777777"}
plt.rcParams.update({"font.size":9,"axes.labelsize":9,"axes.titlesize":10,
                     "legend.fontsize":8,"font.family":"DejaVu Sans",
                     "axes.spines.top":False,"axes.spines.right":False,
                     "pdf.fonttype":42,"savefig.dpi":300})


def save(fig,name):
    fig.savefig(F/(name+".pdf"),bbox_inches="tight")
    fig.savefig(F/(name+".png"),bbox_inches="tight",dpi=220)
    plt.close(fig)


def bootstrap(values,seed=20261018):
    values=np.asarray(values);rng=np.random.default_rng(seed)
    samples=values[rng.integers(0,len(values),(5000,len(values)))].mean(axis=1)
    lo,hi=np.quantile(samples,[.025,.975])
    return float(values.mean()),float(lo),float(hi)


def run():
    df=pd.read_csv(R/"replay.csv"); syn=pd.read_csv(R/"synthetic.csv")
    primary=df[(df.ratio==.9)&(df.delay==4)]
    metric_cols=["critical_shortage","total_shortage","priority_regret","prefix_regret",
                 "capacity_violation","box_violation","quadratic_shortfall","quadratic_gap","variation","tracking_rmse"]
    summary=df.groupby(["dataset","ratio","delay","method"])[metric_cols].mean().reset_index()
    summary.to_csv(R/"summary_all.csv",index=False)
    primary.groupby(["dataset","method"])[metric_cols].mean().to_csv(R/"primary_summary.csv")
    intervals=[]
    for name,g in primary.groupby("dataset"):
        for comparator in ["exact","delayed_hard","delayed_priority","filtered_exact"]:
            for metric in ["variation","quadratic_shortfall","priority_regret"]:
                table=g.pivot(index="block",columns="method",values=metric)
                delta=table.feedback-table[comparator]
                mean,lo,hi=bootstrap(delta)
                intervals.append(dict(dataset=name,comparator=comparator,metric=metric,
                                      mean=mean,ci_low=lo,ci_high=hi,n=len(delta)))
        table=g.pivot(index="block",columns="method",values="variation")
        delta=100*(table.feedback/table.exact-1)
        mean,lo,hi=bootstrap(delta)
        intervals.append(dict(dataset=name,comparator="exact",metric="variation_relative_percent",
                              mean=mean,ci_low=lo,ci_high=hi,n=len(delta)))
    pd.DataFrame(intervals).to_csv(R/"paired_intervals.csv",index=False)
    # Small LaTeX tables are generated, not hand-transcribed.
    table_lines=[]
    for name in ["electricity","bike"]:
        table_lines.append(r"\textit{"+name.capitalize()+r"} & & & & \\")
        for method in LABELS:
            row=primary[(primary.dataset==name)&(primary.method==method)][metric_cols].mean()
            table_lines.append(f"{LABELS[method]} & {100*row.priority_regret:.3f} & {row.quadratic_shortfall:.4f} & {row.variation:.4f} & {100*row.capacity_violation:.3f} \\\\")
    (R/"table_primary.tex").write_text(r"\begin{tabular}{lrrrr}\toprule"+"\n"+r"Controller & $100R_1$ & $\overline Q$ & $V$ & Cap. excess (\%)\\\midrule"+"\n"+"\n".join(table_lines)+"\n"+r"\bottomrule\end{tabular}"+"\n")
    lines=[]
    for item in intervals:
        if item['dataset']=="electricity" and item['metric'] in ["variation","quadratic_shortfall"]:
            lines.append(f"{LABELS[item['comparator']]} & {'Movement' if item['metric']=='variation' else 'Shortfall loss'} & {item['mean']:.5f} & [{item['ci_low']:.5f}, {item['ci_high']:.5f}] \\\\")
    (R/"table_intervals.tex").write_text(r"\begin{tabular}{llrr}\toprule"+"\n"+r"Comparator & Metric & Mean difference & Interval\\\midrule"+"\n"+"\n".join(lines)+"\n"+r"\bottomrule\end{tabular}"+"\n")

    fig,axes=plt.subplots(1,2,figsize=(7.1,2.8),layout="constrained")
    base=primary[primary.dataset=="electricity"]
    methods=["exact","feedback","filtered_exact","delayed_priority","delayed_hard"]
    for ax,metric,title in zip(axes,["variation","quadratic_shortfall"],["(a) Allocation movement","(b) Quadratic shortfall"]):
        for j,m in enumerate(methods):
            mean,lo,hi=bootstrap(base[base.method==m][metric])
            ax.errorbar(j,mean,yerr=[[mean-lo],[hi-mean]],fmt='o',color=COLORS[m],capsize=3)
        ax.set_xticks(range(len(methods)),["Exact","Feedback","Smoothed\ndelayed","Priority\nfilter","Hard\nfilter"],rotation=20)
        ax.set_ylabel("Normalized movement / epoch" if metric=="variation" else "Loss / recipient / epoch")
        ax.set_title(title);ax.grid(axis='y',alpha=.2)
    save(fig,"primary_tradeoff")

    fig,axes=plt.subplots(1,2,figsize=(7.1,2.7),layout="constrained")
    for ax,metric,title in zip(axes,["priority_regret","quadratic_gap"],["(a) Priority deficit","(b) Excess shortfall loss"]):
        for method in ["feedback","exact","delayed_hard","delayed_priority","filtered_exact"]:
            s=summary[(summary.dataset=="electricity")&(summary.ratio==.9)&(summary.method==method)]
            ax.plot(s.delay,100*s[metric] if metric=="priority_regret" else s[metric],marker='o',ms=3,label=LABELS[method],color=COLORS[method])
        ax.set_xlabel("Imposed command age (samples)");ax.set_title(title);ax.grid(alpha=.2)
    axes[0].set_ylabel("Deficit / training total (%)");axes[1].set_ylabel("Excess loss / recipient / epoch")
    axes[1].legend(fontsize=6.5)
    save(fig,"delay")

    fig,axes=plt.subplots(1,2,figsize=(7.1,2.6),layout="constrained")
    for ax,metric,title in zip(axes,["variation","quadratic_gap"],["(a) Smoothing","(b) Tracking cost"]):
        for ratio in [.6,.9,1.2]:
            s=df[(df.dataset=="electricity")&(df.delay==0)&(df.ratio==ratio)]
            vals=[s[s.method==m][metric].mean() for m in ["alpha_010","feedback","alpha_050","alpha_075","exact"]]
            ax.plot([.1,.25,.5,.75,1],vals,'o-',label=f"Capacity ratio {ratio}")
        ax.set_xlabel(r"Gain $\alpha$");ax.set_title(title);ax.grid(alpha=.2)
    axes[0].set_ylabel("Normalized movement / epoch");axes[1].set_ylabel("Excess loss / recipient / epoch");axes[0].legend(fontsize=7)
    save(fig,"gain_sensitivity")

    fig,axes=plt.subplots(1,2,figsize=(7.1,2.6),layout="constrained")
    for ax,metric,title in zip(axes,["quadratic_gap","variation"],["(a) Shortfall tradeoff","(b) Movement tradeoff"]):
        for m in ["feedback","exact","delayed_priority","filtered_exact"]:
            s=syn[(syn.delay==4)&(syn.method==m)].groupby("cycles")[metric].mean()
            ax.plot(s.index,s.values,'o-',ms=3,label=LABELS[m],color=COLORS[m])
        ax.set_xscale("log",base=2);ax.set_xlabel("Demand cycles / 256 epochs");ax.set_title(title);ax.grid(alpha=.2)
    axes[0].set_ylabel("Excess loss / recipient / epoch");axes[1].set_ylabel("Normalized movement / epoch");axes[0].legend(fontsize=7)
    save(fig,"volatility")

    rt=pd.read_csv(R/"runtime.csv")
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.6),layout="constrained")
    for m,label,col in [("exact","Compiled exact","#333333"),("feedback","Projected feedback","#0072B2"),("generic_qp","Generic QP (SLSQP)","#D55E00")]:
        s=rt[rt.method==m].groupby("n").microseconds
        med=s.median();q=s.quantile(.95)
        axes[0].plot(med.index,med.values,'o-',label=label,color=col)
        axes[0].fill_between(med.index,med.values,q.values,color=col,alpha=.12)
    axes[0].set_xscale("log",base=2);axes[0].set_yscale("log");axes[0].set_xlabel("Recipients");axes[0].set_ylabel("Evaluation time (microseconds)");axes[0].set_title("(a) Median to 95th percentile");axes[0].legend(fontsize=7)
    rec=pd.read_csv(R/"recovery.csv")
    for a in [.1,.25,.5,.75]:
        s=rec[rec.alpha==a]
        axes[1].semilogy(s.t,np.maximum(s.error,1e-14),label=rf"$\alpha={a}$")
        axes[1].semilogy(s.t,np.maximum(s.bound,1e-14),'--',color='gray',alpha=.3)
    axes[1].set_xlabel("Updates after perturbation");axes[1].set_ylabel("Distance to fixed optimum");axes[1].set_title("(b) Recovery; dashed: bounds");axes[1].legend(fontsize=7)
    save(fig,"runtime_recovery")

    ex=np.load(R/"example_trajectories.npz");d=ex['electricity_demand'];c=ex['electricity_capacity']
    fig,axes=plt.subplots(2,1,figsize=(7.1,3.8),sharex=True,layout="constrained")
    axes[0].plot(c,color='black',ls='--',label="Capacity")
    axes[0].plot(d.sum(axis=1),color='#999999',label="Requested total")
    for m in ['feedback','exact','delayed_hard']:
        x=ex['electricity_'+m]
        axes[0].plot(x.sum(axis=1),color=COLORS[m],label=LABELS[m],alpha=.8)
        axes[1].plot(x[:,12],color=COLORS[m],label=LABELS[m])
    axes[0].set_ylabel("Normalized total");axes[0].legend(ncol=3,fontsize=7)
    axes[1].set_ylabel("Recipient 13 allocation");axes[1].set_xlabel("15-minute sample within first selected block")
    for ax in axes:ax.axvspan(112,224,color='#E69F00',alpha=.1);ax.grid(alpha=.15)
    save(fig,"example_trace")

    cert=pd.read_csv(R/"certificates.csv");qp=pd.read_csv(R/"qp_checks.csv")
    computed={"replay_rows":len(df),"synthetic_rows":len(syn),"blocks":df[['dataset','block']].drop_duplicates().shape[0],
              "max_certificate_excess":float((cert.error-cert.bound).max()),
              "max_capacity_excess":float(cert.cap_excess.max()),"qp_failures":int((~qp.success).sum()),
              "max_qp_difference":float(qp.max_error.max())}
    timing=rt.groupby(["n","method"]).microseconds.median()
    macros={"ExactThirtyTwo":f"{timing.loc[(32,'exact')]:.1f}",
            "FeedbackThirtyTwo":f"{timing.loc[(32,'feedback')]:.1f}",
            "GenericThirtyTwo":f"{timing.loc[(32,'generic_qp')]:.1f}",
            "ExactLarge":f"{timing.loc[(2048,'exact')]:.1f}",
            "FeedbackLarge":f"{timing.loc[(2048,'feedback')]:.1f}"}
    (R/"numbers.tex").write_text("\n".join("\\newcommand{\\"+k+"}{"+v+"}" for k,v in macros.items())+"\n")
    (R/"validation_summary.json").write_text(json.dumps(computed,indent=2))
    provenance={"python":sys.version,"numpy":np.__version__,"pandas":pd.__version__,
                "matplotlib":matplotlib.__version__,
                "analysis_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "input_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(R.glob("*.csv")) if p.name not in ["primary_summary.csv","summary_all.csv","paired_intervals.csv"]},
                "output_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(F.glob("*.pdf"))}}
    (R/"analysis_provenance.json").write_text(json.dumps(provenance,indent=2))
    print(json.dumps(computed,indent=2))
    print(pd.read_csv(R/"primary_summary.csv").to_string(index=False))
    print(pd.DataFrame(intervals).to_string(index=False))


if __name__=="__main__":run()
