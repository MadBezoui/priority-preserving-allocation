"""Post-review exploratory diagnostics. Does not alter the original experiment grid.

Outputs: marginal-tier ratio fairness; two relative-effect conventions; robustness
by true tier; paired original timings and a new batched affine-combine microbenchmark;
random checks of the observed-increment certificate. No author manuscript included.
"""
from pathlib import Path
import sys,time,json,hashlib,platform
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from abo.allocation import priority_project,feedback,tier_budgets
from run_experiments import trajectory
R=ROOT/'results'; SEED=20261019

def ci(v,stat=np.mean):
    v=np.asarray(v);rng=np.random.default_rng(SEED)
    b=stat(v[rng.integers(0,len(v),(5000,len(v)))],axis=1)
    return float(stat(v)),*map(float,np.quantile(b,[.025,.975]))

def marginal(d,c,g):
    return next((k for k,(idx,b) in enumerate(zip(g,tier_budgets(d,c,g)))
                 if 1e-10<b<d[idx].sum()-1e-10),-1)

def fairness():
    dat=np.load(ROOT/'data/processed/electricity.npz');ds=dat['demand']
    g=[dat['critical'],np.setdiff1d(np.arange(ds.shape[1]),dat['critical'])]
    rows=[]
    for block,start in enumerate(np.linspace(0,len(ds)-336,30,dtype=int)):
        d=ds[start:start+336];c=np.full(336,.9*float(dat['train_total']));c[112:224]*=.5
        targets=np.array([priority_project(z,z,b,g) for z,b in zip(d,c)])
        for method in ['exact','feedback','proportional']:
            if method=='proportional':
                x=np.zeros_like(d)
                for t in range(len(d)):
                    for idx,b in zip(g,tier_budgets(d[t],c[t],g)):
                        if d[t,idx].sum()>0:x[t,idx]=b*d[t,idx]/d[t,idx].sum()
            else:x,_=trajectory(d,c,g,method,0,targets)
            for t in range(len(d)):
                k=marginal(d[t],c[t],g)
                if k<0:continue
                idx=g[k][d[t,g[k]]>1e-12]
                if len(idx)<2:continue
                ratios=x[t,idx]/d[t,idx]
                jain=ratios.sum()**2/(len(idx)*np.sum(ratios**2))
                rows.append(dict(block=block,t=t,tier=k+1,method=method,n=len(idx),jain=jain,ratio_sd=np.std(ratios)))
    raw=pd.DataFrame(rows);raw.to_csv(R/'review_fairness.csv',index=False)
    blocks=raw.groupby(['block','method'])[['jain','ratio_sd']].mean().reset_index()
    blocks.to_csv(R/'review_fairness_blocks.csv',index=False)
    return blocks.groupby('method')[['jain','ratio_sd']].mean().to_dict('index')

def conventions():
    df=pd.read_csv(R/'replay.csv');df=df[(df.ratio==.9)&(df.delay==4)]
    rows=[]
    for dataset in ['electricity','bike']:
        for metric in ['variation','quadratic_shortfall']:
            p=df[df.dataset==dataset].pivot(index='block',columns='method',values=metric)
            a=p.feedback.to_numpy();b=p.exact.to_numpy();assert np.all(b>0)
            for convention in ['mean_of_ratios','ratio_of_means']:
                if convention=='mean_of_ratios':est,lo,hi=ci(100*(a/b-1))
                else:
                    ix=np.random.default_rng(SEED).integers(0,len(a),(5000,len(a)))
                    vals=100*(a[ix].mean(axis=1)/b[ix].mean(axis=1)-1)
                    est=100*(a.mean()/b.mean()-1);lo,hi=np.quantile(vals,[.025,.975])
                rows.append(dict(dataset=dataset,metric=metric,convention=convention,percent_change=est,ci_low=lo,ci_high=hi))
    pd.DataFrame(rows).to_csv(R/'review_relative_effects.csv',index=False)
    return rows

def robustness():
    rows=[];summ=[];g=[np.arange(8),np.arange(8,32)]
    for seed in range(30):
        rng=np.random.default_rng(20261100+seed);phases=rng.uniform(0,2*np.pi,32);t=np.arange(256)
        d=1+.65*np.sin(2*np.pi*8*t[:,None]/256+phases)
        cap=np.full(256,24.);cap[85:170]=12.
        obs=np.maximum(0,d+rng.uniform(-.1,.1,d.shape));co=cap+rng.uniform(-3.2,3.2,len(t))
        for method in ['observed','conservative']:
            x=np.zeros(32);total=0
            for j in t:
                dh=obs[j] if method=='observed' else np.maximum(0,obs[j]-.1)
                ch=co[j] if method=='observed' else max(0,co[j]-3.2)
                x=feedback(x,dh,ch,g,.25);shortage=np.maximum(d[j]-x,0)
                total+=shortage.sum()
                rows.append(dict(seed=seed,t=j,method=method,true_marginal=marginal(d[j],cap[j],g)+1,
                    used_marginal=marginal(dh,ch,g)+1,
                    high_shortage=shortage[g[0]].sum()/d.sum(),
                    ordinary_shortage=shortage[g[1]].sum()/d.sum(),
                    high_demand=d[j,g[0]].sum()/d.sum(),ordinary_demand=d[j,g[1]].sum()/d.sum()))
            summ.append(dict(seed=seed,method=method,total_shortage=total/d.sum()))
    raw=pd.DataFrame(rows);raw.to_csv(R/'review_robustness_tiers.csv',index=False)
    old=pd.read_csv(R/'robustness.csv').sort_values(['seed','method'])
    fresh=pd.DataFrame(summ).sort_values(['seed','method'])
    np.testing.assert_allclose(old.total_shortage,fresh.total_shortage,atol=2e-15,rtol=0)
    byseed=raw.groupby(['seed','method'])[['high_shortage','ordinary_shortage','high_demand','ordinary_demand']].sum().reset_index()
    byseed.to_csv(R/'review_robustness_seeds.csv',index=False)
    out={}
    for method in ['observed','conservative']:
        z=raw[raw.method==method];s=byseed[byseed.method==method]
        out[method]={key:float(s[key].mean()) for key in ['high_shortage','ordinary_shortage']}
        out[method]['marginal_changed_fraction']=float((z.true_marginal!=z.used_marginal).mean())
        out[method]['true_marginal_counts']=z.true_marginal.value_counts().to_dict()
        out[method]['used_marginal_counts']=z.used_marginal.value_counts().to_dict()
    return out

def timings():
    raw=pd.read_csv(R/'runtime.csv');s=raw[raw.n==32].pivot(index='repeat',columns='method',values='microseconds')
    diff=(s.feedback-s.exact).to_numpy()
    out={'original_paired_mean_us':ci(diff),'original_paired_median_us':ci(diff,np.median),
         'original_difference_q05_q95_us':list(map(float,np.quantile(diff,[.05,.95])))}
    rng=np.random.default_rng(SEED);rows=[]
    # Batching reduces timer overhead, but has different cache and load conditions
    # from the original single-call suite: diagnostic, not causal decomposition.
    for repeat in range(60):
        prev=rng.uniform(0,2,32);d=rng.uniform(.1,3,32);g=[np.arange(8),np.arange(8,32)];cap=24.
        funcs={'affine':lambda:.75*prev+.25*d,
               'exact':lambda:priority_project(d,d,cap,g),
               'feedback':lambda:feedback(prev,d,cap,g,.25)}
        for fn in funcs.values():fn()
        for name in rng.permutation(list(funcs)):
            start=time.perf_counter_ns()
            for _ in range(100):funcs[name]()
            elapsed=(time.perf_counter_ns()-start)/100/1000
            rows.append(dict(repeat=repeat,method=name,microseconds=elapsed,batch=100))
    a=pd.DataFrame(rows);a.to_csv(R/'review_timing_batches.csv',index=False)
    out['batched_affine_median_us']=float(a[a.method=='affine'].microseconds.median())
    p=a.pivot(index='repeat',columns='method',values='microseconds')
    out['batched_paired_difference_us']=ci((p.feedback-p.exact).to_numpy(),np.median)
    return out

def certificates():
    rng=np.random.default_rng(SEED);g=[np.arange(3),np.arange(3,7),np.arange(7,12)];rows=[]
    prevd=np.zeros(12);prevc=0.;prevstar=np.zeros(12);x=np.zeros(12);B=0
    for t in range(2000):
        d=rng.uniform(0,4,12);c=rng.uniform(0,48);star=priority_project(d,d,c,g)
        delta=np.linalg.norm(star-prevstar);obs=np.sqrt(2)*np.abs(d-prevd).sum()+abs(c-prevc)
        x=feedback(x,d,c,g,.25);B=.75*(B+obs);err=np.linalg.norm(x-star)
        rows.append(dict(t=t,delta=delta,data_bound=obs,error=err,accumulated_bound=B,
                         marginal_switch=marginal(d,c,g)!=marginal(prevd,prevc,g)))
        prevd,prevc,prevstar=d,c,star
    df=pd.DataFrame(rows);df.to_csv(R/'review_data_certificate.csv',index=False)
    assert np.all(df.delta<=df.data_bound+1e-9) and np.all(df.error<=df.accumulated_bound+1e-9)
    return {'checks':len(df),'switches':int(df.marginal_switch.sum()),'max_increment_excess':float((df.delta-df.data_bound).max())}

def render(summary):
    timing=summary['timing'];m,lo,hi=timing['batched_paired_difference_us']
    macros={'ReviewAffine':timing['batched_affine_median_us'],
            'ReviewBatchDifference':m,'ReviewBatchLow':lo,'ReviewBatchHigh':hi}
    (R/'review_numbers.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+f'{v:.2f}'+'}' for k,v in macros.items())+'\n')
    f=pd.read_csv(R/'review_relative_effects.csv');lines=[]
    for _,r in f[f.dataset=='electricity'].iterrows():
        metric='Movement' if r.metric=='variation' else 'Shortfall'
        conv='Mean of block ratios' if r.convention=='mean_of_ratios' else 'Ratio of block means'
        lines.append(f'{metric} & {conv} & {r.percent_change:.2f} & [{r.ci_low:.2f}, {r.ci_high:.2f}] '+r'\\')
    (R/'table_review_ratios.tex').write_text(r'\begin{tabular}{llrr}\toprule'+'\n'+r'Metric & Convention & Change (\%) & Interval (\%)\\\midrule'+'\n'+'\n'.join(lines)+'\n'+r'\bottomrule\end{tabular}'+'\n')

def run():
    summary={'classification':'post-review exploratory diagnostics', 'seed':SEED,
             'fairness':fairness(),'relative_effects':conventions(),'robustness':robustness(),
             'data_certificate':certificates(),'timing':timings()}
    (R/'review_summary.json').write_text(json.dumps(summary,indent=2))
    render(summary)
    h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    sources=[Path(__file__),ROOT/'src/abo/allocation.py',ROOT/'scripts/run_experiments.py',ROOT/'scripts/robustness.py']
    inputs=[R/'runtime.csv',R/'replay.csv',R/'robustness.csv',ROOT/'data/processed/electricity.npz']
    prov={'python':sys.version,'platform':platform.platform(),'source_sha256':{str(p.relative_to(ROOT)):h(p) for p in sources},
          'input_sha256':{str(p.relative_to(ROOT)):h(p) for p in inputs},
          'output_sha256':{str(p.relative_to(ROOT)):h(p) for p in R.glob('review_*.csv')}}
    (R/'review_provenance.json').write_text(json.dumps(prov,indent=2))
    print(json.dumps(summary,indent=2))
if __name__=='__main__':run()
