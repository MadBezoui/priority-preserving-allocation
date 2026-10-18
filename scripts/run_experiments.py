"""Execute all prespecified historical, synthetic, certificate and timing studies."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import os
import time
import json
import platform
import hashlib
import argparse
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize
from abo.allocation import priority_project, hard_project, feedback, capped_simplex, tier_budgets

OUT=ROOT/"results"
SEED=20261018
METHODS=["feedback", "exact", "delayed_hard", "delayed_priority", "filtered_exact",
         "uniform", "no_projection", "alpha_010", "alpha_050", "alpha_075"]
DELAYED={"delayed_hard", "delayed_priority", "filtered_exact"}


def trajectory(demands, capacities, groups, method, delay=0, targets=None):
    T,n=demands.shape
    if targets is None:
        targets=np.array([priority_project(d,d,c,groups) for d,c in zip(demands,capacities)])
    previous=targets[0].copy()  # common warm initialization
    trajectory=np.empty_like(demands)
    for t,(d,c) in enumerate(zip(demands,capacities)):
        stale=targets[max(0,t-delay)]
        if method=="exact": x=targets[t].copy()
        elif method=="delayed_hard": x=hard_project(stale,d,c)
        elif method=="delayed_priority": x=priority_project(stale,d,c,groups)
        elif method=="filtered_exact": x=priority_project(.75*previous+.25*stale,d,c,groups)
        elif method=="uniform": x=feedback(previous,d,c,[np.arange(n)],.25)
        elif method=="no_projection": x=.75*previous+.25*d
        else:
            a={"feedback":.25,"alpha_010":.1,"alpha_050":.5,"alpha_075":.75}[method]
            x=feedback(previous,d,c,groups,a)
        trajectory[t]=x
        previous=x
    return trajectory,targets


def metrics(x,d,c,g,target,scale):
    T,n=d.shape
    shortage=np.maximum(d-x,0)
    # Prefix benchmark is consistent with lexicographic priority.
    prefix_regret=[]
    for k in range(len(g)):
        idx=np.concatenate(g[:k+1])
        ideal=np.minimum(c,d[:,idx].sum(axis=1))
        prefix_regret.append(np.maximum(ideal-x[:,idx].sum(axis=1),0))
    initial=target[0]
    q=.5*np.sum((d-x)**2,axis=1)
    qstar=.5*np.sum((d-target)**2,axis=1)
    return {
        "critical_shortage":float(shortage[:,g[0]].sum()/max(d[:,g[0]].sum(),1e-12)),
        "total_shortage":float(shortage.sum()/max(d.sum(),1e-12)),
        "priority_regret":float(np.mean(prefix_regret[0])/scale),
        "prefix_regret":float(np.mean(np.sum(prefix_regret,axis=0))/scale),
        "capacity_violation":float(np.max(np.maximum(x.sum(axis=1)-c,0))/scale),
        "box_violation":float(max(np.max(x-d),np.max(-x),0)),
        "quadratic_shortfall":float(q.mean()/n),
        "quadratic_gap":float((q-qstar).mean()/n),
        "variation":float(np.abs(np.diff(np.vstack([initial,x]),axis=0)).sum()/T/scale),
        "tracking_rmse":float(np.sqrt(np.mean((x-target)**2))),
    }


def replay():
    rows=[]; block_meta=[]; saved={}
    for name,count in [("electricity",30),("bike",24)]:
        data=np.load(ROOT/"data/processed"/(name+".npz"))
        all_d=data["demand"]; scale=float(data["train_total"])
        groups=[data["critical"],np.setdiff1d(np.arange(all_d.shape[1]),data["critical"])]
        starts=np.linspace(0,len(all_d)-336,count,dtype=int)
        assert np.all(np.diff(starts)>=336)
        for block,start in enumerate(starts):
            d=all_d[start:start+336]
            block_meta.append({"dataset":name,"block":block,"start_row":int(start),
                               "start":str(data["timestamps"][start]),
                               "end":str(data["timestamps"][start+335]),"length":len(d)})
            for ratio in [.6,.9,1.2]:
                c=np.full(len(d),ratio*scale); c[112:224]*=.5
                target=np.array([priority_project(v,v,cap,groups) for v,cap in zip(d,c)])
                cache={}
                for delay in [0,1,4,12]:
                    for method in METHODS:
                        if method not in DELAYED and method in cache:
                            x,m=cache[method]
                        else:
                            x,_=trajectory(d,c,groups,method,delay,target)
                            m=metrics(x,d,c,groups,target,scale)
                            if method not in DELAYED: cache[method]=(x,m)
                        rows.append(dict(dataset=name,block=block,ratio=ratio,delay=delay,method=method,**m))
                        if block==0 and ratio==.9 and delay==4:
                            saved[name+"_"+method]=x
                if block==0 and ratio==.9:
                    saved[name+"_demand"]=d; saved[name+"_capacity"]=c
            print(f"replay {name} block {block+1}/{count}",flush=True)
            pd.DataFrame(rows).to_csv(OUT/"replay.csv",index=False)
    pd.DataFrame(block_meta).to_csv(OUT/"blocks.csv",index=False)
    np.savez_compressed(OUT/"example_trajectories.npz",**saved)


def synthetic():
    rows=[]
    # Independent seeded phase configurations; no observed-demand forecast.
    for seed in range(20):
        rng=np.random.default_rng(SEED+seed)
        phases=rng.uniform(0,2*np.pi,32)
        groups=[np.arange(8),np.arange(8,32)]
        for cycles in [.5,2.,8.,32.]:
            t=np.arange(256)
            d=1+.65*np.sin(2*np.pi*cycles*t[:,None]/256+phases)
            c=np.full(len(t),24.);c[85:170]=12.
            target=np.array([priority_project(v,v,cap,groups) for v,cap in zip(d,c)])
            for delay in [0,1,4,12]:
                for method in METHODS[:7]:
                    x,_=trajectory(d,c,groups,method,delay,target)
                    rows.append(dict(seed=seed,cycles=cycles,delay=delay,method=method,
                                     **metrics(x,d,c,groups,target,32.)))
        print("synthetic seed",seed,flush=True)
    pd.DataFrame(rows).to_csv(OUT/"synthetic.csv",index=False)


def certificates():
    rng=np.random.default_rng(SEED)
    rows=[]
    for seed in range(20):
        groups=[np.arange(8),np.arange(8,32)]
        x=np.zeros(32);prev_target=np.zeros(32);bound=0.
        for t in range(100):
            d=rng.uniform(0,3,32);c=rng.uniform(0,50);a=.25
            target=priority_project(d,d,c,groups)
            step=np.linalg.norm(target-prev_target)
            new=feedback(x,d,c,groups,a)
            bound=.75*(bound+step)
            err=np.linalg.norm(new-target)
            rows.append(dict(seed=seed,t=t,error=err,bound=bound,
                contraction_excess=err-.75*np.linalg.norm(x-target),
                cap_excess=new.sum()-c,
                priority_excess=abs(new[:8].sum()-min(c,d[:8].sum()))))
            x,prev_target=new,target
    pd.DataFrame(rows).to_csv(OUT/"certificates.csv",index=False)
    # Fixed-environment recovery of the economic target, after an allocation perturbation.
    d=np.linspace(.4,2.5,32);C=19.;g=[np.arange(8),np.arange(8,32)]
    target=priority_project(d,d,C,g);start=priority_project(d[::-1],d,C,g)
    rows=[]
    for a in [.1,.25,.5,.75,1.]:
        x=start.copy();e0=np.linalg.norm(x-target)
        for t in range(51):
            rows.append(dict(alpha=a,t=t,error=np.linalg.norm(x-target),bound=(1-a)**t*e0))
            x=feedback(x,d,C,g,a)
    pd.DataFrame(rows).to_csv(OUT/"recovery.csv",index=False)


def runtime():
    rng=np.random.default_rng(SEED)
    rows=[];checks=[]
    for n in [8,32,128,512,2048]:
        groups=[np.arange(n//4),np.arange(n//4,n)]
        for repeat in range(60):
            d=rng.uniform(.2,2,n);C=.6*d.sum()
            prev=priority_project(rng.normal(1,1,n),d,C,groups)
            # Randomize method order to avoid a systematic order advantage.
            methods=["exact","feedback"] + (["generic_qp"] if n<=128 else [])
            rng.shuffle(methods)
            for method in methods:
                start=time.perf_counter_ns()
                if method=="exact": x=priority_project(d,d,C,groups)
                elif method=="feedback":x=feedback(prev,d,C,groups,.25)
                else:
                    budgets=tier_budgets(d,C,groups)
                    A=np.zeros((len(groups),n))
                    for j,g in enumerate(groups):A[j,g]=1
                    opt=minimize(lambda x:.5*np.sum((x-d)**2),prev,
                        jac=lambda x:x-d,method="SLSQP",bounds=list(zip(np.zeros(n),d)),
                        constraints={"type":"eq","fun":lambda x:A@x-budgets,"jac":lambda x:A},
                        options={"ftol":1e-10,"maxiter":200})
                    x=opt.x
                elapsed=(time.perf_counter_ns()-start)/1000
                rows.append(dict(n=n,repeat=repeat,method=method,microseconds=elapsed))
                if method=="generic_qp":
                    checks.append(dict(n=n,repeat=repeat,success=bool(opt.success),iterations=opt.nit,
                        max_error=float(np.max(np.abs(x-priority_project(d,d,C,groups))))))
        print("runtime n",n,flush=True)
    pd.DataFrame(rows).to_csv(OUT/"runtime.csv",index=False)
    pd.DataFrame(checks).to_csv(OUT/"qp_checks.csv",index=False)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--part",choices=["all","replay","synthetic","certificates","runtime"],default="all")
    args=parser.parse_args();OUT.mkdir(exist_ok=True)
    started=time.time()
    funcs={"replay":replay,"synthetic":synthetic,"certificates":certificates,"runtime":runtime}
    for name,func in funcs.items():
        if args.part in ["all",name]:func()
    source_files=sorted(list((ROOT/"src").rglob("*.py"))+list((ROOT/"scripts").glob("*.py")))
    meta={"seed":SEED,"part":args.part,"elapsed_seconds":time.time()-started,
          "python":sys.version,"numpy":np.__version__,"scipy":scipy.__version__,
          "pandas":pd.__version__,"platform":platform.platform(),"processor":platform.processor(),
          "threads":{k:os.environ.get(k) for k in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","VECLIB_MAXIMUM_THREADS"]},
          "source_sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}}
    (OUT/("environment_"+args.part+".json")).write_text(json.dumps(meta,indent=2))
    print("Completed",args.part,"in",time.time()-started,"seconds",flush=True)
