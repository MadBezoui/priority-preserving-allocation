"""Declared extension: bounded observation error and physical-ramp diagnostics.

This suite was specified after the initial protocol, before inspecting replay
results. It tests limits of certification, not a tuned primary comparison.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
import numpy as np
import pandas as pd
from abo.allocation import feedback, priority_project


def run():
    rows=[];ramp=[]
    for seed in range(30):
        rng=np.random.default_rng(20261100+seed)
        phases=rng.uniform(0,2*np.pi,32);g=[np.arange(8),np.arange(8,32)]
        t=np.arange(256)
        d=1+.65*np.sin(2*np.pi*8*t[:,None]/256+phases)
        cap=np.full(256,24.);cap[85:170]=12.
        d_obs=np.maximum(0,d+rng.uniform(-.1,.1,d.shape))
        c_obs=cap+rng.uniform(-3.2,3.2,len(t))
        for method in ["observed", "conservative"]:
            x=np.zeros(32);max_cap=0;max_box=0;shortage=0
            for j in t:
                dhat=d_obs[j] if method=="observed" else np.maximum(0,d_obs[j]-.1)
                chat=c_obs[j] if method=="observed" else max(0,c_obs[j]-3.2)
                x=feedback(x,dhat,chat,g,.25)
                max_cap=max(max_cap,x.sum()-cap[j]);max_box=max(max_box,float(np.max(x-d[j])))
                shortage+=np.maximum(d[j]-x,0).sum()
            rows.append(dict(seed=seed,method=method,max_capacity_violation=max_cap/32,
                             max_box_violation=max_box,total_shortage=shortage/d.sum()))
        exact=np.array([priority_project(z,z,c,g) for z,c in zip(d,cap)])
        for rate in [.02,.1,.5,2.]:
            for j in range(1,len(t)):
                lower=np.maximum(0,exact[j-1]-rate)
                upper=np.minimum(d[j],exact[j-1]+rate)
                feasible=bool(np.all(lower<=upper+1e-10) and lower.sum()<=cap[j]+1e-10)
                ramp.append(dict(seed=seed,t=j,rate=rate,reachable=feasible,
                                 capacity_lower_excess=max(0,lower.sum()-cap[j]),
                                 box_lower_excess=max(0,float(np.max(lower-upper)))))
    pd.DataFrame(rows).to_csv(ROOT/"results/robustness.csv",index=False)
    pd.DataFrame(ramp).to_csv(ROOT/"results/ramp_diagnostics.csv",index=False)


if __name__=="__main__":run()
