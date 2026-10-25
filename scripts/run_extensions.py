"""Frozen extension experiments using the already archived electricity inputs."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import argparse
import hashlib
import json
import platform
import time
import numpy as np
import pandas as pd
import scipy
import osqp
from abo.allocation import priority_project, tier_budgets
from abo.extensions import (weighted_feedback,reachable_feedback,hierarchical_face,
    hierarchical_project,offline_horizon,horizon_cost,online_horizon_certificate)

OUT=ROOT/'results/extensions'
SEED=20261025
PROTOCOL={
 'date':'2026-10-25','seed':SEED,'dataset':'data/processed/electricity.npz',
 'blocks':30,'length':336,'capacity_ratio':.9,'drop_interval_zero_indexed':[112,224],
 'capacity_drop_factor':.5,'initialization':'Common unweighted exact target of first observation',
 'ramps':[.1,.5,2.],'gains':[.25,1.],'weight_exponents':[0,1,2],'weight_floor':.05,
 'hierarchy':{'seeds':10,'n':8,'length':128,'cycles':2,'demand_amplitude':.65,
              'root_capacity':4.8,'local_capacities':[1.8,3.2],'drop_interval':[42,85],
              'root_drop_factor':.75,'groups':[[0,1,4,5],[2,3,6,7]]},
 'offline':{'blocks':list(range(10)),'window':[80,176],'mu':3.,'relative_certificate_tolerance':1e-9},
 'static_shock':{'n':32,'requests':1.,'capacity_before':24.,'capacity_after':12.,'length':128},
 'feasibility_tolerance':1e-8,'historical_outputs_modified':False,
 'scope':'Constructed allocation scenarios. Neither electrical power-flow nor deployment data.'}


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def blocks():
    with np.load(ROOT/PROTOCOL['dataset']) as f:
        demand=f['demand'];scale=float(f['train_total']);critical=f['critical']
    groups=[critical,np.setdiff1d(np.arange(demand.shape[1]),critical)]
    starts=np.linspace(0,len(demand)-336,30,dtype=int)
    for b,start in enumerate(starts):
        d=demand[start:start+336].copy();c=np.full(336,.9*scale);c[112:224]*=.5
        targets=np.array([priority_project(dt,dt,ct,groups) for dt,ct in zip(d,c)])
        yield b,int(start),d,c,groups,targets,scale


def base_metrics(x,d,c,g,initial,scale):
    movement=np.abs(np.diff(np.vstack([initial,x]),axis=0)).sum()/len(x)/scale
    q=.5*np.mean(np.sum((x-d)**2,axis=1))/d.shape[1]
    return {'movement':float(movement),'unweighted_shortfall':float(q),
            'capacity_violation_max':float(np.maximum(x.sum(axis=1)-c,0).max()),
            'demand_violation_max':float(np.maximum(x-d,0).max())}


def ramp_suite():
    rows=[];epochs=[];saved={}
    for block,start,d,c,g,targets,scale in blocks():
        for r in PROTOCOL['ramps']:
            for alpha in PROTOCOL['gains']:
                initial=targets[0].copy();previous=initial.copy();xs=[];checks=[]
                for t,(dt,ct,target) in enumerate(zip(d,c,targets)):
                    step=reachable_feedback(previous,dt,ct,g,np.full(len(dt),r),alpha)
                    x=step.allocation;xs.append(x)
                    hard=max(step.capacity_violation,step.demand_violation)<=1e-8
                    gap=max(abs(x[group].sum()-target[group].sum()) for group in g)
                    record=dict(block=block,alpha=alpha,ramp=r,t=t,
                        emergency=step.emergency,hard_feasible=hard,
                        capacity_violation=step.capacity_violation,demand_violation=step.demand_violation,
                        ramp_violation=step.ramp_violation,
                        priority_total_gap=gap if hard else np.nan,
                        full_priority=bool(hard and gap<=1e-8),
                        total_allocation=x.sum(),capacity=ct,high_allocation=x[g[0]].sum(),
                        high_target=target[g[0]].sum())
                    epochs.append(record);checks.append(record);previous=x
                x=np.array(xs);f=pd.DataFrame(checks)
                rows.append(dict(block=block,start_row=start,alpha=alpha,ramp=r,
                    **base_metrics(x,d,c,g,initial,scale),
                    emergency_epochs=int(f.emergency.sum()),hard_feasible_fraction=float(f.hard_feasible.mean()),
                    full_priority_fraction=float(f.full_priority.mean()),
                    capacity_violation_sum=float(f.capacity_violation.sum()),
                    demand_violation_sum=float(f.demand_violation.sum()),
                    ramp_violation_max=float(f.ramp_violation.max())))
                saved[f'b{block}_r{r}_a{alpha}']=x
        print('ramp block',block,flush=True)
    pd.DataFrame(rows).to_csv(OUT/'ramps.csv',index=False)
    pd.DataFrame(epochs).to_csv(OUT/'ramp_epochs.csv',index=False)
    np.savez_compressed(OUT/'ramp_trajectories.npz',**saved)
    rows=[];epochs=[];g=[np.arange(8),np.arange(8,32)];d=np.ones(32)
    initial=priority_project(d,d,24.,g);target=priority_project(d,d,12.,g)
    for r in PROTOCOL['ramps']:
        for alpha in PROTOCOL['gains']:
            previous=initial.copy();first_hard=None;first_priority=None
            for t in range(128):
                s=reachable_feedback(previous,d,12.,g,np.full(32,r),alpha);x=s.allocation
                hard=max(s.capacity_violation,s.demand_violation)<=1e-8
                gap=max(abs(x[z].sum()-target[z].sum()) for z in g)
                full=hard and gap<=1e-8
                if hard and first_hard is None:first_hard=t+1
                if full and first_priority is None:first_priority=t+1
                epochs.append(dict(ramp=r,alpha=alpha,update=t+1,total=x.sum(),
                    high=x[g[0]].sum(),capacity_violation=s.capacity_violation,
                    priority_gap=gap if hard else np.nan,hard_feasible=hard,full_priority=full))
                previous=x
            lower=initial.copy();bound=0
            while lower.sum()>12.+1e-10:
                lower=np.maximum(lower-r,0);bound+=1
            rows.append(dict(ramp=r,alpha=alpha,hard_recovery_updates=first_hard,
                priority_recovery_updates=first_priority,hard_recovery_bound=bound,
                hard_censored=first_hard is None,priority_censored=first_priority is None))
    pd.DataFrame(rows).to_csv(OUT/'static_recovery.csv',index=False)
    pd.DataFrame(epochs).to_csv(OUT/'static_recovery_epochs.csv',index=False)


def weighted_suite():
    rows=[];epochs=[];saved={}
    for block,start,d,c,g,targets,scale in blocks():
        for beta in PROTOCOL['weight_exponents']:
            for alpha in PROTOCOL['gains']:
                previous=targets[0].copy();initial=previous.copy();xs=[];js=[];rsd=[]
                for t,(dt,ct) in enumerate(zip(d,c)):
                    w=np.maximum(dt,.05)**(-beta)
                    x=weighted_feedback(previous,dt,ct,g,w,alpha);xs.append(x)
                    budgets=tier_budgets(dt,ct,g);jain=np.nan;sd=np.nan;m=0
                    for group,b in zip(g,budgets):
                        if 1e-10<b<float(dt[group].sum())-1e-10:
                            ids=group[dt[group]>1e-12];m=len(ids)
                            if m>=2:
                                fractions=x[ids]/dt[ids]
                                if np.sum(fractions**2)>1e-24:
                                    jain=float(fractions.sum()**2/m/np.sum(fractions**2))
                                    sd=float(fractions.std());js.append(jain);rsd.append(sd)
                    epochs.append(dict(block=block,beta=beta,alpha=alpha,t=t,jain=jain,
                        ratio_sd=sd,retained_count=m,floor_coordinates=int(np.sum(dt<.05)),
                        quadratic_shortfall=.5*np.sum((dt-x)**2),
                        weighted_shortfall=.5*np.sum(w*(dt-x)**2)))
                    previous=x
                x=np.array(xs)
                rows.append(dict(block=block,start_row=start,beta=beta,alpha=alpha,
                    **base_metrics(x,d,c,g,initial,scale),jain=float(np.mean(js)),
                    ratio_sd=float(np.mean(rsd)),floor_fraction=float(np.mean(d<.05))))
                saved[f'b{block}_beta{beta}_a{alpha}']=x
        print('weighted block',block,flush=True)
    pd.DataFrame(rows).to_csv(OUT/'weights.csv',index=False)
    pd.DataFrame(epochs).to_csv(OUT/'weight_epochs.csv',index=False)
    np.savez_compressed(OUT/'weighted_trajectories.npz',**saved)


def hierarchy_suite():
    rows=[];epochs=[];saved={};subsets=[np.arange(8),np.arange(4),np.arange(4,8)]
    g=[np.array([0,1,4,5]),np.array([2,3,6,7])]
    for seed in range(10):
        rng=np.random.default_rng(SEED+seed);t=np.arange(128)
        d=1+.65*np.sin(2*np.pi*2*t[:,None]/128+rng.uniform(0,2*np.pi,8))
        c=np.tile([4.8,1.8,3.2],(128,1));c[42:85,0]*=.75
        faces=[hierarchical_face(dt,ct,subsets,g) for dt,ct in zip(d,c)]
        exact=[]
        for dt,ct,face in zip(d,c,faces):
            x,_=hierarchical_project(dt,dt,ct,subsets,g,face);exact.append(x)
        exact=np.array(exact)
        for alpha in PROTOCOL['gains']:
            previous=exact[0].copy();initial=previous.copy();xs=[];max_res=0.;multi=0;bound_excess=0.
            for k,(dt,ct,face) in enumerate(zip(d,c,faces)):
                x,info=hierarchical_project((1-alpha)*previous+alpha*dt,dt,ct,subsets,g,face)
                bound_excess=max(bound_excess,np.linalg.norm(x-exact[k])-(1-alpha)*np.linalg.norm(previous-exact[k]))
                totals=np.array([dt[z].sum() for z in g]);b=face[3]
                partial=int(np.sum((b>1e-8)&(b<totals-1e-8)));multi+=partial>=2
                max_res=max(max_res,info['residual']);xs.append(x)
                epochs.append(dict(seed=seed,alpha=alpha,t=k,partial_tiers=partial,
                    high_service=b[0],ordinary_service=b[1],qp_residual=info['residual'],
                    quadratic_shortfall=.5*np.sum((dt-x)**2),
                    movement=np.abs(x-previous).sum(),tracking_error=np.linalg.norm(x-exact[k])))
                previous=x
            x=np.array(xs)
            rows.append(dict(seed=seed,alpha=alpha,movement=float(np.abs(np.diff(np.vstack([initial,x]),axis=0)).sum()/128/8),
                unweighted_shortfall=float(.5*np.mean((x-d)**2)),
                multiple_partial_fraction=multi/128,qp_residual_max=max_res,
                contraction_excess_max=bound_excess))
            saved[f'seed{seed}_a{alpha}']=x
        saved[f'seed{seed}_demand']=d;saved[f'seed{seed}_capacities']=c
        print('hierarchy seed',seed,flush=True)
    pd.DataFrame(rows).to_csv(OUT/'hierarchy.csv',index=False)
    pd.DataFrame(epochs).to_csv(OUT/'hierarchy_epochs.csv',index=False)
    np.savez_compressed(OUT/'hierarchy_trajectories.npz',**saved)


def offline_suite():
    rows=[];saved={}
    for block,start,d,c,g,targets,scale in blocks():
        if block>=10:break
        d=d[80:176];c=c[80:176];targets=targets[80:176];initial=targets[0].copy();mu=3.
        offline,info=offline_horizon(d,c,g,initial,mu,tolerance=1e-9)
        upper=info['objective'];lower=upper-info['gap_bound'];saved[f'b{block}_offline']=offline
        for alpha in PROTOCOL['gains']:
            previous=initial.copy();xs=[]
            for dt,ct in zip(d,c):
                previous=priority_project((1-alpha)*previous+alpha*dt,dt,ct,g);xs.append(previous)
            x=np.array(xs);cost=horizon_cost(x,d,initial,mu)
            tight=online_horizon_certificate(x,d,initial,mu,(1-alpha)/alpha)
            loose=online_horizon_certificate(x,d,initial,mu,(1-alpha)/alpha,tight=False)
            assert cost-upper<=tight+1e-6 and tight<=loose+1e-8
            rows.append(dict(block=block,start_row=start+80,alpha=alpha,mu=mu,cost=cost,
                offline_lower=lower,offline_upper=upper,offline_iterations=info['iterations'],
                offline_certificate=info['gap_bound'],offline_relative_certificate=info['relative_gap_bound'],
                gap_lower=cost-upper,gap_upper=cost-lower,certificate_tight=tight,certificate_simple=loose,
                gap_percent_lower=100*(cost/upper-1),gap_percent_upper=100*(cost/lower-1)))
            saved[f'b{block}_a{alpha}']=x
        print('offline block',block,'iterations',info['iterations'],flush=True)
    pd.DataFrame(rows).to_csv(OUT/'offline.csv',index=False)
    np.savez_compressed(OUT/'offline_trajectories.npz',**saved)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--part',choices=['all','ramps','weights','hierarchy','offline'],default='all');args=p.parse_args()
    OUT.mkdir(exist_ok=True)
    protocol={**PROTOCOL,'input_sha256':sha(ROOT/PROTOCOL['dataset'])}
    protocol_path=OUT/'protocol.json'
    if protocol_path.exists():assert json.loads(protocol_path.read_text())==protocol
    else:protocol_path.write_text(json.dumps(protocol,indent=2)+'\n')
    started=time.time()
    for name,fn in [('ramps',ramp_suite),('weights',weighted_suite),('hierarchy',hierarchy_suite),('offline',offline_suite)]:
        if args.part in ['all',name]:fn()
    paths=['src/abo/allocation.py','src/abo/extensions.py','scripts/run_extensions.py','tests/test_extensions.py']
    meta={'part':args.part,'date':'2026-10-25','elapsed_seconds':time.time()-started,
        'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'osqp':osqp.__version__,'platform':platform.platform(),
        'protocol_sha256':sha(protocol_path),'source_sha256':{f:sha(ROOT/f) for f in paths},
        'output_sha256':{f.name:sha(f) for f in OUT.iterdir() if f.suffix in ['.csv','.npz']}}
    (OUT/f'environment_{args.part}.json').write_text(json.dumps(meta,indent=2)+'\n')
    print('Completed',args.part,'seconds',meta['elapsed_seconds'],flush=True)
