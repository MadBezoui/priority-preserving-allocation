"""Independently check the saved extension outputs. Does not regenerate experiments."""
from pathlib import Path
import sys, json, hashlib
import numpy as np
import pandas as pd
from scipy.optimize import brentq, linprog
from scipy import sparse
import osqp
R=Path(__file__).resolve().parents[1]
O=R/'results/extensions'
report={}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
meta=json.loads((O/'environment_all.json').read_text())
for file,digest in meta['source_sha256'].items():assert sha(R/file)==digest,file
for file,digest in meta['output_sha256'].items():assert sha(O/file)==digest,file
assert sha(O/'protocol.json')==meta['protocol_sha256']
p=json.loads((O/'protocol.json').read_text()); assert sha(R/p['dataset'])==p['input_sha256']
report['provenance']={'sources':len(meta['source_sha256']),'outputs':len(meta['output_sha256']),'protocol_and_input_verified':True}
f=np.load(R/p['dataset']);raw=f['demand'];scale=float(f['train_total']);n=raw.shape[1]
g=[f['critical'],np.setdiff1d(np.arange(n),f['critical'])]
starts=np.linspace(0,len(raw)-336,30,dtype=int)
def block(k):
 d=raw[starts[k]:starts[k]+336].copy(); c=np.full(336,.9*scale);c[112:224]*=.5
 return d,c

def budgets(d,c):
 totals=np.array([d[z].sum() for z in g]);return np.minimum(totals,np.maximum(0,c-np.r_[0,np.cumsum(totals[:-1])]))

def wp(v,d,c,w):
 out=np.zeros_like(d)
 for ids,b in zip(g,budgets(d,c)):
  if b<=0:continue
  if b>=d[ids].sum()-1e-14:out[ids]=d[ids];continue
  a=w[ids]*v[ids];u=d[ids];ww=w[ids]
  fun=lambda lam:np.clip((a-lam)/ww,0,u).sum()-b
  lam=brentq(fun,float(np.min(a-ww*u))-1.,float(np.max(a))+1.,xtol=1e-13)
  out[ids]=np.clip((a-lam)/ww,0,u)
 return out

def initial(d,c):return wp(d[0],d[0],c[0],np.ones(n))

def basic(x,d,c,x0):
 return {'movement':np.abs(np.diff(np.vstack([x0,x]),axis=0)).sum()/len(x)/scale,
 'unweighted_shortfall':.5*np.mean(np.sum((x-d)**2,axis=1))/n,
 'capacity_violation_max':np.maximum(x.sum(axis=1)-c,0).max(),
 'demand_violation_max':np.maximum(x-d,0).max()}

# All weighted records and raw trajectory invariants.
wfile=np.load(O/'weighted_trajectories.npz');wt=pd.read_csv(O/'weights.csv');we=pd.read_csv(O/'weight_epochs.csv')
maxw=0.;rows=0
for rr in wt.to_dict('records'):
 b=int(rr['block']);beta=int(rr['beta']);a=rr['alpha'];d,c=block(b);x=wfile[f'b{b}_beta{beta}_a{a}'];x0=initial(d,c)
 vals=basic(x,d,c,x0)
 for k,v in vals.items():assert abs(v-rr[k])<1e-10,(b,beta,a,k,v,rr[k])
 budget=np.array([budgets(dt,ct) for dt,ct in zip(d,c)])
 actual=np.column_stack([x[:,z].sum(axis=1) for z in g])
 assert np.max(np.abs(actual-budget))<1e-9
 assert x.min()>-1e-10 and np.max(x-d)<1e-10
 ee=we[(we.block==b)&(we.beta==beta)&(we.alpha==a)].sort_values('t')
 js=[];sds=[]
 for t,(dt,xx,bb) in enumerate(zip(d,x,budget)):
  ids=None
  for zz,amount in zip(g,bb):
   if 1e-10<amount<dt[zz].sum()-1e-10:ids=zz[dt[zz]>1e-12]
  if ids is not None and len(ids)>=2:
   ratios=xx[ids]/dt[ids]
   if np.sum(ratios**2)>1e-24:
    j=ratios.sum()**2/len(ids)/np.sum(ratios**2);sd=ratios.std();js.append(j);sds.append(sd)
    assert abs(j-ee.iloc[t].jain)<1e-10 and abs(sd-ee.iloc[t].ratio_sd)<1e-10
 assert abs(np.mean(js)-rr['jain'])<1e-10
 assert abs(np.mean(sds)-rr['ratio_sd'])<1e-10
 if b in [0,29]:
  prev=x0
  for dt,ct,xx in zip(d,c,x):
   pred=wp((1-a)*prev+a*dt,dt,ct,np.maximum(dt,.05)**(-beta))
   maxw=max(maxw,float(np.max(np.abs(pred-xx))));prev=pred
 rows+=len(x)
assert maxw<1e-8,maxw
report['weighted']={'trajectories':len(wt),'epochs':rows,'independent_brent_replay_trajectories':12,'max_coordinate_difference':maxw,'means':wt.groupby(['beta','alpha'])[['movement','unweighted_shortfall','jain','ratio_sd']].mean().reset_index().to_dict('records')}
print('weighted validated',flush=True)

# Closed-loop conditions checked at every recorded decision, using that own past.
rfile=np.load(O/'ramp_trajectories.npz');rt=pd.read_csv(O/'ramps.csv');re=pd.read_csv(O/'ramp_epochs.csv');maxr=0.;rows=0
for rr in rt.to_dict('records'):
 b=int(rr['block']);r=rr['ramp'];a=rr['alpha'];d,c=block(b);x=rfile[f'b{b}_r{r}_a{a}'];x0=initial(d,c)
 vals=basic(x,d,c,x0)
 for k,v in vals.items():assert abs(v-rr[k])<1e-9,(b,r,a,k,v,rr[k])
 ee=re[(re.block==b)&(re.ramp==r)&(re.alpha==a)].sort_values('t').reset_index(drop=True)
 prev=x0
 for t,(dt,ct,xx) in enumerate(zip(d,c,x)):
  L=np.maximum(0,prev-r);U=np.minimum(dt,prev+r)
  capv=max(0,xx.sum()-ct);demv=max(0,np.max(xx-dt));rampv=max(0,np.max(np.abs(xx-prev))-r)
  assert rampv<1e-9
  assert abs(capv-max(0,L.sum()-ct))<1e-9
  assert abs(demv-max(0,np.max(L-dt)))<1e-9
  assert abs(capv-ee.iloc[t].capacity_violation)<1e-9
  assert abs(demv-ee.iloc[t].demand_violation)<1e-9
  hard=max(capv,demv)<=1e-8
  assert bool(ee.iloc[t].hard_feasible)==hard
  if hard:
   gap=max(abs(xx[zz].sum()-bb) for zz,bb in zip(g,budgets(dt,ct)))
   assert abs(gap-ee.iloc[t].priority_total_gap)<1e-9
  else:assert pd.isna(ee.iloc[t].priority_total_gap)
  if b in [0,29]:
   softd=np.maximum(dt,L);softc=max(ct,L.sum());u=np.minimum(softd,prev+r)-L
   pred=L+wp((1-a)*prev+a*dt-L,u,max(0,softc-L.sum()),np.ones(n))
   maxr=max(maxr,float(np.max(np.abs(pred-xx))))
  prev=xx
 assert abs(ee.capacity_violation.sum()-rr['capacity_violation_sum'])<1e-8
 assert abs(ee.demand_violation.sum()-rr['demand_violation_sum'])<1e-8
 rows+=len(x)
assert maxr<1e-8,maxr
report['ramps']={'trajectories':len(rt),'epochs':rows,'independent_brent_decision_checks':12*336,'max_coordinate_difference':maxr,'means':rt.groupby(['ramp','alpha'])[['hard_feasible_fraction','full_priority_fraction','capacity_violation_sum','demand_violation_sum','ramp_violation_max']].mean().reset_index().to_dict('records')}
print('ramps validated',flush=True)

# Tree rank formula is independent of the sequential LP construction.
hf=np.load(O/'hierarchy_trajectories.npz');ht=pd.read_csv(O/'hierarchy.csv');he=pd.read_csv(O/'hierarchy_epochs.csv');hg=[np.array([0,1,4,5]),np.array([2,3,6,7])]
A=np.array([np.ones(8),[1,1,1,1,0,0,0,0],[0,0,0,0,1,1,1,1]])
E=np.array([[1,1,0,0,1,1,0,0],[0,0,1,1,0,0,1,1]])
rankerr=0.;minvi=0.;checks=0
for seed in range(10):
 d=hf[f'seed{seed}_demand'];c=hf[f'seed{seed}_capacities'];target=hf[f'seed{seed}_a1.0']
 for a in [.25,1.]:
  x=hf[f'seed{seed}_a{a}'];prev=target[0];multi=0;maxres=0.
  for t,(dt,ct,xx) in enumerate(zip(d,c,x)):
   rank=lambda ids:min(ct[0],sum(min(ct[j+1],sum(dt[i] for i in ids if i//4==j)) for j in range(2)))
   b1=rank(hg[0]);b2=rank(np.arange(8))-b1;b=np.array([b1,b2])
   record=he[(he.seed==seed)&(he.alpha==a)&(he.t==t)].iloc[0]
   rankerr=max(rankerr,float(np.max(np.abs(b-[record.high_service,record.ordinary_service]))))
   res=max(0,np.max(A@xx-ct),np.max(np.abs(E@xx-b)),np.max(-xx),np.max(xx-dt));maxres=max(maxres,res)
   assert res<1e-7
   partial=np.sum((b>1e-8)&(b<np.array([dt[z].sum() for z in hg])-1e-8));multi+=partial>=2
   assert partial==record.partial_tiers
   if t in [0,41,42,84,85,127]:
    v=(1-a)*prev+a*dt;grad=xx-v
    sol=linprog(grad,A_ub=A,b_ub=ct,A_eq=E,b_eq=b,bounds=list(zip(np.zeros(8),dt)),method='highs')
    assert sol.success
    val=float(grad@(sol.x-xx));minvi=min(minvi,val);checks+=1
   prev=xx
  row=ht[(ht.seed==seed)&(ht.alpha==a)].iloc[0]
  assert abs(multi/128-row.multiple_partial_fraction)<1e-12
  assert abs(maxres-row.qp_residual_max)<1e-9
assert rankerr<1e-8 and minvi>-1e-7,(rankerr,minvi)
report['hierarchy']={'rank_checks':2560,'rank_difference_max':rankerr,'variational_inequality_lp_checks':checks,'minimum_projection_VI':minvi,'means':ht.groupby('alpha')[['movement','unweighted_shortfall','multiple_partial_fraction','qp_residual_max','contraction_excess_max']].mean().reset_index().to_dict('records')}
print('hierarchy validated',flush=True)

# Recalculate costs and quadratic certificate with dense inverse system.
of=np.load(O/'offline_trajectories.npz');ot=pd.read_csv(O/'offline.csv');maxofflineprimal=0.;independent=[]
for b in range(10):
 d,c=block(b);d=d[80:176];c=c[80:176];T=len(d);mu=3.;x0=initial(d,c)
 H=np.diag(np.r_[np.full(T-1,1+2*mu),1+mu])+np.diag(np.full(T-1,-mu),1)+np.diag(np.full(T-1,-mu),-1)
 cost=lambda x:.5*np.sum((x-d)**2)+mu/2*np.sum(np.diff(np.vstack([x0,x]),axis=0)**2)
 offline=of[f'b{b}_offline'];rows=ot[ot.block==b]
 assert abs(cost(offline)-rows.iloc[0].offline_upper)<1e-8
 B=np.array([budgets(dt,ct) for dt,ct in zip(d,c)])
 for a in [.25,1.]:
  x=of[f'b{b}_a{a}'];row=rows[rows.alpha==a].iloc[0];diff=np.diff(np.vstack([x0,x]),axis=0)
  h=(mu-(1-a)/a)*diff;h[:-1]-=mu*diff[1:]
  tight=.5*np.sum(h*np.linalg.solve(H,h));loose=.5*np.sum(h*h)
  assert abs(tight-row.certificate_tight)<1e-9 and abs(loose-row.certificate_simple)<1e-9
  assert abs(cost(x)-row.cost)<1e-8
  assert row.gap_upper<=tight+1e-6
 for x in [offline,of[f'b{b}_a0.25'],of[f'b{b}_a1.0']]:
  res=max(0,np.max(-x),np.max(x-d),np.max(x.sum(axis=1)-c),np.max(np.abs(np.column_stack([x[:,z].sum(axis=1) for z in g])-B)))
  maxofflineprimal=max(maxofflineprimal,float(res));assert res<1e-8
 if b in [0,9]:
  Q=sparse.kron(sparse.csc_matrix(H),sparse.eye(n,format='csc'),format='csc');q=-d.copy();q[0]-=mu*x0
  tier=np.zeros((len(g),n))
  for j,z in enumerate(g):tier[j,z]=1
  Eq=sparse.kron(sparse.eye(T,format='csc'),sparse.csc_matrix(tier),format='csc')
  cons=sparse.vstack([sparse.eye(T*n,format='csc'),Eq],format='csc')
  solver=osqp.OSQP();solver.setup(P=Q,q=q.ravel(),A=cons,l=np.r_[np.zeros(T*n),B.ravel()],u=np.r_[d.ravel(),B.ravel()],eps_abs=1e-10,eps_rel=1e-10,max_iter=100000,polishing=True,verbose=False)
  sol=solver.solve(raise_error=False);assert sol.info.status_val==1
  xv=sol.x.reshape(T,n);value=cost(xv);row=rows.iloc[0]
  assert row.offline_lower-1e-6<=value<=row.offline_upper+1e-6
  independent.append({'block':b,'objective':value,'saved_lower':row.offline_lower,'saved_upper':row.offline_upper,'coordinate_difference_max':float(np.max(np.abs(xv-offline))),'primal_residual':sol.info.prim_res,'dual_residual':sol.info.dual_res})
report['offline']={'cost_and_certificate_rows':len(ot),'raw_trajectory_primal_residual_max':maxofflineprimal,'independent_sparse_qp':independent,'means':ot.groupby('alpha')[['cost','offline_upper','gap_percent_lower','gap_percent_upper','certificate_tight','certificate_simple']].mean().reset_index().to_dict('records'),'max_relative_solver_certificate':ot.offline_relative_certificate.max()}
print('offline validated',flush=True)

# Static recovery: independent emergency+projection replay and first-return times.
sr=pd.read_csv(O/'static_recovery.csv');se=pd.read_csv(O/'static_recovery_epochs.csv');d=np.ones(32);x0=wp(d,d,24.,np.ones(32));target=budgets(d,12.)
for rr in sr.to_dict('records'):
 r=rr['ramp'];a=rr['alpha'];prev=x0;firsth=None;firstf=None
 ee=se[(se.ramp==r)&(se.alpha==a)].sort_values('update')
 for j in range(128):
  L=np.maximum(0,prev-r);U=np.minimum(d,prev+r);cap=max(12.,L.sum())
  x=L+wp((1-a)*prev+a*d-L,U-L,max(0,cap-L.sum()),np.ones(32))
  hard=x.sum()<=12.+1e-8;gap=max(abs(x[z].sum()-b) for z,b in zip(g,target))
  if hard and firsth is None:firsth=j+1
  if hard and gap<=1e-8 and firstf is None:firstf=j+1
  assert abs(x.sum()-ee.iloc[j].total)<1e-9;prev=x
 assert firsth==rr['hard_recovery_updates'] and firstf==rr['priority_recovery_updates']
report['static_recovery']={'independent_replayed_sequences':6,'rows':sr.to_dict('records')}
(O/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
