"""Fail on incomplete scenarios, broken numerical claims, or provenance drift."""
from pathlib import Path
import hashlib
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import numpy as np
import pandas as pd
from run_experiments import trajectory, metrics


def verify():
    checks=[]
    def check(name,condition):
        if not condition: raise AssertionError(name)
        checks.append(name)
    R=ROOT/"results"
    df=pd.read_csv(R/"replay.csv");sy=pd.read_csv(R/"synthetic.csv")
    check("complete replay grid",len(df)==6480 and not df.duplicated(["dataset","block","ratio","delay","method"]).any())
    check("complete synthetic grid",len(sy)==2240 and not sy.duplicated(["seed","cycles","delay","method"]).any())
    for name,table in [("replay",df),("synthetic",sy)]:
        check(name+" finite numeric output",np.isfinite(table.select_dtypes(include="number")).all().all())
        proj=table[~table.method.isin(["no_projection"])]
        check(name+" projected hard feasibility",proj.capacity_violation.max()<1e-9 and proj.box_violation.max()<1e-9)
        priority=proj[proj.method!="uniform"]
        # Hard-only correction deliberately does not enforce priority or full utilization.
        priority=priority[priority.method!="delayed_hard"]
        check(name+" priority face",priority.prefix_regret.max()<1e-9)
        check(name+" exact loss lower bound",priority.quadratic_gap.min()>-1e-9)
    bike=df[(df.dataset=="bike")&(~df.method.isin(["no_projection","uniform","delayed_hard"]))]
    check("singleton control invariance",bike.tracking_rmse.max()<1e-10)
    check("exact optimizer zero error",df[df.method=="exact"].tracking_rmse.max()<1e-12)
    blocks=pd.read_csv(R/"blocks.csv")
    check("54 selected blocks",len(blocks)==54)
    for name,b in blocks.groupby("dataset"):
        check(name+" blocks disjoint",np.all(np.diff(b.sort_values("start_row").start_row)>=336))
    cert=pd.read_csv(R/"certificates.csv");rec=pd.read_csv(R/"recovery.csv")
    check("2000 bounded transitions",len(cert)==2000 and (cert.error<=cert.bound+1e-9).all())
    check("local contraction",cert.contraction_excess.max()<1e-9)
    check("static recovery bounds",(rec.error<=rec.bound+1e-9).all())
    qp=pd.read_csv(R/"qp_checks.csv")
    check("180 independent QP solves",len(qp)==180 and qp.success.all() and qp.max_error.max()<2e-5)
    robust=pd.read_csv(R/"robustness.csv")
    c=robust[robust.method=="conservative"]
    check("bounded-error robust feasibility",len(robust)==60 and c.max_capacity_violation.max()<1e-9 and c.max_box_violation.max()<1e-9)
    check("ramp suite complete",len(pd.read_csv(R/"ramp_diagnostics.csv"))==30*4*255)
    # Re-execute a fixed held-out output slice to detect stale summaries.
    data=np.load(ROOT/"data/processed/electricity.npz")
    d=data["demand"][:336];cap=np.full(336,.9*32);cap[112:224]*=.5
    g=[data["critical"],np.setdiff1d(np.arange(32),data["critical"])]
    for method in ["feedback","exact","delayed_hard","delayed_priority","filtered_exact"]:
        x,target=trajectory(d,cap,g,method,4)
        measured=metrics(x,d,cap,g,target,32)
        row=df[(df.dataset=="electricity")&(df.block==0)&(df.ratio==.9)&(df.delay==4)&(df.method==method)].iloc[0]
        check("replay reproducibility "+method,all(abs(row[k]-v)<1e-10 for k,v in measured.items()))
    manifest=json.loads((ROOT/"data/source_manifest.json").read_text())
    for name,meta in manifest.items():
        h=hashlib.sha256()
        with (ROOT/"data/raw"/(name+".zip")).open("rb") as f:
            for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
        check("raw archive checksum "+name,h.hexdigest()==meta["sha256"])
    options=[p for p in [R/"environment_replay.json",R/"environment_all.json"] if p.exists()]
    env=json.loads(max(options,key=lambda p:p.stat().st_mtime).read_text())
    for path in ["src/abo/allocation.py","scripts/run_experiments.py","scripts/download_data.py"]:
        check("experimental source unchanged "+path,hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==env["source_sha256"][path])
    report={"passed":len(checks),"failed":0,"checks":checks}
    (R/"verification.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=="__main__":verify()
