"""Build and audit the explicitly allowlisted DOI-deposit package; never upload."""
from pathlib import Path
import hashlib,json,zipfile
from build_code_archive import ALLOW as CODE, FORBIDDEN
ROOT=Path(__file__).resolve().parents[1]
# This is an explicit list, deliberately not a recursive directory or glob.
EXTRA=[
 'data/source_manifest.json','data/processing_manifest.json',
 'results/blocks.csv','results/certificates.csv','results/environment_all.json',
 'results/example_trajectories.npz','results/paired_intervals.csv','results/primary_summary.csv',
 'results/qp_checks.csv','results/ramp_diagnostics.csv','results/recovery.csv','results/replay.csv',
 'results/robustness.csv','results/runtime.csv','results/summary_all.csv','results/synthetic.csv',
 'results/analysis_provenance.json','results/validation_summary.json','results/verification.json',
 'results/review_data_certificate.csv','results/review_fairness.csv',
 'results/review_fairness_blocks.csv','results/review_provenance.json',
 'results/review_relative_effects.csv','results/review_robustness_seeds.csv',
 'results/review_robustness_tiers.csv','results/review_summary.json','results/review_timing_batches.csv',
]

def build():
    names=CODE+EXTRA;assert len(names)==len(set(names))
    hashes={}
    for name in names:
        p=ROOT/name
        if not p.is_file() or p.is_symlink() or p.suffix.lower() in FORBIDDEN:
            raise ValueError(name)
        if p.suffix!='.npz':
            raw=p.read_bytes()
            for marker in [bytes([92])+b'documentclass',bytes([92])+b'begin{document}',b'@'+b'article{']:
                assert marker not in raw,name
        hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest()
    # Explicitly validate the only permitted binary file's scientific array keys.
    import numpy as np
    with np.load(ROOT/'results/example_trajectories.npz',allow_pickle=False) as a:
        allowed={dataset+'_'+field for dataset in ['bike','electricity'] for field in [
            'feedback','exact','delayed_hard','delayed_priority','filtered_exact','uniform',
            'no_projection','alpha_010','alpha_050','alpha_075','demand','capacity']}
        assert set(a.files)==allowed
        assert all(np.issubdtype(a[k].dtype,np.number) and np.isfinite(a[k]).all() for k in a.files)
    out=ROOT/'dist';out.mkdir(exist_ok=True);dest=out/'abo-reproducibility-v0.2.0.zip'
    inventory={'package_version':'0.2.0','files_sha256':hashes,
               'excluded':'manuscript, bibliography, templates, editorial documents, PDF, original dataset archives, Git history'}
    with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name in names:z.write(ROOT/name,name)
        z.writestr('MANIFEST.json',json.dumps(inventory,indent=2))
    with zipfile.ZipFile(dest) as z:
        assert z.namelist()==names+['MANIFEST.json'] and z.testzip() is None
        for name,digest in hashes.items():assert hashlib.sha256(z.read(name)).hexdigest()==digest
    audit={'archive':dest.name,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),
           'bytes':dest.stat().st_size,'entries':len(names)+1,'files':names+['MANIFEST.json'],
           'manuscript_sources_included':False,'uploaded':False,'doi':None}
    (out/'reproducibility-archive-audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2))
if __name__=='__main__':build()
