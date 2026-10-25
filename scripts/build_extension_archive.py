"""Build a local allowlisted v0.3.0 experiment package. Never publish or upload."""
from pathlib import Path
import hashlib
import json
import zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
# Explicitly enumerated public code, documentation, provenance and numerical outputs.
ALLOW = [
    'README.md',
    'pyproject.toml',
    'requirements-lock.txt',
    'DATA_SOURCES.md',
    'LICENSE',
    'LICENSE-DATA.md',
    'CITATION.cff',
    '.zenodo.json',
    '.gitignore',
    'src/abo/__init__.py',
    'src/abo/allocation.py',
    'scripts/download_data.py',
    'scripts/run_experiments.py',
    'scripts/robustness.py',
    'scripts/analyze.py',
    'scripts/verify_outputs.py',
    'tests/test_allocation.py',
    'tests/test_review_claims.py',
    'scripts/review_analysis.py',
    'data/source_manifest.json',
    'data/processing_manifest.json',
    'results/blocks.csv',
    'results/certificates.csv',
    'results/environment_all.json',
    'results/example_trajectories.npz',
    'results/paired_intervals.csv',
    'results/primary_summary.csv',
    'results/qp_checks.csv',
    'results/ramp_diagnostics.csv',
    'results/recovery.csv',
    'results/replay.csv',
    'results/robustness.csv',
    'results/runtime.csv',
    'results/summary_all.csv',
    'results/synthetic.csv',
    'results/analysis_provenance.json',
    'results/validation_summary.json',
    'results/verification.json',
    'results/review_data_certificate.csv',
    'results/review_fairness.csv',
    'results/review_fairness_blocks.csv',
    'results/review_provenance.json',
    'results/review_relative_effects.csv',
    'results/review_robustness_seeds.csv',
    'results/review_robustness_tiers.csv',
    'results/review_summary.json',
    'results/review_timing_batches.csv',
    'EXTENSIONS.md',
    'src/abo/extensions.py',
    'tests/test_extensions.py',
    'scripts/run_extensions.py',
    'scripts/verify_extensions.py',
    'scripts/render_extensions.py',
    'scripts/build_extension_archive.py',
    'results/extensions/protocol.json',
    'results/extensions/environment_all.json',
    'results/extensions/figure_provenance.json',
    'results/extensions/verification.json',
    'results/extensions/hierarchy.csv',
    'results/extensions/hierarchy_epochs.csv',
    'results/extensions/hierarchy_trajectories.npz',
    'results/extensions/offline.csv',
    'results/extensions/offline_trajectories.npz',
    'results/extensions/ramp_epochs.csv',
    'results/extensions/ramp_trajectories.npz',
    'results/extensions/ramps.csv',
    'results/extensions/static_recovery.csv',
    'results/extensions/static_recovery_epochs.csv',
    'results/extensions/weight_epochs.csv',
    'results/extensions/weighted_trajectories.npz',
    'results/extensions/weights.csv',
]
FORBIDDEN={'.tex','.bib','.bst','.sty','.cls','.pdf','.doc','.docx','.pptx','.ppt','.odt','.odp','.svg','.png','.jpg','.jpeg'}

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def expected_keys(name):
    if name=='results/example_trajectories.npz':
        return {d+'_'+f for d in ['bike','electricity'] for f in ['feedback','exact','delayed_hard','delayed_priority','filtered_exact','uniform','no_projection','alpha_010','alpha_050','alpha_075','demand','capacity']}
    if name.endswith('weighted_trajectories.npz'):
        return {f'b{b}_beta{w}_a{a}' for b in range(30) for w in range(3) for a in [.25,1.]}
    if name.endswith('ramp_trajectories.npz'):
        return {f'b{b}_r{r}_a{a}' for b in range(30) for r in [.1,.5,2.] for a in [.25,1.]}
    if name.endswith('hierarchy_trajectories.npz'):
        return {f'seed{s}_{field}' for s in range(10) for field in ['a0.25','a1.0','demand','capacities']}
    if name.endswith('offline_trajectories.npz'):
        return {f'b{b}_{field}' for b in range(10) for field in ['offline','a0.25','a1.0']}
    raise ValueError(f'Unexpected binary file {name}')

def build():
    assert len(ALLOW)==len(set(ALLOW))
    hashes={}
    for name in ALLOW:
        p=ROOT/name
        assert p.is_file() and not p.is_symlink() and p.suffix.lower() not in FORBIDDEN,name
        assert not any(part in ['.git','original','submission','logs'] for part in p.relative_to(ROOT).parts),name
        if p.suffix=='.npz':
            with np.load(p,allow_pickle=False) as a:
                assert set(a.files)==expected_keys(name),name
                assert all(np.issubdtype(a[k].dtype,np.number) and np.isfinite(a[k]).all() for k in a.files),name
        else:
            raw=p.read_bytes();raw.decode('utf-8')
            for marker in [bytes([92])+b'documentclass',bytes([92])+b'begin{document}',b'@'+b'article{']:
                assert marker not in raw,name
        hashes[name]=sha(p)
    meta=json.loads((ROOT/'results/extensions/environment_all.json').read_text())
    for name,digest in meta['source_sha256'].items():assert hashes[name]==digest,name
    for name,digest in meta['output_sha256'].items():assert hashes['results/extensions/'+name]==digest,name
    assert hashes['results/extensions/protocol.json']==meta['protocol_sha256']
    inventory={'package_version':'0.3.0','release_status':'versioned_release_package','doi':None,'files_sha256':hashes,
      'excluded':'manuscript sources, bibliography, templates, editorial documents, paper and figure PDF, editable schematics, original datasets and Git history'}
    out=ROOT/'dist';out.mkdir(exist_ok=True);dest=out/'abo-reproducibility-v0.3.0.zip'
    with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name in ALLOW:z.write(ROOT/name,name)
        z.writestr('MANIFEST.json',json.dumps(inventory,indent=2)+'\n')
    with zipfile.ZipFile(dest) as z:
        assert z.namelist()==ALLOW+['MANIFEST.json'] and z.testzip() is None
        for name,digest in hashes.items():assert hashlib.sha256(z.read(name)).hexdigest()==digest,name
    report={'archive':dest.name,'sha256':sha(dest),'bytes':dest.stat().st_size,'entries':len(ALLOW)+1,
      'files':ALLOW+['MANIFEST.json'],'manuscript_sources_included':False,'uploaded':False,'doi':None}
    (out/'extension-archive-audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='files'},indent=2))

if __name__=='__main__':build()
