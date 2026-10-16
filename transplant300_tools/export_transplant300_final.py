#!/usr/bin/env python3
"""Create an immutable final-parameter export, not an optimizer-resume archive."""
import argparse, hashlib, json, tarfile
from pathlib import Path

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return 'sha256:'+h.hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--training',type=Path,required=True)
    ap.add_argument('--predictions',type=Path,required=True); ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args(); out=a.output.resolve()
    for root in (a.training.resolve(),a.predictions.resolve()):
        if out==root or root in out.parents or out in root.parents: raise ValueError('Output must be disjoint')
    out.mkdir(exist_ok=False)
    selected={}; completed=sorted(a.training.glob('svgp/transplant/runs/*/*/M*/*/seed*/COMPLETE.json'))
    if len(completed)!=5120: raise ValueError(f'Expected 5120 completed checkpoints, got {len(completed)}')
    for cp in completed:
        rec=json.loads(cp.read_text())
        for p in sorted(cp.parent.iterdir()):
            if not p.is_file() or p.name.startswith('checkpoint_epoch'): continue
            if p.name in rec['files'] and sha(p)!=rec['files'][p.name]: raise ValueError(f'Hash mismatch {p}')
            if p.name=='params.npz' and p.name not in rec['files']: raise ValueError('Unbound parameters')
            selected[p]=Path('training300')/p.relative_to(a.training)
    for name in ('continuation_plan.json','EXPECTED_HEAD','jobs.env'):
        selected[a.training/name]=Path('training300')/name
    for p in sorted(a.predictions.rglob('*')):
        if p.is_file(): selected[p]=Path('predictions300')/p.relative_to(a.predictions)
    manifest=dict(training_root=str(a.training),prediction_root=str(a.predictions),trained=5120,
                  predicted=len(list(a.predictions.rglob('prediction.json'))),
                  scope='Final parameters, training metadata/curves and available predictions. Intermediate optimizer checkpoints are retained on cluster, excluded here. Not a full release certification.',
                  files={str(dst):sha(src) for src,dst in selected.items()})
    mp=out/'EXPORT_MANIFEST.json'; mp.write_text(json.dumps(manifest,indent=2))
    archive=out/'transplant300_final.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=1) as tf:
        for src,dst in selected.items(): tf.add(src,arcname=str(dst),recursive=False)
        tf.add(mp,arcname=mp.name)
    digest=sha(archive)
    (out/'archive.sha256').write_text(digest.split(':')[1]+'  '+archive.name+'\n')
    print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,sha256=digest,trained=5120,predicted=manifest['predicted'])),flush=True)

if __name__=='__main__': main()
