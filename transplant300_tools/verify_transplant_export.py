#!/usr/bin/env python3
"""Validate the compressed export and every included file without extraction."""
import hashlib, json, sys, tarfile
from pathlib import Path

root=Path(sys.argv[1]); archive=root/'transplant300_final.tar.gz'
expected_archive=(root/'archive.sha256').read_text().split()[0]
def digest_stream(f):
    h=hashlib.sha256()
    for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()
with archive.open('rb') as f:
    actual=digest_stream(f)
if actual!=expected_archive: raise ValueError('Archive checksum mismatch')
manifest=json.loads((root/'EXPORT_MANIFEST.json').read_text())
expected=manifest['files']; seen=set(); params=0; predicted=0
with tarfile.open(archive,'r|gz') as tf:
    for member in tf:
        name=member.name
        if not member.isfile() or Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError(f'Unexpected archive member: {name}')
        if name=='EXPORT_MANIFEST.json':
            if json.load(tf.extractfile(member))!=manifest: raise ValueError('Manifest differs')
            continue
        if name in seen or name not in expected: raise ValueError(f'Unexpected/duplicate: {name}')
        if 'sha256:'+digest_stream(tf.extractfile(member))!=expected[name]: raise ValueError(f'File checksum mismatch: {name}')
        seen.add(name)
        params+=name.startswith('training300/') and name.endswith('/params.npz')
        predicted+=name.startswith('predictions300/') and name.endswith('/prediction.json')
if seen!=set(expected) or params!=5120 or predicted!=manifest['predicted']: raise ValueError('Incomplete export')
result=dict(archive_sha256=actual,verified_files=len(seen),final_params=params,predictions=predicted,
            missing_predictions=5120-predicted,integrity_verified=True,release_certified=False)
(root/'LOCAL_VERIFICATION.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
