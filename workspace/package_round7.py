"""Create a standalone submission only after all prospective local gates pass."""
import json
import shutil
import zipfile
from experiment_round7 import ROOT, OUT, source_hashes
from round6_common import D5_SOURCE, D5_ARCHIVE, file_hash, save

def build():
    decision=json.loads((OUT/'selection.json').read_text(encoding='utf-8'))
    if decision['status']!='LOCAL_CONFIRMED':raise RuntimeError('Candidate has not passed all local gates')
    variant=decision['candidate']
    frozen=json.loads((OUT/'candidate-freezes'/(variant+'.json')).read_text(encoding='utf-8'))
    if frozen['source_hashes']!=source_hashes(variant):raise RuntimeError('Candidate source drift')
    source={'U3':'round6_feedback.py','X1':'round7_exchange.py','S48':'round7_scan.py','S49':'round7_scan.py'}[variant]
    extension=(ROOT/'workspace'/source).read_text(encoding='utf-8')
    extension='\n'.join(line for line in extension.splitlines() if not (
        line.startswith('from __future__ import') or line.startswith('from round6_common import') or
        line.replace(' ','')=='D5=load_d5_module()'))
    bridge='\nimport types\nD5 = types.SimpleNamespace(ParticipantSquadModel=ParticipantSquadModel, FiniteRollout=FiniteRollout, score=score, SEARCH=SEARCH, L0=L0)\n'
    tail='\n\nParticipantSquadModel = '+variant+'\n'
    if variant in ('S48','S49'):
        tail='\n\nclass ParticipantSquadModel('+variant+'):\n    def __init__(self, host_env, person_list, llm):\n        super().__init__(host_env, person_list, llm)\n        if self.node_count != 50:\n            self.scan_target = self.node_count\n'
    model=D5_SOURCE.read_text(encoding='utf-8')+bridge+extension+tail
    assert 'from round6_common import' not in model and 'load_d5_module()' not in model
    destination=OUT/('submission-'+variant)/'team_submission'
    destination.mkdir(parents=True,exist_ok=True)
    target=destination/'starnet_model.py'
    if target.exists() and target.read_text(encoding='utf-8')!=model:raise RuntimeError('Existing package source differs')
    target.write_text(model,encoding='utf-8')
    shutil.copy2(D5_SOURCE.with_name('config.json'),destination/'config.json')
    shutil.copytree(D5_SOURCE.with_name('prompt'),destination/'prompt',dirs_exist_ok=True)
    files={p.relative_to(destination).as_posix():file_hash(p) for p in sorted(destination.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
    archive=ROOT/'dist'/('team_submission-round7-'+variant.lower()+'-local-confirmed-20260918.zip')
    if archive.exists():raise RuntimeError('Refuse to overwrite existing submission archive')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for rel in files:z.write(destination/rel,rel)
    manifest=dict(variant=variant,archive=str(archive),sha256=file_hash(archive),
        baseline_sha256=file_hash(D5_ARCHIVE),submission_files=files,candidate_freeze=frozen,
        status='LOCAL_CONFIRMED_PENDING_PACKAGE_AND_SERVICE_CHECKS',source=str(target),
        validation_study=decision.get('study','original'),validation_record=str(OUT/'selection.json'),
        scope='Frozen candidate on 50 nodes; scan variants retain D5 full-scan behavior outside that validated size')
    save(destination.parent/'manifest.json',manifest)
    print(json.dumps(manifest,ensure_ascii=False,indent=2))
    return manifest

if __name__=='__main__':build()
