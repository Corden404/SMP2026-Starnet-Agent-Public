"""Build a standalone Q48 archive only after independent local gates pass."""
import inspect
import json
import zipfile
from round8_common import ROOT, OUT, STAGE, ARCHIVE, ARCHIVE_SHA, file_hash, save
from round8_scan import Q48
from experiment_round8 import fingerprint

def build():
    freeze=json.loads((OUT/'candidate-freeze.json').read_text(encoding='utf-8'))
    if freeze['variant']!='Q48' or freeze['code']!=fingerprint('Q48'):raise RuntimeError('Candidate drift')
    confirm=json.loads((OUT/'confirm-fast-summary.json').read_text(encoding='utf-8'))['Q48']
    stress=json.loads((OUT/'stress-fast-summary.json').read_text(encoding='utf-8'))['Q48']
    if not(confirm['n']==384 and confirm['ci95'][0]>0 and confirm['without_best_two']>0):
        raise RuntimeError('Independent mean gate not passed')
    if not(stress['n']==96 and stress['mean']>=0):raise RuntimeError('Stress mean gate not passed')
    if file_hash(ARCHIVE)!=ARCHIVE_SHA:raise RuntimeError('Online baseline changed')
    source=(STAGE/'starnet_model.py').read_text(encoding='utf-8')
    source+='\n\nimport types\nS48 = types.SimpleNamespace(ParticipantSquadModel=ParticipantSquadModel)\n\n'
    source+=inspect.getsource(Q48)
    source+='\n\nclass ParticipantSquadModel(Q48):\n    pass\n'
    compile(source,'starnet_model.py','exec')
    folder=OUT/'submission-Q48'/'team_submission';folder.mkdir(parents=True,exist_ok=True)
    files={}
    with zipfile.ZipFile(ARCHIVE) as base:
        for name in base.namelist():
            if name.endswith('/'):continue
            data=source.encode('utf-8') if name=='starnet_model.py' else base.read(name)
            path=folder/name;path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists() and path.read_bytes()!=data:raise RuntimeError('Existing stage differs')
            path.write_bytes(data);files[name]=file_hash(path)
    target=ROOT/'dist/team_submission-round8-q48-local-confirmed-20260919.zip'
    if target.exists():raise RuntimeError('Refuse to overwrite archive')
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        for name in sorted(files):z.write(folder/name,name)
    manifest=dict(variant='Q48',archive=str(target),sha256=file_hash(target),baseline_sha256=ARCHIVE_SHA,
        submission_files=files,freeze=freeze,status='LOCAL_CONFIRMED_PENDING_PACKAGE_AND_SERVICE_ACCEPTANCE',
        confirm={k:v for k,v in confirm.items() if k!='deltas'},stress={k:v for k,v in stress.items() if k!='deltas'})
    save(folder.parent/'manifest.json',manifest)
    print(json.dumps(dict(archive=str(target),sha256=manifest['sha256']),ensure_ascii=True))

if __name__=='__main__':build()
