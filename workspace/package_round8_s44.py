"""Package S44 from the verified original ZIP after independent S44 study gates."""
import json
import zipfile
from round8_common import ROOT,OUT,ARCHIVE,ARCHIVE_SHA,file_hash,save
from experiment_round8_s44 import fingerprint

def build():
    study=OUT/'s44-study'
    freeze=json.loads((study/'candidate-freeze.json').read_text(encoding='utf-8'))
    if freeze['code']!=fingerprint('S44'):raise RuntimeError('S44 source drift')
    confirm=json.loads((study/'confirm-fast-summary.json').read_text(encoding='utf-8'))['S44']
    stress=json.loads((study/'stress-fast-summary.json').read_text(encoding='utf-8'))['S44']
    if not(confirm['n']==384 and confirm['ci95'][0]>0 and confirm['without_best_two']>0):raise RuntimeError('Confirmation failed')
    if not(stress['n']==96 and stress['mean']>=0):raise RuntimeError('Stress failed')
    if file_hash(ARCHIVE)!=ARCHIVE_SHA:raise RuntimeError('Online baseline changed')
    folder=study/'submission-S44'/'team_submission';folder.mkdir(parents=True,exist_ok=True);files={}
    with zipfile.ZipFile(ARCHIVE) as z:
        for name in z.namelist():
            if name.endswith('/'):continue
            data=z.read(name)
            if name=='starnet_model.py':
                source=data.decode('utf-8')+'\n\nclass ParticipantSquadModel(ParticipantSquadModel):\n    """Validated late scan stop for the 50-node competition only."""\n    scan_target = 44\n'
                compile(source,name,'exec');data=source.encode('utf-8')
            path=folder/name;path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists() and path.read_bytes()!=data:raise RuntimeError('Existing S44 package differs')
            path.write_bytes(data);files[name]=file_hash(path)
    archive=ROOT/'dist/team_submission-round8-s44-local-confirmed-20260919.zip'
    if archive.exists():raise RuntimeError('Refuse to overwrite ZIP')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name in sorted(files):z.write(folder/name,name)
    manifest=dict(variant='S44',archive=str(archive),sha256=file_hash(archive),baseline_sha256=ARCHIVE_SHA,
        submission_files=files,freeze=freeze,status='LOCAL_CONFIRMED_PENDING_PACKAGE_AND_SERVICE_ACCEPTANCE',
        change='50-node scan_target 48 -> 44; full scan retained for other sizes',
        confirm={k:v for k,v in confirm.items() if k!='deltas'},stress={k:v for k,v in stress.items() if k!='deltas'})
    save(folder.parent/'manifest.json',manifest)
    print(json.dumps(dict(archive=str(archive),sha256=manifest['sha256']),ensure_ascii=True))

if __name__=='__main__':build()
