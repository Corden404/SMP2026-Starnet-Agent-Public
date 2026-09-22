"""Round19 status and explicitly recorded local-only recovery. No API replay or retries."""
from __future__ import annotations
import argparse
import collections
import json
from pathlib import Path
import shutil
import uuid
import round19_state as s
from round18_resume import pid_alive
from round9_lock import ExclusiveBatch


def inventory():
    groups={};incomplete=[]
    folders=['diagnostic/states','diagnostic/branches']+['study/runs/'+x for x in ('dev','integration','confirm','stress')]+['real/runs']
    for folder in folders:
        counts=collections.Counter()
        for p in sorted((s.OUT/folder).glob('*.json')):
            try:row=s.read(p);status=row.get('status','UNKNOWN')
            except (ValueError,OSError):row={};status='MALFORMED'
            counts[status]+=1
            if status!='PASS':incomplete.append(dict(path=p.relative_to(s.OUT).as_posix(),status=status,
                pid=row.get('pid'),pid_alive=pid_alive(row.get('pid'))))
        groups[folder]=dict(counts)
    return dict(groups=groups,incomplete=incomplete)


def status():
    record=dict(progress=s.read(s.OUT/'progress.json'),inventory=inventory(),originals=s.preserved())
    print(json.dumps(record,ensure_ascii=False,indent=2));return record


def target(relative):
    p=(s.OUT/relative).resolve();root=s.OUT.resolve()
    if root not in p.parents or p.suffix!='.json':raise ValueError('Not a round19 local JSON result')
    parts=p.relative_to(root).parts
    if not (parts[:2] in (('diagnostic','states'),('diagnostic','branches')) or
            len(parts)>=4 and parts[:2]==('study','runs') and parts[2] in ('dev','integration','confirm','stress')):
        raise ValueError('Only local unfinished strategy/diagnostic records may be recovered')
    return p


def recover(relative,reason):
    if not reason.strip():raise ValueError('Concrete recovery reason required')
    with ExclusiveBatch(s.OUT/'execution.lock'):
        p=target(relative);row=s.read(p)
        if row.get('status') not in ('STARTED','FAILED'):raise RuntimeError('Never overwrite a successful result')
        if pid_alive(row.get('pid')) or any(r['pid_alive'] for r in inventory()['incomplete']):
            raise RuntimeError('Recorded process still running; no duplicate execution')
        if row.get('http_calls') or row.get('identity',{}).get('model_config'):raise RuntimeError('Paid results require separate exact replay')
        if p.relative_to(s.OUT.resolve()).parts[0]=='diagnostic':
            import round19_diagnose as module
        else:
            import round19_study as module
        if row['identity']['code']!=module.check()['code']:raise RuntimeError('Frozen source changed; cannot resume mixed versions')
        archive=s.OUT/'recovery'/('local-'+uuid.uuid4().hex);archive.mkdir(parents=True,exist_ok=False)
        files=[p]
        if p.parent.name=='states':files+=list((s.OUT/'diagnostic/shadows').glob(p.stem+'-*.json'))
        record=dict(reason=reason,time=s.now(),files=[dict(path=x.relative_to(s.OUT.resolve()).as_posix(),sha256=s.sha(x)) for x in files],
            policy='Only unfinished state envelope and its numerical shadow records archived. Completed real-world branches and complete states reused. No new independent sample; no paid calls.')
        s.write(archive/'recovery.json',record,True)
        for f in files:shutil.move(str(f),str(archive/f.name))
        s.update(detail='本地未完成尝试已归档：'+relative,overall='READY_TO_RESUME',
            next_action='按原固定块命令恢复；成功分支/整局按身份复用。',evidence=record)
        print(json.dumps(record,ensure_ascii=False));return record


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['status','recover-local'])
    ap.add_argument('--path');ap.add_argument('--reason');a=ap.parse_args()
    if a.stage=='status':status()
    else:
        if not a.path or not a.reason:ap.error('--path and --reason required')
        recover(a.path,a.reason)
