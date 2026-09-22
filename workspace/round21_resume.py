"""Round21 status / recovery. No API replay."""
from __future__ import annotations
import argparse,collections,json,shutil,uuid
from pathlib import Path
import round21_state as s
from round18_resume import pid_alive
from round9_lock import ExclusiveBatch

def inventory():
    groups={};incomplete=[]
    for pool in ('dev','integration','confirm','stress'):
        folder=s.OUT/'study/runs'/pool;counts=collections.Counter()
        for p in sorted(folder.glob('*.json')):
            try:r=s.read(p);st=r.get('status','UNKNOWN')
            except (ValueError,OSError):r={};st='MALFORMED'
            counts[st]+=1
            if st!='PASS':incomplete.append(dict(path=p.relative_to(s.OUT).as_posix(),status=st,pid=r.get('pid'),pid_alive=pid_alive(r.get('pid'))))
        groups[pool]=dict(counts)
    return dict(groups=groups,incomplete=incomplete)

def status():
    r=dict(progress=s.read(s.OUT/'progress.json'),inventory=inventory(),originals=s.preserved())
    print(json.dumps(r,ensure_ascii=False,indent=2));return r

def recover(relative,reason):
    if not reason.strip():raise ValueError('Concrete reason required')
    with ExclusiveBatch(s.OUT/'execution.lock'):
        p=(s.OUT/relative).resolve();root=s.OUT.resolve()
        if root not in p.parents or p.suffix!='.json':raise ValueError('Not a Round21 JSON result')
        parts=p.relative_to(root).parts
        if len(parts)<4 or parts[:2]!=('study','runs') or parts[2] not in ('dev','integration','confirm','stress'):
            raise ValueError('Only local unfinished study records may be recovered')
        row=s.read(p)
        if row.get('status') not in ('STARTED','FAILED'):raise RuntimeError('Never overwrite PASS')
        if pid_alive(row.get('pid')):raise RuntimeError('Recorded process still alive')
        archive=s.OUT/'recovery'/('local-'+uuid.uuid4().hex);archive.mkdir(parents=True,exist_ok=False)
        rec=dict(time=s.now(),reason=reason,path=p.relative_to(s.OUT).as_posix(),sha256=s.sha(p),no_api=True)
        s.write(archive/'recovery.json',rec,True);shutil.move(str(p),str(archive/p.name))
        s.update(detail='已归档Round21未完成本地记录：'+relative,overall='READY_TO_RESUME',
                 next_action='按冻结协议恢复相同固定块；PASS不重跑。',evidence=rec)
        print(json.dumps(rec,ensure_ascii=False))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['status','recover-local']);ap.add_argument('--path');ap.add_argument('--reason');a=ap.parse_args()
    if a.stage=='status':status()
    else:
        if not a.path or not a.reason:ap.error('--path/--reason required')
        recover(a.path,a.reason)
