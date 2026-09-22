"""Inspect/reconcile round18 progress. Explicit local recovery never touches paid runs."""
from __future__ import annotations
import argparse
import collections
import ctypes
import json
import os
from pathlib import Path
import shutil
import uuid
import round18_state as s
from round9_lock import ExclusiveBatch


def pid_alive(pid):
    if not isinstance(pid,int) or pid<=0:return False
    if os.name!='nt':
        try:os.kill(pid,0)
        except ProcessLookupError:return False
        except PermissionError:return True
        return True
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong]
    kernel.OpenProcess.restype=ctypes.c_void_p
    kernel.GetExitCodeProcess.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_ulong)]
    kernel.GetExitCodeProcess.restype=ctypes.c_int
    kernel.CloseHandle.argtypes=[ctypes.c_void_p]
    handle=kernel.OpenProcess(0x1000,False,pid)
    if not handle:return ctypes.get_last_error()!=87
    try:
        code=ctypes.c_ulong()
        if not kernel.GetExitCodeProcess(handle,ctypes.byref(code)):return True
        return code.value==259
    finally:kernel.CloseHandle(handle)


def inventory():
    groups={};incomplete=[]
    folders=['diagnostic/baseline','diagnostic/branches','diagnostic/graphs']
    folders += ['study/runs/'+p for p in ('dev','integration','confirm','stress')]
    folders += ['real/runs']
    for folder in folders:
        counts=collections.Counter()
        for path in sorted((s.OUT/folder).glob('*.json')):
            try:
                row=s.read(path);status=row.get('status','UNKNOWN')
            except (ValueError,OSError):
                row={};status='MALFORMED'
            counts[status]+=1
            if status!='PASS':incomplete.append(dict(path=path.relative_to(s.OUT).as_posix(),status=status,
                pid=row.get('pid'),pid_alive=pid_alive(row.get('pid')),started_at=row.get('started_at')))
        groups[folder]=dict(counts)
    return dict(groups=groups,incomplete=incomplete)


def status(sync=False):
    result=dict(progress=s.read(s.OUT/'progress.json'),inventory=inventory(),originals=s.preserved())
    if sync:
        with ExclusiveBatch(s.OUT/'execution.lock'):
            active=[r for r in result['inventory']['incomplete'] if r['pid_alive']]
            if active:raise RuntimeError('A recorded worker still runs; not reconciling it as interrupted')
            s.update(detail='已检查真实落盘文件与原文件保护；恢复前不得只看 Markdown 计数。',
                overall='BLOCKED' if result['inventory']['incomplete'] else result['progress']['status'],
                next_action='检查 inventory 中未完成项；完整局可复用，失败项须逐项有理由恢复。',evidence='resume-status.json')
            s.write(s.OUT/'resume-status.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result


def local_target(relative):
    target=(s.OUT/relative).resolve();root=s.OUT.resolve()
    if root not in target.parents or target.suffix!='.json':raise ValueError('Not a round18 JSON result')
    parts=target.relative_to(root).parts
    allowed=(parts[:2] in (('diagnostic','baseline'),('diagnostic','branches')) or
             len(parts)>=4 and parts[:2]==('study','runs') and parts[2] in ('dev','integration','confirm','stress'))
    if not allowed:raise ValueError('Only incomplete LOCAL baseline/branch/study records can be recovered')
    return target


def recover_local(relative,reason):
    if not reason.strip():raise ValueError('A concrete recovery reason is required')
    with ExclusiveBatch(s.OUT/'execution.lock'):
        target=local_target(relative);relative=target.relative_to(s.OUT.resolve()).as_posix();row=s.read(target)
        if row.get('status') not in ('STARTED','FAILED'):raise RuntimeError('Never recover/overwrite a successful record')
        if pid_alive(row.get('pid')):raise RuntimeError('Recorded process still alive; do not duplicate it')
        if any(r['pid_alive'] for r in inventory()['incomplete']):raise RuntimeError('Another recorded worker is alive')
        if row.get('http_calls',0) or row.get('new_http_calls',0) or row.get('identity',{}).get('model_config'):
            raise RuntimeError('Paid/uncertain requests need a separately verified exact replay protocol')
        s.preserved()
        if relative.startswith('diagnostic/'):
            import round18_diagnose as d
            current=d.check()['code']
        else:
            import round18_study as e
            current=e.check()['code']
        if row['identity']['code']!=current:raise RuntimeError('Cannot mix source revisions during recovery')
        token=uuid.uuid4().hex
        archive=s.OUT/'recovery'/token/target.name
        archive.parent.mkdir(parents=True,exist_ok=False)
        record=dict(original=target.relative_to(s.OUT).as_posix(),archive=archive.relative_to(s.OUT).as_posix(),
            original_sha256=s.sha(target),reason=reason,time=s.now(),code_unchanged=True,
            rule='Archive incomplete attempt; same frozen graph/variant only. Completed records untouched. No extra independent sample.')
        s.write(archive.parent/'recovery.json',record,True)
        shutil.move(str(target),str(archive))
        s.update(detail='归档并释放一个已结束的本地未完成尝试：'+relative,overall='READY_TO_RESUME',
            next_action='重执行该原固定块命令；所有已完成局将按身份复用。',evidence=record['archive'])
        print(json.dumps(record,ensure_ascii=False))
        return record


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['status','sync','recover-local'])
    ap.add_argument('--path');ap.add_argument('--reason');a=ap.parse_args()
    if a.stage=='recover-local':
        if not a.path or not a.reason:ap.error('--path and --reason are required')
        recover_local(a.path,a.reason)
    else:status(sync=a.stage=='sync')

if __name__=='__main__':main()
