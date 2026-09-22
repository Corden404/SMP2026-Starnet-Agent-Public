"""Bounded sequential block queue; no new seeds, policy edits or result retries."""
from __future__ import annotations
import argparse,json,subprocess,sys,time
from pathlib import Path
import round26_continue as c

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pool',choices=['confirm','stress'],required=True);ap.add_argument('--first',type=int,required=True);ap.add_argument('--stop',type=int,required=True);a=ap.parse_args()
    protocol=c.check();bounds=protocol['blocks'][a.pool]
    if not 0<=a.first<a.stop<=len(bounds):raise ValueError('Invalid fixed block interval')
    step='S2' if a.pool=='confirm' else 'S3';start=time.monotonic()
    for i in range(a.first,a.stop):
        done=c.OUT/'blocks'/('%s-%02d.json'%(a.pool,i))
        if done.exists():
            record=c.s.read(done)
            if record.get('status')!='PASS' or record.get('code')!=protocol['study_code']:raise RuntimeError('Stored block identity mismatch')
            print(json.dumps(dict(reused=True,pool=a.pool,block=i)),flush=True);continue
        # Do not start a block too near the outer 3600s tool boundary.
        if time.monotonic()-start>1600:
            c.mark(step,'RUNNING','当前有界队列在区块%d前正常暂停；已完成PASS全部保留，下一队列从此区块续跑。'%i)
            print(json.dumps(dict(status='BOUNDED_CHECKPOINT',next_block=i,pool=a.pool)),flush=True);return
        args=[sys.executable,'-X','utf8','-B','workspace/round26_continue.py','block','--pool',a.pool,'--index',str(i)]
        log=c.OUT/'blocks'/('%s-%02d-command.log'%(a.pool,i));log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('x',encoding='utf8') as f:
            try:
                result=subprocess.run(args,cwd=c.s.ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=1800,check=False)
            except subprocess.TimeoutExpired:
                c.mark(step,'BLOCKED','区块%d子进程超时；先核对工作进程和STARTED，不自动重试。'%i,log.relative_to(c.s.ROOT).as_posix());raise
        if result.returncode!=0:
            c.mark(step,'BLOCKED','区块%d失败exit=%d；检查保存日志，禁止重跑成功局。'%(i,result.returncode),log.relative_to(c.s.ROOT).as_posix());raise SystemExit(result.returncode)
        if not done.exists() or c.s.read(done).get('status')!='PASS':raise RuntimeError('Successful command did not finalize block')
        print(json.dumps(dict(status='BLOCK_COMPLETE',pool=a.pool,index=i)),flush=True)
    print(json.dumps(dict(status='QUEUE_COMPLETE',pool=a.pool,first=a.first,stop=a.stop)),flush=True)
if __name__=='__main__':main()
