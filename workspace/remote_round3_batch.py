"""Bounded independent sandbox workers; terminate ALL owned workers on empty credit."""
import argparse
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cases',nargs='+',required=True)
    p.add_argument('--variants',nargs='+',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--workers',type=int,default=3)
    p.add_argument('--repeat',type=int,default=0)
    a=p.parse_args()
    folder=Path(a.output)
    folder.mkdir(parents=True,exist_ok=True)
    pending=list(a.cases)
    if pending==['holdout']:pending=[f'holdout-{i:03}' for i in range(20)]
    active=[]
    try:
        while pending or active:
            while pending and len(active)<a.workers:
                case=pending.pop(0)
                log=open(folder/(case+f'-worker-r{a.repeat}.log'),'w',encoding='utf-8')
                cmd=[sys.executable,str(ROOT/'workspace/experiment_round3.py'),'--mode','remote','--cases',case,
                     '--no-summary','--variants',*a.variants,'--repeat',str(a.repeat),'--output',str(folder)]
                proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0)
                active.append((case,proc,log))
                print(f'START {case} pid={proc.pid}',flush=True)
            for item in list(active):
                case,proc,log=item
                status=proc.poll()
                if status is None:continue
                log.close();active.remove(item)
                print(f'END {case} exit={status}',flush=True)
                if status:
                    # Includes credit exhaustion (13), network or validation error.
                    # Stop scheduling; finish no further billable work on a fault.
                    raise RuntimeError(f'Worker {case} stopped with exit {status}; all owned workers terminated')
            time.sleep(.5)
    finally:
        for _,proc,log in active:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=10)
            log.close()

if __name__=='__main__':main()
