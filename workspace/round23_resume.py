"""Round23 durable status inventory. No API replay."""
from __future__ import annotations
import json
from pathlib import Path
import round23_state as s

def api_usage():
    usage=dict(requests=0,responses=0,uncertain=[])
    for fp in s.OUT.rglob('*.jsonl'):
        if fp.name=='events.jsonl':continue
        for line in fp.read_text(encoding='utf8').splitlines():
            try:e=json.loads(line)
            except ValueError:continue
            if e.get('event')=='request':usage['requests']+=1
            elif e.get('event')=='response':usage['responses']+=1
            elif e.get('event')=='uncertain':usage['uncertain'].append(fp.relative_to(s.OUT).as_posix())
    return usage

def status():
    result=dict(progress=s.read(s.OUT/'progress.json'),originals=s.preserved(),api_usage=api_usage())
    print(json.dumps(result,ensure_ascii=False,indent=2));return result

if __name__=='__main__':status()
