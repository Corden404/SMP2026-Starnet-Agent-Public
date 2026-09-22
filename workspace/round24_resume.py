"""Round24 status and API inventory."""
from __future__ import annotations
import json
import round24_state as s
def api_usage():
    out=dict(requests=0,responses=0,uncertain=[])
    for fp in s.OUT.rglob('*.jsonl'):
        if fp.name=='events.jsonl':continue
        for line in fp.read_text(encoding='utf8').splitlines():
            try:e=json.loads(line)
            except ValueError:continue
            if e.get('event')=='request':out['requests']+=1
            elif e.get('event')=='response':out['responses']+=1
            elif e.get('event')=='uncertain':out['uncertain'].append(fp.relative_to(s.OUT).as_posix())
    return out
def main():
    r=dict(progress=s.read(s.OUT/'progress.json'),originals=s.preserved(),api_usage=api_usage())
    print(json.dumps(r,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
