"""Audit completed records and aggregate GLM usage without reading credentials."""
import collections
import json
import statistics
from round8_common import OUT, ARCHIVE, ARCHIVE_SHA, file_hash, save

def main():
    result=dict(baseline_unchanged=file_hash(ARCHIVE)==ARCHIVE_SHA,pools={},glm_usage=collections.Counter(),failures=[])
    for folder in sorted(OUT.glob('r8-*')):
        if not folder.is_dir():continue
        rows=[json.loads(p.read_text(encoding='utf-8')) for p in folder.glob('*.json')]
        if not rows:continue
        complete=[r for r in rows if r.get('status')=='PASS']
        result['pools'][folder.name]=dict(statuses=dict(collections.Counter(r['status'] for r in rows)),
            variants=dict(collections.Counter(r['variant'] for r in complete)),
            max_operations=max((r['operation_count'] for r in complete),default=0),
            max_calls=max((r['logical_calls'] for r in complete),default=0))
        for r in complete:
            assert r['identity']['code']['archive']==ARCHIVE_SHA
            assert r['operation_count']<120 and r['logical_calls']<120
            assert r['network_attempts']==0
            assert len(r['actions'])==r['operation_count']
            if r['mode']=='glm':
                result['glm_usage']['games']+=1;result['glm_usage']['http_calls']+=r['http_calls']
                for md in r['metadata']:
                    for k,v in md.get('usage',{}).items():
                        if isinstance(v,(int,float)):result['glm_usage'][k]+=v
        result['failures'] += [dict(case=r['case'],variant=r['variant'],error=r.get('error_type')) for r in rows if r['status']=='FAILED']
    basefolder=OUT/'r8-glm-base-glm';base={}
    for p in basefolder.glob('*.json'):
        r=json.loads(p.read_text(encoding='utf-8'))
        if r['status']=='PASS':base.setdefault(r['case'],[]).append(r)
    records=[]
    for case,rs in sorted(base.items()):
        reference=OUT/'r8-glm-base-integration'/(case+'-S48-r0.json')
        if not reference.exists():continue
        fake=json.loads(reference.read_text(encoding='utf-8'))
        key=lambda r: [(a['action'],a['args']) for a in r['actions']]
        records.append(dict(case=case,repeats=len(rs),glm_mean=statistics.mean(r['final_score'] for r in rs),
            glm_range=max(r['final_score'] for r in rs)-min(r['final_score'] for r in rs),
            fake_score=fake['final_score'],delta_vs_fake=statistics.mean(r['final_score'] for r in rs)-fake['final_score'],
            changed_trajectories=sum(key(r)!=key(fake) for r in rs),omitted=[r['omitted'] for r in rs],
            null_chain_outputs=sum(c['output'] is None for r in rs for c in r['chains'])))
    result['glm_baseline']=records
    save(OUT/'audit.json',result)
    print(json.dumps(result,ensure_ascii=True,indent=2))

if __name__=='__main__':main()
