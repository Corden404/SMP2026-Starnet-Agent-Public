"""Independent result and integration consistency checks for round7."""
import collections
import json
import statistics
from experiment_round7 import OUT, source_hashes
from round6_common import D5_SOURCE, D5_ARCHIVE, D5_SHA256, file_hash, save

def audit():
    manifest=json.loads((D5_SOURCE.parent.parent/'manifest.json').read_text(encoding='utf-8'))
    assert file_hash(D5_ARCHIVE)==D5_SHA256
    assert all(file_hash(D5_SOURCE.parent/p)==h for p,h in manifest['submission_files'].items())
    result=dict(baseline_intact=True,staged_files_verified=len(manifest['submission_files']),pools={})
    costs=dict(scan_node=.5,communicate=2,cut_link=3,shield_node=5)
    for folder in sorted(OUT.glob('r7-*')):
        rows=[json.loads(p.read_text(encoding='utf-8')) for p in folder.glob('*.json')]
        if not rows:continue
        variants={}
        for r in rows:
            assert r['identity']['code']==source_hashes(r['variant'])
            assert r['operation_count']==len(r['actions'])
            assert abs(100-sum(costs[a['action']] for a in r['actions'])-r['remaining_budget'])<1e-8
            assert r['network_attempts']==0
            assert r['logical_calls']<120 and r['operation_count']<120
            if r['variant'] in ('S48','S49'):
                assert sum(a['action']=='scan_node' for a in r['actions'])==int(r['variant'][1:])
                for rec in r['scan_diagnostic_records']:
                    if rec['event']=='intervention':
                        assert rec['graph_vertices']+rec['dead']==50
                        assert rec['known_alive']+rec['unseen']==rec['graph_vertices']
        for v in sorted({r['variant'] for r in rows}):
            rs=[r for r in rows if r['variant']==v]
            post=collections.Counter();multiple=0
            for r in rs:
                names=[a['action'] for a in r['actions']]
                multiple+=names.count('shield_node')>1
                if 'shield_node' in names:post.update(names[names.index('shield_node')+1:])
            exchange=[rec for r in rs for rec in r['exchange_records']]
            variants[v]=dict(n=len(rs),mean_seconds=statistics.mean(r['elapsed_seconds'] for r in rs),
                max_seconds=max(r['elapsed_seconds'] for r in rs),
                max_operations=max(r['operation_count'] for r in rs),
                max_logical_calls=max(r['logical_calls'] for r in rs),
                post_first_shield_actions=dict(post),games_with_multiple_shields=multiple,
                mean_remaining_budget=statistics.mean(r['remaining_budget'] for r in rs),
                exchange_evaluations=sum(x.get('evaluations',0) for x in exchange),
                exchange_selected=sum(x['selected'] for x in exchange),
                exchange_changed_first=sum(x.get('changed_first',False) for x in exchange),
                exchange_timeouts=sum(x.get('reason')=='reference_timeout' for x in exchange),
                exchange_truncations=sum(x.get('truncated',False) for x in exchange))
        result['pools'][folder.name]=variants
    integration=[]
    for p in sorted((OUT/'integration').glob('*.json')):
        r=json.loads(p.read_text(encoding='utf-8'));variant=r['variant']
        comparator='D5' if variant=='S50' else variant
        base=json.loads((OUT/r['spec']['pool']/(r['case']+'-'+comparator+'.json')).read_text(encoding='utf-8'))
        same_actions=[(x['action'],x['args']) for x in r['actions']]==[(x['action'],x['args']) for x in base['actions']]
        same_score=abs(r['final_score']-base['final_score'])<1e-8
        assert same_actions and same_score
        assert r['network_attempts']==0
        integration.append(dict(case=r['case'],variant=variant,same_actions=same_actions,same_score=same_score))
    result['integration']=integration
    save(OUT/'audit.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result

if __name__=='__main__':audit()
