"""Read-only audit helpers: exact identities, physical request accounting, complete real summaries."""
from __future__ import annotations
import collections
import json
from pathlib import Path
import round18_state as s
import round18_common as h


def validate_identity(folder,path,row,study,diagnostic,real=None):
    identity=row.get('identity',{})
    if folder.startswith('diagnostic/'):
        entries={e['spec']['case']:e for e in diagnostic['seeds']}
        entry=entries.get(row.get('spec',{}).get('case'))
        if entry is None or identity.get('entry')!=entry or identity.get('code')!=diagnostic['code']:
            raise RuntimeError('Diagnostic identity mismatch: '+str(path))
        if h.c.digest(h.c.generate(row['spec']))!=entry['sha256']:
            raise RuntimeError('Diagnostic seed digest mismatch')
        if folder=='diagnostic/branches':
            op=identity['snapshot']['operation']
            baseline=s.read(s.OUT/'diagnostic/baseline'/(entry['spec']['case']+'.json'))
            if row['actions'][:op]!=baseline['actions'][:op] or row['prefix_sha256']!=s.digest(baseline['actions'][:op]):
                raise RuntimeError('Diagnostic branch public prefix mismatch')
            if row['forced_action']!=identity['action'] or row['start_operation']!=op:
                raise RuntimeError('Diagnostic forced-action identity mismatch')
    elif folder.startswith('study/runs/'):
        pool=folder.split('/')[-1]
        entries={e['spec']['case']:e for e in study['seeds'][pool]}
        entry=entries.get(row.get('spec',{}).get('case'))
        if entry is None or identity.get('entry')!=entry or identity.get('code')!=study['code']:
            raise RuntimeError('Study identity mismatch: '+str(path))
        if row.get('variant')!=identity.get('variant') or path.name!=entry['spec']['case']+'-'+row['variant']+'.json':
            raise RuntimeError('Study variant/path mismatch')
        if row['variant'] not in study['variants'] or h.c.digest(h.c.generate(row['spec']))!=entry['sha256']:
            raise RuntimeError('Study seed/variant mismatch')
    elif folder=='real/runs':
        if real is None:raise RuntimeError('Real game has no frozen real protocol')
        entries={e['spec']['case']:e for e in real['seeds']}
        entry=entries.get(row.get('spec',{}).get('case'));variant=row.get('variant');repeat=row.get('repeat')
        if (entry is None or identity.get('entry')!=entry or identity.get('code')!=real['code'] or
            identity.get('model_config')!=real['model_config'] or variant not in real['variants'] or repeat not in (0,1) or
            identity.get('variant')!=variant or identity.get('repeat')!=repeat):
            raise RuntimeError('Real graph/model/variant/repeat identity mismatch')
        if path.name!=entry['spec']['case']+'-'+variant+'-r'+str(repeat)+'.json':raise RuntimeError('Real filename mismatch')
        if h.c.digest(h.c.generate(row['spec']))!=entry['sha256']:raise RuntimeError('Real seed digest mismatch')
    else:raise ValueError('Unexpected audit folder')
    if not 0<=row['logical_calls']<120 or not 0<=row['operations']<120:
        raise RuntimeError('Unexpected resource limit')
    if row.get('http_calls',0)>row['logical_calls']:raise RuntimeError('Physical calls exceed recorded logical calls')
    return True


def request_usage(folder):
    requests=collections.Counter();statuses=collections.Counter();tokens=collections.Counter();uncertain=[]
    for path in sorted(Path(folder).rglob('*.jsonl')):
        events=[json.loads(line) for line in path.read_text(encoding='utf8').splitlines() if line.strip()]
        req=[v for v in events if v.get('event')=='request'];response=[v for v in events if v.get('event')=='response']
        numbers=[v['number'] for v in req]
        if numbers!=list(range(1,len(numbers)+1)):raise RuntimeError('Physical request journal numbering mismatch')
        seen_response=set()
        for row in req:requests[row['data']['model']]+=1
        for row in response:
            if row['number'] in seen_response or row['number'] not in numbers:raise RuntimeError('Duplicate/unpaired API response')
            seen_response.add(row['number']);statuses[str(row['status'])]+=1
            try:body=json.loads(row['body'])
            except (ValueError,KeyError):body={}
            usage=body.get('usage') if isinstance(body,dict) else None
            if isinstance(usage,dict):
                for key,value in usage.items():
                    if key in ('prompt_tokens','completion_tokens','total_tokens') and isinstance(value,int):tokens[key]+=value
        marked={v['number']:v for v in events if v.get('event')=='uncertain'}
        for row in req:
            number=row['number']
            if number not in seen_response:
                marker=marked.get(number,{})
                uncertain.append(dict(path=path.relative_to(folder).as_posix(),number=number,
                    error_type=marker.get('error_type','ORPHAN_REQUEST_NO_RESPONSE'),may_be_billed=True))
    return dict(requests=dict(requests),request_count=sum(requests.values()),responses=sum(statuses.values()),
        http_statuses=dict(statuses),reported_tokens=dict(tokens),uncertain=uncertain,
        usage_is_not_cost=True,replayed_responses_counted_as_new=False)


def audit_real_summary(protocol):
    folder=s.OUT/'real';path=folder/'summary.json'
    if not path.exists():return dict(status='NOT_COMPLETE')
    summary=s.read(path);pairs=[];prefix_diffs=0
    for entry in protocol['seeds']:
        case=entry['spec']['case'];deltas=[]
        for repeat in range(2):
            rows={}
            for variant in protocol['variants']:
                p=folder/'runs'/(case+'-'+variant+'-r'+str(repeat)+'.json');row=s.read(p)
                if row['status']!='PASS':raise RuntimeError('Real summary contains failed/incomplete game')
                rows[variant]=row
                events=[json.loads(line) for line in p.with_suffix('.http.jsonl').read_text(encoding='utf8').splitlines()]
                requests=[x for x in events if x.get('event')=='request'];responses=[x for x in events if x.get('event')=='response']
                if len(requests)!=row['http_calls'] or len(responses)!=row['http_calls'] or any(x['status']!=200 for x in responses):
                    raise RuntimeError('Real PASS journal/resource mismatch')
                if any(x['data']['model']!=protocol['model'] for x in requests):raise RuntimeError('Mixed request model')
                if any(json.loads(x['body']).get('model')!=protocol['model'] for x in responses):raise RuntimeError('Mixed response model')
            deltas.append(rows[protocol['candidate']]['final_score']-rows['PC2']['final_score'])
            prefix_diffs+=int(h.scan_prefix(rows['PC2'])!=h.scan_prefix(rows[protocol['candidate']]))
        pairs.append(dict(case=case,family=entry['spec']['family'],delta=sum(deltas)/2,repeat_deltas=deltas))
    if summary['statistics']!=h.c.paired_stats(pairs) or summary['pairs']!=pairs or summary['scan_prefix_differences']!=prefix_diffs:
        raise RuntimeError('Real summary fails independent graph-first recomputation')
    return dict(status='PASS',independent_graphs=len(pairs),games=len(pairs)*4,model=protocol['model'])
