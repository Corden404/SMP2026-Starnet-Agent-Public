"""Prospective fixed-size study launcher; additive outputs, never upload or overwrite."""
from __future__ import annotations
import argparse
import collections
import concurrent.futures
import json
import math
import statistics
import numpy as np
from round9_common import (ROOT, OUT, VARIANTS, COUNTS, ARCHIVE_SHA, stage, source_files,
    snapshot, specs, fingerprint, immutable_json, save, digest, verify_originals)
from round9_runner import run_one, folder
from round4_seeds import generate


def prepare():
    snapshot()
    for v in VARIANTS:
        stage(v)
    protocol = dict(baseline='S44',baseline_archive_sha256=ARCHIVE_SHA,
        purpose='Synthetic public-state policy comparison; not Tianchi hidden score',
        variants={'Q44':'refresh within approved queue','B44':'batch2 after32; original scan prompt',
                  'I44':'public degree/features and remaining slots; unchanged batch10',
                  'S42':'scan42 only','S40':'scan40 only'},
        counts=COUNTS,workers=2,
        phases=['Q44 versus S44 on dev24','B44 S40 S42 on same development bank',
                'scout-only real GLM on glm-dev12 for S44 Q44 B44 I44; deterministic singleton commander, explicitly not full GLM',
                'freeze up to one local algorithm and one real-scout candidate before new confirmation',
                'algorithm candidate confirm96 and stress24, no retuning',
                'full CaseVO+real GLM finalist versus S44 on glm-check12 x2 repeats; no result-dependent extension',
                'actual packaged entry, 4/100 compatibility and public sandbox acceptance if reachable'],
        selection='Positive development mean and positive mean after removing best two; rank by mean; if no candidate qualifies preserve S44. Real-scout candidates also require >0 mean; report small-n uncertainty.',
        final_gate='Full fixed pool; paired bootstrap95 lower>0 and without_best_two>0 and resource valid. Positive inconclusive results are exploratory only; nonpositive means do not replace S44.',
        no_addition_of_variant_gains=True,no_tuning_on_confirmation=True,
        model_priority=['glm-4.5-air','glm-4.6v'],
        model_replacement='New complete paired directory per model; never switch mid-game or pool models; zero blind retries',
        code={v:fingerprint(v) for v in VARIANTS},
        seeds={pool:[dict(spec=s,sha256=digest(generate(s))) for s in specs(pool)] for pool in COUNTS})
    hashes=[row['sha256'] for pool in protocol['seeds'].values() for row in pool]
    if len(hashes)!=len(set(hashes)):
        raise RuntimeError('Seed collision')
    immutable_json(OUT/'protocol.json',protocol)
    return protocol


def summarize(pool,mode,api_model=None):
    directory=folder('r9-'+pool,mode,api_model)
    rows=[json.loads(p.read_text(encoding='utf-8')) for p in directory.glob('r9-*.json')]
    valid=[r for r in rows if r.get('status')=='PASS']
    groups=collections.defaultdict(list)
    for r in valid:
        groups[(r['case'],r['variant'])].append(r)
    output={}
    for variant in sorted({r['variant'] for r in valid}-{'S44'}):
        pairs=[]
        for (case,v),rs in sorted(groups.items()):
            if v!=variant:continue
            bs=groups.get((case,'S44'),[])
            if not bs or {r['repeat'] for r in rs}!={r['repeat'] for r in bs}:continue
            pairs.append(dict(case=case,family=rs[0]['spec']['family'],
                              delta=statistics.mean(r['final_score'] for r in rs)-statistics.mean(r['final_score'] for r in bs)))
        if not pairs:continue
        ds=[r['delta'] for r in pairs]; n=len(ds)
        rng=np.random.default_rng(9192026);boot=np.zeros(6000)
        for f in sorted({r['family'] for r in pairs}):
            values=np.array([r['delta'] for r in pairs if r['family']==f])
            boot+=rng.choice(values,(6000,len(values)),replace=True).sum(axis=1)/n
        vr=[r for r in valid if r['variant']==variant]
        output[variant]=dict(n=n,games=len(vr),mean=statistics.mean(ds),
            ci95=np.quantile(boot,[.025,.975]).tolist(),sd=statistics.stdev(ds) if n>1 else 0.,
            wins=sum(d>1e-7 for d in ds),ties=sum(abs(d)<=1e-7 for d in ds),losses=sum(d< -1e-7 for d in ds),
            without_best_two=statistics.mean(sorted(ds)[:-2]) if n>2 else None,
            worst_decile=statistics.mean(sorted(ds)[:max(1,math.ceil(n*.1))]),min=min(ds),max=max(ds),
            families={f:statistics.mean(r['delta'] for r in pairs if r['family']==f) for f in sorted({r['family'] for r in pairs})},
            max_operations=max(r['operation_count'] for r in vr),max_calls=max(r['logical_calls'] for r in vr),
            mean_seconds=statistics.mean(r['elapsed_seconds'] for r in vr),
            search_truncations=sum(bool(s['plans'][0].get('search_truncated')) for r in vr for s in r['search_records'] if s['plans']),
            action_counts=dict(sum((collections.Counter(r['action_counts']) for r in vr),collections.Counter())),
            deltas=pairs)
    result=dict(pool=pool,mode=mode,api_model=api_model,statuses=dict(collections.Counter(r.get('status') for r in rows)),
                comparisons=output,independent_unit='graph; repeat scores averaged within graph')
    save(OUT/'summaries'/(pool+'-'+mode+('-'+api_model if api_model else '')+'.json'),result)
    return result


def run_batch(pool,variants,mode,repeats,workers):
    protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    for v in variants:
        if fingerprint(v)!=protocol['code'][v]:
            raise RuntimeError('Frozen candidate or harness changed')
    api_model=None
    if mode in ('glm','scout_glm'):
        from round9_api import probe
        api_model=probe()['model']
    if pool in ('confirm','stress','glm-check'):
        selection=json.loads((OUT/'candidate-freeze.json').read_text(encoding='utf-8'))
        allowed=set(selection['variants'])|{'S44'}
        if not set(variants)<=allowed:
            raise RuntimeError('Unfrozen candidate requested on confirmation pool')
    jobs=[]
    for i,s in enumerate(specs(pool)):
        for repeat in range(repeats):
            order=list(variants)
            if (i//6+i+repeat)%2:order.reverse()
            jobs.extend((s,v,mode,repeat,api_model) for v in order)
    batch_identity=dict(pool=pool,variants=variants,mode=mode,repeats=repeats,api_model=api_model,
        workers=workers,jobs=len(jobs),code={v:fingerprint(v) for v in variants})
    manifest=OUT/'batches'/(pool+'-'+mode+('-'+api_model if api_model else '')+'-'+'-'.join(variants)+'.json')
    immutable_json(manifest,batch_identity)
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures=[executor.submit(run_one,*job) for job in jobs]
        for index,future in enumerate(concurrent.futures.as_completed(futures),1):
            print(json.dumps(dict(completed=index,total=len(jobs),**future.result())),flush=True)
    summary=summarize(pool,mode,api_model)
    print(json.dumps({v:{k:x for k,x in s.items() if k!='deltas'} for v,s in summary['comparisons'].items()}),flush=True)
    print(json.dumps(verify_originals()),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['prepare','run','summarize','verify','probe'])
    ap.add_argument('--pool',choices=list(COUNTS),default='dev')
    ap.add_argument('--variants',nargs='+',choices=list(VARIANTS),default=['S44','Q44'])
    ap.add_argument('--mode',choices=['fast','integration','scout_glm','glm'],default='fast')
    ap.add_argument('--repeats',type=int,choices=[1,2],default=1)
    ap.add_argument('--workers',type=int,choices=[1,2],default=2)
    args=ap.parse_args()
    if args.command=='prepare':
        p=prepare();print(json.dumps(dict(prepared=True,variants=list(p['code']),counts=COUNTS)));return
    if args.command=='verify':print(json.dumps(verify_originals()));return
    if args.command=='probe':
        from round9_api import probe
        print(json.dumps(probe()));return
    if args.command=='run':
        from round9_lock import ExclusiveBatch
        with ExclusiveBatch(OUT/'execution.lock'):
            run_batch(args.pool,args.variants,args.mode,args.repeats,args.workers)
        return
    model=None
    if args.mode in ('glm','scout_glm'):
        model=json.loads((OUT/'api/selected.json').read_text(encoding='utf-8'))['model']
    result=summarize(args.pool,args.mode,model)
    for value in result['comparisons'].values():value.pop('deltas',None)
    print(json.dumps(result,ensure_ascii=True,indent=2))

if __name__=='__main__':main()
