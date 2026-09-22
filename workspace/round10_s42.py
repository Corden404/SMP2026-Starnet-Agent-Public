"""Fixed, fresh S42 follow-up. Never edits prior sources, studies, or submissions.

S42 was selected after round9 outcomes were viewed, so every outcome cohort here
has a NEW RNG identity. No tuning, recycling old confirmation, or adaptive N.
All participant code is copied byte-for-byte from round9's frozen S44/S42 builds.
"""
from __future__ import annotations
import argparse
import collections
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

os.environ['ANONYMIZED_TELEMETRY'] = 'False'
import numpy as np
import round9_common as common
import round9_runner as runner
import round9_api as api
from round9_lock import ExclusiveBatch
from round4_seeds import generate, FAMILIES

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'agent_notes/experiments/round10-s42'
OLD = ROOT / 'agent_notes/experiments/round9'
VARIANTS = ('S44', 'S42')
COUNTS = {'confirm': 96, 'stress': 24, 'glm': 24, 'accept': 6}
ORIGINAL_FINGERPRINT = common.fingerprint
save = common.save
immutable = common.immutable_json
file_hash = common.file_hash
digest = common.digest


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def specs(pool):
    """Six equally weighted graph families; cycle scale and persona separately."""
    rows = []
    for i in range(COUNTS[pool]):
        f, rep = i % 6, i // 6
        rows.append(dict(case=f'r10-s42-{pool}-{i:03}', pool=f'r10-s42-{pool}',
            family=FAMILIES[f], scale=(.55, 1., 1.5)[(rep + f) % 3],
            persona_mix=(rep // 3 + 2 * f) % 3, rep=rep,
            mode=('degree_w', 'degree_r', 'community', 'w_r')[rep % 4] if pool == 'stress' else 'independent',
            n=50, ood=pool == 'stress', generator_version=1))
    return rows


def fingerprint(variant):
    result = ORIGINAL_FINGERPRINT(variant)
    result['followup_files'] = {name: file_hash(ROOT / name) for name in
        ('workspace/round10_s42.py', 'tests/test_round10_s42.py')}
    return result


def configure():
    # Per-process module bindings only. No source file is changed.
    common.OUT = OUT
    runner.OUT = OUT
    api.OUT = OUT
    runner.fingerprint = fingerprint


def snapshot():
    path = OUT / 'original-workspace.json'
    if path.exists():
        return read(path)
    raw = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=ROOT)
    names = {n for n in raw.decode('utf-8').split('\0') if n and (ROOT / n).is_file()}
    names.update(p.relative_to(ROOT).as_posix() for p in (ROOT / 'dist').glob('*.zip'))
    # Freeze the old scientific evidence too; do not include an active closeout's output.
    for sub in ('protocol.json', 'candidate-freeze.json', 'primary-candidate-freeze.json', 'q44-supplement.json'):
        p = OLD / sub
        if p.exists(): names.add(p.relative_to(ROOT).as_posix())
    names.update(p.relative_to(ROOT).as_posix() for p in (OLD / 'summaries').glob('*.json'))
    result = dict(head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        branch=subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip(),
        status=subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True),
        files={n: file_hash(ROOT / n) for n in sorted(names)})
    immutable(path, result)
    return result


def verify():
    initial = read(OUT / 'original-workspace.json')
    changed = [n for n, h in initial['files'].items() if not (ROOT / n).is_file() or file_hash(ROOT / n) != h]
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip()
    if changed or head != initial['head'] or branch != initial['branch']:
        raise RuntimeError('Preserved source/evidence drift: ' + repr(changed))
    if file_hash(common.ARCHIVE) != common.ARCHIVE_SHA:
        raise RuntimeError('876-point S44 ZIP changed')
    return dict(status='PASS', protected_files=len(initial['files']), head_unchanged=True,
                branch_unchanged=True, s44_sha256=file_hash(common.ARCHIVE))


def prepare():
    configure()
    snapshot()
    for v in VARIANTS: common.stage(v)
    old = read(OLD / 'protocol.json')
    for v in VARIANTS:
        if ORIGINAL_FINGERPRINT(v) != old['code'][v]:
            raise RuntimeError('Round9 frozen candidate/evaluator has drifted: ' + v)
    hashes = {r['sha256'] for pool in old['seeds'].values() for r in pool}
    seeds = {p: [dict(spec=s, sha256=digest(generate(s))) for s in specs(p)] for p in COUNTS}
    new = [r['sha256'] for pool in seeds.values() for r in pool]
    if len(new) != len(set(new)) or hashes.intersection(new):
        raise RuntimeError('New cohort collision')
    protocol = dict(baseline='S44', candidate='S42', change='scan_target 44 -> 42 only',
        rationale='Remaining positive development candidate. Selected AFTER viewing round9 Q44/B44, therefore use entirely fresh cohorts.',
        independent_unit='graph; same synthetic graph paired across policies; no old results pooled',
        counts=COUNTS, repeats=1, workers=2, seeds=seeds,
        code={v: fingerprint(v) for v in VARIANTS},
        local_gate='Complete confirm96 AND stress24. Confirm mean>0 and without_best_two>0; stress mean>=0. Otherwise no paid validation.',
        real_gate='Fixed 24 NEW independent graphs, one complete real-LLM run per policy (48 games). Mean>0 and without_best_two>0; ci95 lower>0 for confirmed evidence. No sample extension.',
        final_gate='Recommend replacement only when local and real means/without_best_two positive, both confirm and real ci95 lower>0, stress nonnegative, resource and actual ZIP acceptance pass. Positive inconclusive evidence is exploratory, never claimed confirmed.',
        no_tuning=True, no_old_confirmation_reuse=True, no_upload=True,
        api_priority=list(api.MODELS), model_change='Only service/credit failure; complete separate same-cohort paired directory per model, never pool or switch mid-game',
        unknown_diagnostic='Prior 19/36 sensitivity states are not measured regret; do not turn them directly into a new policy or assume rescanning improves score.')
    immutable(OUT / 'protocol.json', protocol)
    print(json.dumps(dict(prepared=True, counts=COUNTS, verification=verify())), flush=True)


def check_protocol():
    protocol = read(OUT / 'protocol.json')
    for v in VARIANTS:
        if protocol['code'][v] != fingerprint(v):
            raise RuntimeError('Frozen policy or follow-up harness changed')
    verify()
    return protocol


def run_pair(pool, index, mode, model):
    configure()
    protocol = check_protocol()
    entry = protocol['seeds'][pool][index]
    s = entry['spec']
    if s != specs(pool)[index] or digest(generate(s)) != entry['sha256']:
        raise RuntimeError('Frozen cohort changed')
    # Alternate order within each family, balanced over its repetitions.
    order = VARIANTS if (index // 6 + index % 6) % 2 == 0 else tuple(reversed(VARIANTS))
    results = [runner.run_one(s, v, mode, 0, model) for v in order]
    return dict(case=s['case'], scores={r['variant']: r['final_score'] for r in results})


def summary(pool, mode='fast', model=None):
    configure()
    protocol = check_protocol()
    directory = runner.folder('r10-s42-' + pool, mode, model)
    pairs, all_rows = [], []
    costs = dict(scan_node=.5, communicate=2., cut_link=3., shield_node=5.)
    for entry in protocol['seeds'][pool]:
        s = entry['spec']; rows = {}
        for v in VARIANTS:
            path = directory / (s['case'] + '-' + v + '-r0.json')
            r = read(path)
            if r['status'] != 'PASS' or r['identity']['code'] != fingerprint(v) or r['identity']['seed_sha256'] != entry['sha256']:
                raise RuntimeError('Incomplete/changed pair: ' + str(path))
            if r['identity']['model_config'] != (dict(api.CONFIG, model=model) if model else None):
                raise RuntimeError('Model-config mixing')
            if not (r['operation_count'] == len(r['actions']) < 120 and r['http_calls'] <= r['logical_calls'] < 120):
                raise RuntimeError('Resource accounting error')
            if abs(100 - sum(costs[a['action']] for a in r['actions']) - r['remaining_budget']) > 1e-8:
                raise RuntimeError('Budget error')
            if r['action_counts']['scan_node'] != int(v[1:]):
                raise RuntimeError('Scan-count drift')
            rows[v] = r; all_rows.append(r)
        b, c = rows['S44'], rows['S42']
        def actions(row):
            return collections.Counter((a['action'], tuple(a['args'])) for a in row['actions'] if a['action'] != 'scan_node')
        ba, ca = actions(b), actions(c)
        pairs.append(dict(case=s['case'], family=s['family'], delta=c['final_score'] - b['final_score'],
            comm_delta=c['action_counts'].get('communicate', 0) - b['action_counts'].get('communicate', 0),
            shield_delta=c['action_counts'].get('shield_node', 0) - b['action_counts'].get('shield_node', 0),
            cut_delta=c['action_counts'].get('cut_link', 0) - b['action_counts'].get('cut_link', 0),
            budget_left_delta=c['remaining_budget'] - b['remaining_budget'],
            same_intervention_multiset=ba == ca))
    ds = [p['delta'] for p in pairs]
    rng = np.random.default_rng(109192026); boot = np.zeros(10000)
    for f in FAMILIES:
        values = np.array([p['delta'] for p in pairs if p['family'] == f])
        boot += rng.choice(values, (len(boot), len(values)), replace=True).sum(axis=1) / len(ds)
    result = dict(pool=pool, mode=mode, model=model, n=len(ds), games=len(all_rows),
        mean=statistics.mean(ds), ci95=np.quantile(boot, [.025, .975]).tolist(),
        sd=statistics.stdev(ds), without_best_two=statistics.mean(sorted(ds)[:-2]),
        wins=sum(d > 1e-7 for d in ds), ties=sum(abs(d) <= 1e-7 for d in ds), losses=sum(d < -1e-7 for d in ds),
        worst_decile=statistics.mean(sorted(ds)[:max(1, math.ceil(.1 * len(ds)))]), min=min(ds), max=max(ds),
        families={f: statistics.mean(p['delta'] for p in pairs if p['family'] == f) for f in FAMILIES},
        max_operations=max(r['operation_count'] for r in all_rows), max_calls=max(r['logical_calls'] for r in all_rows),
        http_calls=sum(r['http_calls'] for r in all_rows),
        mechanism={k: statistics.mean(p[k] for p in pairs) for k in ('comm_delta', 'shield_delta', 'cut_delta', 'budget_left_delta')},
        same_intervention_multiset=sum(p['same_intervention_multiset'] for p in pairs),
        truncations={v:sum(bool(s['plans'][0].get('search_truncated')) for r in all_rows if r['variant'] == v for s in r['search_records'] if s['plans']) for v in VARIANTS},
        max_seconds=max(r['elapsed_seconds'] for r in all_rows), pairs=pairs,
        scope='Custom synthetic finite50 paired outcomes, NOT Tianchi hidden score', originals=verify())
    save(OUT / 'summaries' / (pool + '-' + mode + ('-' + model if model else '') + '.json'), result)
    return result


def run_batch(pool, mode='fast', model=None):
    configure(); check_protocol()
    manifest = dict(pool=pool, mode=mode, model=model, count=COUNTS[pool], repeats=1, workers=2)
    immutable(OUT / 'batches' / (pool + '-' + mode + ('-' + model if model else '') + '.json'), manifest)
    # At most two pairs in flight, never queue an entire paid cohort past failure.
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as ex:
        pending = {}; next_index = 0; complete = 0
        for _ in range(min(2, COUNTS[pool])):
            future = ex.submit(run_pair, pool, next_index, mode, model); pending[future] = next_index; next_index += 1
        while pending:
            done, _ = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                pending.pop(future)
                try:
                    result = future.result()
                except BaseException:
                    for f in pending: f.cancel()
                    raise
                complete += 1
                if mode != 'fast' or complete % 6 == 0 or complete == COUNTS[pool]:
                    print(json.dumps(dict(pool=pool, completed_pairs=complete, total_pairs=COUNTS[pool], **result)), flush=True)
                if next_index < COUNTS[pool]:
                    f = ex.submit(run_pair, pool, next_index, mode, model); pending[f] = next_index; next_index += 1
    result = summary(pool, mode, model)
    print(json.dumps({k: v for k, v in result.items() if k != 'pairs'}), flush=True)
    return result


def local_eligible(confirm, stress):
    return (confirm['n'] == 96 and stress['n'] == 24 and confirm['mean'] > 0
            and confirm['without_best_two'] > 0 and stress['mean'] >= 0)


def local():
    c = run_batch('confirm'); s = run_batch('stress')
    decision = dict(status='LOCAL_COMPLETE', eligible_for_real_glm=local_eligible(c, s),
        confirm_mean=c['mean'], stress_mean=s['mean'], no_result_dependent_extension=True,
        no_new_policy=True, originals=verify())
    immutable(OUT / 'local-decision.json', decision)
    print(json.dumps(decision), flush=True)


def real(model_override=None):
    if not read(OUT / 'local-decision.json')['eligible_for_real_glm']:
        raise RuntimeError('Local gate failed: do not spend on a rejected candidate')
    configure()
    if model_override:
        if model_override not in api.MODELS: raise ValueError(model_override)
        selected = read(OUT / 'api/selected.json')['model']
        if model_override != selected:
            old_stop = runner.folder('r10-s42-glm', 'glm', selected) / 'STOP.json'
            if not old_stop.exists(): raise RuntimeError('Model changes require an observed service failure, never a bad score')
        model = model_override
    else:
        model = api.probe()['model']
    result = run_batch('glm', 'glm', model)
    immutable(OUT / 'real-decision.json', dict(model=model, mean=result['mean'], ci95=result['ci95'],
        positive=result['mean'] > 0 and result['without_best_two'] > 0,
        confirmed=result['ci95'][0] > 0 and result['without_best_two'] > 0))


def report():
    configure(); check_protocol()
    summaries = {p:read(OUT / 'summaries' / (p + '-fast.json')) for p in ('confirm', 'stress')}
    decision = read(OUT / 'local-decision.json')
    status = 'KEEP_S44_LOCAL_GATE_FAILED'
    if decision['eligible_for_real_glm']:
        status = 'LOCAL_POSITIVE_REAL_VALIDATION_PENDING'
        if (OUT / 'real-decision.json').exists():
            rd = read(OUT / 'real-decision.json')
            summaries['glm'] = read(OUT / 'summaries' / ('glm-glm-' + rd['model'] + '.json'))
            status = ('POSITIVE_PENDING_ACTUAL_ZIP_ACCEPTANCE' if rd['confirmed'] and summaries['confirm']['ci95'][0] > 0
                      else 'EXPLORATORY_ONLY' if rd['positive'] else 'KEEP_S44_REAL_GATE_FAILED')
    compact = {p: {k: v for k, v in r.items() if k != 'pairs'} for p, r in summaries.items()}
    complete = dict(status=status, recommended='S44 / user-reported 876 remains unchanged',
        summaries=compact, originals=verify(), submitted=False, git_commit=False,
        paid_validation_performed='glm' in summaries)
    save(OUT / 'completion.json', complete)
    lines = ['# S42 独立补充验证', '', '状态：`' + status + '`。官网 S44 / 876 未替换，未上传官网。', '',
        'S42 只把扫描数 44 改为 42；评分、搜索、响应估计、提示词全部保持不变。',
        '选型发生在第九轮结果已知之后，因此本轮使用全新 RNG 身份的确认/压力/真实模型种子，不复用旧确认结果。', '',
        '| 阶段 | 独立图数 | 候选−S44均值 | 95%分图族bootstrap区间 | 胜/平/负 | 去掉最好两图 |',
        '|---|---:|---:|---|---|---:|']
    for p, r in summaries.items():
        lines.append(f"| {p} | {r['n']} | {r['mean']:+.4f} | [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}] | {r['wins']}/{r['ties']}/{r['losses']} | {r['without_best_two']:+.4f} |")
    lines += ['', '这些分差仅来自自建图的 finite50 结算，不是官网分差；置信区间也不覆盖未知官方分布差异。', '',
        '本地门槛：完整96图确认均值及去掉最好两图后的均值均为正，完整24图压力均值非负，才进入真实GLM。真实模型验证固定24张新图，每策略每图一次；不因结果扩样，不拼接模型。', '',
        '未知节点敏感性19/36只证明动作排名会变化，不证明补扫可得分。本轮不加入没有整局收益依据的图补全、自适应补扫或候选组合。', '',
        '所有原文件和基线包的保护检查：', '```json', json.dumps(complete['originals'], ensure_ascii=False, indent=2), '```', '',
        '完整统计（含图族、尾部、动作数量与截断）：', '```json', json.dumps(compact, ensure_ascii=False, indent=2), '```', '',
        '复现：`.venv/python.exe workspace/round10_s42.py prepare`，随后 `local`；只有本地门槛通过才运行 `glm`，最后 `report`。既有同身份成功结果复用，失败记录不自动删除或盲目重试。',
        '数据：`agent_notes/experiments/round10-s42/`。本轮没有修改原源码、提示词、配置、旧实验结果，也没有提交/推送 Git。']
    target = OUT / 'execution-report.md'
    target.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=status, report=str(target), originals=verify())), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=('prepare', 'local', 'glm', 'report', 'verify'))
    ap.add_argument('--model', choices=list(api.MODELS))
    args = ap.parse_args(); configure()
    with ExclusiveBatch(OUT / 'execution.lock'):
        # Cooperate with any old closeout/runner without editing its lock content.
        with ExclusiveBatch(OLD / 'execution.lock'):
            if args.command == 'prepare': prepare()
            elif args.command == 'local': local()
            elif args.command == 'glm': real(args.model)
            elif args.command == 'report': report()
            else: print(json.dumps(verify()), flush=True)


if __name__ == '__main__': main()
