"""Round 9: exact S44 baseline, additive-only files and frozen prospective studies."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'agent_notes/experiments/round9'
ARCHIVE = ROOT / 'dist/team_submission-round8-s44-local-confirmed-20260919.zip'
ARCHIVE_SHA = '7b502a2865ef14b726810ed3aa51e44ff14127af07bad3b8e4acb2f90b1c87f5'
VARIANTS = ('S44', 'Q44', 'B44', 'I44', 'S42', 'S40')
COUNTS = {'dev': 24, 'confirm': 96, 'stress': 24, 'glm-dev': 12,
          'glm-check': 12, 'accept': 6}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != value:
            raise RuntimeError('Frozen record drift: ' + str(path))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)


QUEUE_PATCH = '''

class ParticipantSquadModel(ParticipantSquadModel):
    """Re-rank only the remaining LLM-approved scan batch using public edges."""
    def _should_scan(self, budget):
        result = super()._should_scan(budget)
        if result and self.node_count == 50 and self.scan_queue:
            rank = {node: index for index, node in enumerate(self._scan_priority())}
            self.scan_queue = sorted((i for i in self.scan_queue if i in rank), key=rank.__getitem__)
        return result
'''

SCAN_PATCH = '''

class ParticipantSquadModel(ParticipantSquadModel):
    """Isolated scan-only variant; original intervention path is unchanged."""
    enrich_scan = ENRICH_VALUE
    late_batch = BATCH_VALUE

    def _planned_step(self):
        if self.node_count != 50 or self.stopped:
            return super()._planned_step()
        budget = self._budget()
        if budget is None:
            return self._stop('Environment budget unavailable')
        if self.operation_count >= self.operation_limit - 1 or self.llm_counter.call_count >= self.llm_counter.safe_limit:
            return self._stop('Resource safety limit')
        if not self._should_scan(budget):
            return super()._planned_step()
        if budget < .5:
            return self._stop('No scan budget')
        priority = self._scan_priority()
        self.scan_queue = [i for i in self.scan_queue if i in priority]
        # A single transition at 32 known vertices; subsequent batches have size two.
        if self.late_batch < 10 and self._known_count() == 32:
            self.scan_queue = []
        if not self.scan_queue:
            batch_size = self.late_batch if self._known_count() >= 32 else 10
            allowed = priority[:batch_size]
            data = {'budget': budget, 'unknown_ids': allowed,
                    'frontier': allowed, 'node_count': self.node_count}
            if self.enrich_scan:
                features = []
                for rank, i in enumerate(allowed):
                    observed_neighbors = sorted(v if u == i else u for u, v in self.local_edges if i in (u, v))
                    features.append({'id': i, 'rank': rank + 1,
                                     'observed_degree': len(observed_neighbors),
                                     'scanned_neighbors': observed_neighbors})
                data.update(scanned=self._known_count(), scan_target=self.scan_target,
                            remaining_scans=max(0, self.scan_target-self._known_count()),
                            candidate_features=json.dumps(features, ensure_ascii=False))
            decision = self.scout._run_chain('batch', data)
            targets = decision.get('targets', []) if isinstance(decision, dict) else []
            if isinstance(targets, list):
                self.scan_queue = list(dict.fromkeys(i for i in targets if type(i) is int and i in allowed))
            self.scan_queue = self.scan_queue or allowed
        action = self._action_dict('scan', self.scan_queue.pop(0))
        return self._execute_planned_action(action, budget, None)
'''

INFO_PROMPT = '''你是{{agent.description.role}}。目标是在有限扫描预算内选择有助于最终援助总分的节点，不是扫描全部节点。
当前预算{{extra.budget}}；总节点{{extra.node_count}}；已扫描{{extra.scanned}}；本局计划扫描{{extra.scan_target}}；剩余扫描名额{{extra.remaining_scans}}。
本批唯一合法候选ID（按已发现连接数降序、同分按ID升序排列）：{{extra.unknown_ids}}
每个候选的公开特征：{{extra.candidate_features}}
observed_degree是已发现的连接数下界，不是完整度数；尚不知道候选的人设、权重或游说响应，不能臆测这些属性。优先保留连接数高、可能影响更多已知节点的目标。不要把ID大小解释为影响力。
最多选择10个且不超过剩余扫描名额，返回互不重复的候选ID，按实际扫描顺序排列。没有足够依据改变已有优先级时，保持该顺序。扫描达到计划数量后即进入干预，不会自动扫描其余节点。
只输出JSON：{"targets":[1,2,3],"reason":"简短理由"}
'''


def source_files(variant):
    if variant not in VARIANTS:
        raise ValueError(variant)
    if file_hash(ARCHIVE) != ARCHIVE_SHA:
        raise RuntimeError('Online S44 archive changed')
    with zipfile.ZipFile(ARCHIVE) as archive:
        files = {n: archive.read(n) for n in archive.namelist() if not n.endswith('/')}
    expected = {'config.json', 'starnet_model.py'} | {
        'prompt/' + n + '.txt' for n in ('budget_plan', 'commander_react', 'executor_react',
                                        'reflect', 'scan_batch', 'score_action', 'scout_react')}
    if set(files) != expected:
        raise RuntimeError('Unexpected S44 package members')
    suffix = ''
    if variant == 'Q44':
        suffix = QUEUE_PATCH
    elif variant in ('S42', 'S40'):
        suffix = '\n\nclass ParticipantSquadModel(ParticipantSquadModel):\n    scan_target = ' + variant[1:] + '\n'
    elif variant in ('B44', 'I44'):
        suffix = SCAN_PATCH.replace('ENRICH_VALUE', str(variant == 'I44')).replace('BATCH_VALUE', '2' if variant == 'B44' else '10')
    if suffix:
        files['starnet_model.py'] += suffix.encode('utf-8')
    if variant == 'I44':
        original_prompt = files['prompt/scan_batch.txt'].decode('utf-8')
        files['prompt/scan_batch.txt'] = ('{% if extra.remaining_scans is defined %}\n' + INFO_PROMPT + '\n{% else %}\n' + original_prompt + '\n{% endif %}\n').encode('utf-8')
    compile(files['starnet_model.py'], 'starnet_model.py', 'exec')
    return files


def stage(variant):
    folder = OUT / 'packages' / variant / 'team_submission'
    for name, data in source_files(variant).items():
        path = folder / name
        if path.exists():
            if path.read_bytes() != data:
                raise RuntimeError('Staged source drift: ' + str(path))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as stream:
                stream.write(data)
    return folder


def load(variant):
    name = 'round9_frozen_' + variant
    if name not in sys.modules:
        path = OUT / 'packages' / variant / 'team_submission/starnet_model.py'
        if not path.exists():
            raise RuntimeError('Run round9_study.py prepare first')
        if path.read_bytes() != source_files(variant)['starnet_model.py']:
            raise RuntimeError('Candidate source drift')
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def people():
    return json.loads(source_files('S44')['config.json'])['person']


def specs(pool):
    from round4_seeds import FAMILIES
    result = []
    for i in range(COUNTS[pool]):
        f, rep = i % 6, i // 6
        result.append(dict(case=f'r9-{pool}-{i:03}', pool='r9-' + pool,
            family=FAMILIES[f], scale=(.55, 1., 1.5)[(rep+f)%3],
            persona_mix=(rep//3+f)%3, rep=rep,
            mode=('degree_w', 'degree_r', 'community', 'w_r')[rep%4] if pool=='stress' else 'independent',
            n=50, ood=pool=='stress', generator_version=1))
    return result


def fingerprint(variant):
    names = ('round9_common.py', 'round9_runner.py', 'round9_api.py', 'round4_seeds.py',
             'round4_common.py', 'experiment_round3.py', 'experiment_round5.py',
             'mechanism_round5.py', 'benchmark.py', 'zhipu.py', 'team_submission/starnet_model.py')
    return {'archive': ARCHIVE_SHA,
            'files': {name:file_hash(ROOT/'workspace'/name) for name in names},
            'submission': {name:hashlib.sha256(data).hexdigest() for name,data in source_files(variant).items()}}


def snapshot():
    path = OUT / 'original-workspace.json'
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode('utf-8').split('\0')
    names = [n for n in tracked if n and (ROOT/n).is_file()]
    names += [p.relative_to(ROOT).as_posix() for p in (ROOT/'dist').glob('*.zip')]
    result = {'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'branch':subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
              'initial_status':subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True),
              'files':{n:file_hash(ROOT/n) for n in names}}
    immutable_json(path, result)
    return result


def verify_originals():
    initial = json.loads((OUT/'original-workspace.json').read_text(encoding='utf-8'))
    changed = [n for n,h in initial['files'].items() if not (ROOT/n).exists() or file_hash(ROOT/n) != h]
    head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if changed or head != initial['head']:
        raise RuntimeError('Pre-existing workspace drift: ' + repr(changed))
    return {'status':'PASS','preexisting_files_unchanged':len(initial['files']),
            'head_unchanged':True,'s44_archive_sha256':file_hash(ARCHIVE)}
