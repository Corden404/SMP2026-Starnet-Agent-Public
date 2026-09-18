"""Experiment identity and local-only tracing. Never shipped in the competition ZIP."""
import copy
import hashlib
import importlib.metadata
import json
import platform
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'agent_notes/experiments/round4'
SOURCE = ROOT / 'workspace/team_submission'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_files(folder=SOURCE):
    return {p.relative_to(folder).as_posix(): file_hash(p) for p in sorted(folder.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def freeze_baseline():
    import zipfile
    archive = ROOT/'dist/team_submission-round3-roi-tight-candidate.zip'
    expected = 'a19b873d75d22ccc436d3284e5ce7a1989802f18feea50b7ac519bc5c547b735'
    if file_hash(archive) != expected:
        raise RuntimeError('Baseline archive hash mismatch')
    folder = OUT/'baseline'
    with zipfile.ZipFile(archive) as z:
        for entry in z.infolist():
            if entry.is_dir():
                continue
            target = (folder/entry.filename).resolve()
            if not target.is_relative_to(folder.resolve()):
                raise ValueError('Unsafe archive entry')
            content = z.read(entry)
            if target.exists():
                if target.read_bytes() != content:
                    raise RuntimeError('Frozen baseline drift')
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
    manifest = {'archive_sha256': expected, 'files': source_files(folder)}
    save(OUT/'baseline-manifest.json', manifest)
    return manifest


def identity(seed, runtime, model_config, batch, environment, repeat=0, attempt=0):
    endpoint = model_config.get('endpoint', '')
    url = urlsplit(endpoint)
    if url.username or url.password or url.query or url.fragment:
        raise ValueError('Credentials/query not allowed in recorded endpoint')
    forbidden = {'api_key', 'authorization', 'headers', 'key', 'token'}
    def safe(value):
        if isinstance(value, dict):
            if any(str(k).lower() in forbidden for k in value):
                raise ValueError('Secret field in experiment definition')
            for x in value.values(): safe(x)
        elif isinstance(value, list):
            for x in value: safe(x)
    safe(model_config)
    packages = {}
    for name in ('networkx', 'casevo', 'chromadb', 'requests'):
        try: packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name] = 'unknown'
    dependencies = [p for p in (ROOT/'workspace').glob('*.py')]
    definition = {'schema': 4, 'seed_sha256': digest(seed), 'runtime': runtime,
                  'model_config': model_config, 'submission_files': source_files(),
                  'harness_files': {p.name: file_hash(p) for p in sorted(dependencies)},
                  'python': platform.python_version(), 'packages': packages,
                  'environment': environment, 'batch': batch}
    eid = digest(definition)
    return {'experiment_id': eid, 'run_id': digest([eid, repeat, attempt]),
            'definition': definition, 'repeat': repeat, 'attempt': attempt}


def cached(path, expected):
    path = Path(path)
    if not path.exists(): return None
    result = json.loads(path.read_text(encoding='utf-8'))
    if result.get('identity') != expected:
        raise RuntimeError('Identity mismatch or legacy cache: choose a new batch/output')
    if result.get('status') != 'completed' or 'final_score' not in result:
        raise RuntimeError('Incomplete/failed run cannot be cached: use a new attempt')
    return result


class TraceLLM:
    """Capture returned text before CaseVO parsing, without adding requests."""
    def __init__(self, delegate):
        self.delegate = delegate
        self.requests = []

    def send_message(self, prompt, json_flag=False):
        event = {'number': len(self.requests)+1, 'kind': 'chat', 'prompt': prompt,
                 'json_flag': json_flag, 'started': time.time()}
        self.requests.append(event)
        try:
            result = self.delegate.send_message(prompt, json_flag=True) if json_flag else self.delegate.send_message(prompt)
            event['response'] = result
            event['status'] = 'returned'
            return result
        except BaseException as exc:
            event.update(status='failed', error_type=type(exc).__name__)
            raise
        finally:
            event['seconds'] = time.time()-event['started']

    def send_embedding(self, texts):
        event = {'number': len(self.requests)+1, 'kind': 'embedding', 'items': len(texts)}
        self.requests.append(event)
        try:
            result = self.delegate.send_embedding(texts)
            event['status'] = 'returned'
            return result
        except BaseException as exc:
            event.update(status='failed', error_type=type(exc).__name__)
            raise

    def get_lang_embedding(self): return self.delegate.get_lang_embedding()


def attach_traces(model, llm):
    """Observe chain inputs/outputs; preserve original methods and fallback policy."""
    events = []
    for role in ('commander', 'scout', 'executor'):
        agent = getattr(model, role, None)
        if agent is None: continue
        original = agent._run_chain
        def traced(name, data, original=original, role=role, agent=agent):
            start = len(llm.requests)
            event = {'role': role, 'chain': name, 'input': copy.deepcopy(data),
                     'first_request': start+1}
            events.append(event)
            output = original(name, data)
            event['output'] = copy.deepcopy(output)
            event['last_request'] = len(llm.requests)
            detail = getattr(agent, 'last_chain_trace', {})
            if detail.get('status') == 'exception':
                event['decision_origin'] = 'chain_exception_fallback'
                event['exception_type'] = detail.get('exception_type')
            elif not isinstance(output, dict):
                event['decision_origin'] = 'parse_fallback'
            elif name in ('score', 'plan'):
                candidates = json.loads(data.get('candidates', data.get('plans', '[]')))
                index = output.get('index')
                event['candidate_count'] = len(candidates)
                event['decision_origin'] = ('model_valid' if type(index) is int and 0 <= index < len(candidates)
                                            else 'invalid_index_fallback')
            elif name == 'batch':
                targets = output.get('targets')
                valid = data['unknown_ids']
                event['decision_origin'] = ('model_valid' if isinstance(targets, list) and
                    any(type(i) is int and i in valid for i in targets) else 'invalid_targets_fallback')
            else: event['decision_origin'] = 'unclassified_legacy_chain'
            return output
        agent._run_chain = traced
    return events
