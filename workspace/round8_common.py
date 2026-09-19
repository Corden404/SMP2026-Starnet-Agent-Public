"""Load the exact online S48 archive; never use the working submission as baseline."""
import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'agent_notes/experiments/round8'
ARCHIVE = ROOT / 'dist/team_submission-round7-s48-local-confirmed-20260918.zip'
ARCHIVE_SHA = 'd4bc04b61a8ef5188c1377f2ba405d4eae53a2eb43c683b59d6a87df43cc0860'
STAGE = OUT / 'baseline'

def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)

def prepare_baseline():
    if file_hash(ARCHIVE) != ARCHIVE_SHA:
        raise RuntimeError('S48 archive drift')
    files = {}
    with zipfile.ZipFile(ARCHIVE) as z:
        for name in z.namelist():
            if name.endswith('/'): continue
            if name not in ('config.json', 'starnet_model.py') and not name.startswith('prompt/'):
                raise RuntimeError('Unexpected archive member')
            if '..' in Path(name).parts or Path(name).is_absolute():
                raise RuntimeError('Unsafe archive member')
            data = z.read(name); path = STAGE / name
            files[name] = hashlib.sha256(data).hexdigest()
            if path.exists():
                if path.read_bytes() != data: raise RuntimeError('Staged S48 drift')
            else:
                path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
    save(OUT / 'baseline-manifest.json', dict(archive_sha256=ARCHIVE_SHA, files=files))
    return files

def load_s48():
    name = 'round8_frozen_s48'
    if name in sys.modules: return sys.modules[name]
    path = STAGE / 'starnet_model.py'
    if not path.exists(): raise RuntimeError('Run experiment_round8.py --prepare first')
    with zipfile.ZipFile(ARCHIVE) as z:
        if z.read('starnet_model.py') != path.read_bytes(): raise RuntimeError('S48 source drift')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def people():
    return json.loads((STAGE/'config.json').read_text(encoding='utf-8'))['person']
