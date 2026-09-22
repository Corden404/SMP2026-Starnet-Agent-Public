"""Quarantine the entire duplicate-run stress pool after both owners stopped.

No results are selected by score and no frozen policies are modified. Run once,
then start the original complete stress cohort through the locked study CLI.
"""
from __future__ import annotations
import collections
import json
from pathlib import Path
import time
import psutil
from round9_common import OUT, ROOT, file_hash, fingerprint, immutable_json, save, verify_originals
from round9_lock import ExclusiveBatch

OLD_PIDS = (55888,68988,70464,70772,70616,73332,74060)
SOURCE_PATHS = ('runs/r9-stress-fast',
                'batches/stress-fast-S44-B44-Q44.json',
                'summaries/stress-fast.json',
                'completion-phases')


def main():
    with ExclusiveBatch(OUT/'execution.lock'):
        target=OUT/'quarantine/duplicate-stress-01'
        audit_path=OUT/'duplicate-stress-recovery.json'
        if target.exists() or audit_path.exists():
            raise RuntimeError('Recovery is one-shot; inspect the existing quarantine instead of repeating it')
        alive=[pid for pid in OLD_PIDS if psutil.pid_exists(pid)]
        if alive:
            raise RuntimeError('An old stress process still exists; inspect ownership before proceeding')
        original=verify_originals()
        frozen=json.loads((OUT/'candidate-freeze.json').read_text(encoding='utf-8'))
        for variant in frozen['variants']:
            if fingerprint(variant)!=frozen['code'][variant]:
                raise RuntimeError('Frozen participant or evaluator changed')
        paid_folder=OUT/'runs/r9-glm-check-glm-glm-4.5-air'
        if paid_folder.exists() and any(paid_folder.iterdir()):
            raise RuntimeError('Unexpected full-GLM start; inspect before recovering stress')
        rows=[json.loads(p.read_text(encoding='utf-8')) for p in (OUT/'runs/r9-stress-fast').glob('r9-*.json')]
        files={}
        present=[]
        for relative in SOURCE_PATHS:
            path=OUT/relative
            if not path.exists():continue
            present.append(relative)
            for p in (path.rglob('*') if path.is_dir() else [path]):
                if p.is_file():files[p.relative_to(OUT).as_posix()]=file_hash(p)
        if not rows or 'runs/r9-stress-fast' not in present:
            raise RuntimeError('Expected overlapping stress outputs are absent')
        audit=dict(status='PREPARED',reason='Two independent workflow sessions started the identical stress batch at the same time; both jobs were stopped before the full-GLM stage',
            time=time.time(),jobs=['8c13e4ac-57c5-49b8-80a6-d77f260a6b93','b6507f87-b4a9-4cd6-8f3a-7b42e1540133'],
            previous_processes_absent=True,originals=original,
            record_statuses=dict(collections.Counter(r['variant']+':'+r['status'] for r in rows)),
            quarantine=str(target),files=files,moved_paths=present,
            selection_by_score=False,all_stress_results_invalidated=True,paid_calls_in_duplicate_stress=0,
            unchanged_pools=['dev','confirm','glm-dev'],
            next_step='Re-run all original 24 stress graphs for S44 B44 Q44, not just incomplete records, under the OS-exclusive lock',
            execution_guard=dict(old_study_sha256='9d614d70a77ac558ddf6dd1274287b32e10ea464f142b154c0d2f023654c3be3',
                new_study_sha256=file_hash(ROOT/'workspace/round9_study.py'),
                lock_sha256=file_hash(ROOT/'workspace/round9_lock.py'),
                policies_seeds_and_API_parameters_unchanged=True))
        immutable_json(OUT/'duplicate-stress-recovery-preflight.json',audit)
        target.mkdir(parents=True)
        for relative in present:
            source=OUT/relative; destination=target/relative
            destination.parent.mkdir(parents=True,exist_ok=True)
            source.rename(destination)
        for relative,h in files.items():
            if file_hash(target/relative)!=h:
                raise RuntimeError('Quarantine hash mismatch')
        audit.update(status='QUARANTINED',quarantined_files=len(files))
        save(audit_path,audit)
        print(json.dumps({k:audit[k] for k in ('status','quarantined_files','record_statuses','selection_by_score','paid_calls_in_duplicate_stress')}))


if __name__=='__main__':main()
