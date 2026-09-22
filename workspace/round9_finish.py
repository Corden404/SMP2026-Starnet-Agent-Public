"""Finish the already frozen round9 study in sequence; never tune or upload.

Run only after the existing Q44 confirmation batch has finished. Each phase is
journaled, existing successful records are reused by their original runners,
and failed phases require explicit inspection instead of automatic retries.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time

from round9_common import ROOT, OUT, fingerprint, save, verify_originals


def phases():
    return [
        ('stress', ['round9_study.py', 'run', '--pool', 'stress', '--variants', 'S44', 'B44', 'Q44', '--workers', '2'], True),
        ('full-glm', ['round9_study.py', 'run', '--pool', 'glm-check', '--variants', 'S44', 'B44', 'Q44', '--mode', 'glm', '--repeats', '2', '--workers', '2'], True),
        ('q44-package', ['round9_delivery.py', 'accept', '--variant', 'Q44'], True),
        ('q44-sizes', ['round9_delivery.py', 'accept', '--variant', 'Q44', '--sizes'], True),
        ('b44-package', ['round9_delivery.py', 'accept', '--variant', 'B44'], True),
        ('b44-sizes', ['round9_delivery.py', 'accept', '--variant', 'B44', '--sizes'], True),
        ('q44-sandbox', ['round9_delivery.py', 'accept', '--variant', 'Q44', '--remote'], False),
        ('b44-sandbox', ['round9_delivery.py', 'accept', '--variant', 'B44', '--remote'], False),
        ('report', ['round9_report.py'], True),
    ]


def preflight():
    verify_originals()
    freeze = json.loads((OUT / 'candidate-freeze.json').read_text(encoding='utf-8'))
    if set(freeze['variants']) != {'B44', 'Q44'}:
        raise RuntimeError('This completion path is only for the existing two-candidate freeze')
    for variant in freeze['variants']:
        if fingerprint(variant) != freeze['code'][variant]:
            raise RuntimeError('Frozen policy changed')
    # Check completeness, not interim scores. Do not start a duplicate batch.
    directory = OUT / 'runs/r9-confirm-fast'
    for variant in ('S44', 'B44', 'Q44'):
        records = list(directory.glob('*-' + variant + '-r0.json'))
        if len(records) != 96 or any(json.loads(p.read_text(encoding='utf-8'))['status'] != 'PASS' for p in records):
            raise RuntimeError('Existing confirmation is not complete: ' + variant)


def main():
    raise SystemExit('Retired coordinator: do not restart rejected or duplicate studies. See round9 completion.json and round9_closeout.py.')
    preflight()
    directory = OUT / 'completion-phases'
    directory.mkdir(parents=True, exist_ok=True)
    outcomes = []
    for name, arguments, required in phases():
        marker = directory / (name + '.json')
        command = [sys.executable, str(ROOT / 'workspace' / arguments[0]), *arguments[1:]]
        if marker.exists():
            old = json.loads(marker.read_text(encoding='utf-8'))
            if old.get('command') != command or old.get('status') != 'PASS':
                raise RuntimeError('Inspect previous phase before recovery: ' + name)
            outcomes.append(old)
            continue
        row = dict(phase=name, command=command, required=required, status='STARTED', started=time.time())
        save(marker, row)
        print(json.dumps(dict(phase=name, status='STARTED')), flush=True)
        # Separate files preserve the original experiment and all service journals.
        with (directory / (name + '.log')).open('x', encoding='utf-8') as log:
            completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
        row.update(status='PASS' if completed.returncode == 0 else 'FAILED',
                   exit_code=completed.returncode, ended=time.time())
        save(marker, row)
        outcomes.append(row)
        print(json.dumps(dict(phase=name, status=row['status'], exit_code=completed.returncode)), flush=True)
        verify_originals()
        if completed.returncode and required:
            raise RuntimeError('Required phase stopped; inspect logs without blind retry: ' + name)
    save(directory / 'summary.json', dict(
        status='PASS' if all(r['status'] == 'PASS' for r in outcomes) else 'SERVICE_ACCEPTANCE_INCOMPLETE',
        phases=outcomes, originals=verify_originals(), no_upload=True, no_git_commit=True))


if __name__ == '__main__':
    main()
