"""Read/audit closeout and network-disabled tests; never starts a strategy or API game."""
from __future__ import annotations
import argparse
import collections
import contextlib
import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import round11_experiment as r11
import round12_experiment as r12
from round9_lock import ExclusiveBatch

ROOT = r11.ROOT
OUT = ROOT / 'agent_notes/experiments/round11-12-closeout'


def tests():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'tests-final.log'
    with contextlib.ExitStack() as stack, path.open('w', encoding='utf-8') as stream:
        for name in ('socket.socket.connect', 'socket.socket.connect_ex', 'socket.create_connection'):
            stack.enter_context(patch(name, r11.denied))
        stack.enter_context(contextlib.redirect_stdout(stream))
        stack.enter_context(contextlib.redirect_stderr(stream))
        suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'), pattern='test_*.py')
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    record = dict(status='PASS' if result.wasSuccessful() else 'FAIL', tests=result.testsRun,
                  failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
                  network_disabled=True, log=path.relative_to(ROOT).as_posix(),
                  test_sources={p.relative_to(ROOT).as_posix(): r11.sha(p) for p in sorted((ROOT / 'tests').glob('test_*.py'))},
                  runner_sha256=r11.sha(__file__), originals=r11.preserved())
    r11.save(OUT / 'tests-final.json', record)
    print(json.dumps({k: v for k, v in record.items() if k != 'test_sources'}), flush=True)
    if not result.wasSuccessful():
        raise SystemExit(1)
    return record


def actions(row):
    return collections.Counter((a['action'], tuple(a['args'])) for a in row['actions'] if a['action'] != 'scan_node')


def scan_mechanism():
    details = []
    folder = r12.OUT / 'runs/dev-fast'
    for spec in r12.specs('dev'):
        b, c = [r11.read(folder / (spec['case'] + '-' + v + '.json')) for v in r12.VARIANTS]
        if b['status'] != 'PASS' or c['status'] != 'PASS':
            raise RuntimeError('Incomplete mechanism pair')
        extra = c['action_counts']['scan_node'] - 44
        initial = [a for a in c['actions'] if a['action'] == 'scan_node'][:44]
        baseline = [a for a in b['actions'] if a['action'] == 'scan_node'][:44]
        if initial != baseline:
            raise RuntimeError('Initial scan prefix differs')
        added, removed = actions(c) - actions(b), actions(b) - actions(c)
        purchase = next((e for e in c['active_records'] if e.get('reason') == 'purchase'), None)
        if not extra and actions(c) != actions(b):
            same_multiset = False
        else:
            same_multiset = True
        details.append(dict(case=spec['case'], family=spec['family'], extra_scans=extra,
            delta=c['final_score'] - b['final_score'], purchase=copy.deepcopy(purchase),
            nontriggered_same_multiset=same_multiset,
            additional_targets=[a['args'][0] for a in c['actions'] if a['action'] == 'scan_node'][44:],
            added=[dict(action=k[0], args=k[1], count=v) for k, v in added.items()],
            removed=[dict(action=k[0], args=k[1], count=v) for k, v in removed.items()]))
    result = dict(scope='Fixed24 development, descriptive mechanism only', records=details,
                  all_initial_scan_prefixes_identical=True,
                  nontriggered_multiset_changes=sum(not r['nontriggered_same_multiset'] for r in details))
    r11.save(r12.OUT / 'mechanism.json', result)
    return result


def evidence():
    r11.check(); r12.check()
    with ExclusiveBatch(r11.OUT / 'execution.lock'):
        first_audit = r11.audit(); first_report = r11.report()
    with ExclusiveBatch(r12.OUT / 'execution.lock'):
        second_audit = r12.audit(); second_report = r12.report()
    if first_audit['incomplete_records'] or second_audit['incomplete']:
        raise RuntimeError('Incomplete runs remain; inspect before closeout')
    mechanism = scan_mechanism()
    journal_counts = collections.Counter()
    for folder in (r11.OUT, r12.OUT):
        for path in folder.rglob('*.jsonl'):
            for line in path.read_text(encoding='utf-8').splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get('event') == 'request':
                    journal_counts[row.get('data', {}).get('model', 'unknown')] += 1
    record = dict(status='AUDITED', tests=r11.read(OUT / 'tests-final.json'),
        round11=first_report['status'], round12=second_report['status'],
        audited_games=first_audit['games'] + second_audit['games'],
        max_forward_error=max(first_audit['max_forward_error'], second_audit['max_forward_error']),
        actual_http_requests=dict(journal_counts), api_order=list(r11.MODELS),
        nontriggered_scan_multiset_changes=mechanism['nontriggered_multiset_changes'],
        originals=r11.preserved(), source_files={p: r11.sha(ROOT / p) for p in
            ('workspace/round11_policy.py', 'workspace/round11_api.py', 'workspace/round11_experiment.py',
             'workspace/round12_policy.py', 'workspace/round12_experiment.py',
             'tests/test_round11.py', 'tests/test_round12.py', 'workspace/round11_12_closeout.py')},
        caveat='Strategy gates use prospective development/confirmation rules. Integration games and old diagnostic graphs are not independent confirmation.')
    if record['tests']['status'] != 'PASS':
        raise RuntimeError('Offline regression has not passed')
    r11.save(OUT / 'final-audit.json', record)
    print(json.dumps({k: v for k, v in record.items() if k not in ('tests', 'source_files')}), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['tests', 'evidence'])
    args = parser.parse_args()
    with ExclusiveBatch(OUT / 'execution.lock'):
        tests() if args.stage == 'tests' else evidence()


if __name__ == '__main__':
    main()
