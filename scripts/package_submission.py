"""Validate the frozen submission and build a deterministic nine-file ZIP."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / 'workspace' / 'team_submission'
MANIFEST = ROOT / 'submission_manifest.json'


def verified_files():
    manifest = json.loads(MANIFEST.read_text(encoding='utf8'))
    expected = manifest['files']
    actual = {path.relative_to(SUBMISSION).as_posix(): path
              for path in SUBMISSION.rglob('*') if path.is_file()
              and '__pycache__' not in path.parts}
    if set(actual) != set(expected):
        raise ValueError('Submission file layout differs from the frozen manifest')
    files = {name: path.read_bytes() for name, path in actual.items()}
    for name, data in files.items():
        if hashlib.sha256(data).hexdigest() != expected[name]:
            raise ValueError('Frozen source identity mismatch: ' + name)
    return files


def build(output):
    files = verified_files()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, 'x', compression=ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'dist' / 'ip44.zip')
    args = parser.parse_args()
    print(build(args.output))


if __name__ == '__main__':
    main()
