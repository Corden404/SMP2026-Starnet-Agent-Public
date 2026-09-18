"""Round 6 experiment identity and frozen D5 loading helpers."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "agent_notes/experiments/round6"
D5_ARCHIVE = ROOT / "dist/team_submission-round5-dsearch-experimental-20260918.zip"
D5_SOURCE = (
    ROOT
    / "agent_notes/experiments/round5/submission-dsearch-20260918/team_submission/starnet_model.py"
)
D5_CONFIG = D5_SOURCE.with_name("config.json")
D5_SHA256 = "30fb036beb7541246f7131c63f1716095aac83187fe17e8385ce5f4747c1e5c5"


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_d5_module():
    if file_hash(D5_ARCHIVE) != D5_SHA256:
        raise RuntimeError("Frozen D5 archive hash mismatch")
    manifest = json.loads(
        (D5_SOURCE.parent.parent / "manifest.json").read_text(encoding="utf-8")
    )
    if file_hash(D5_SOURCE) != manifest["submission_files"]["starnet_model.py"]:
        raise RuntimeError("Frozen D5 staged source hash mismatch")
    name = "round6_frozen_d5"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, D5_SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def people():
    return json.loads(D5_CONFIG.read_text(encoding="utf-8-sig"))["person"]


def fixed_dev_cases():
    """Six graph families x two extreme scales x two persona mixes."""
    families = ("ba", "ws", "er", "sbm", "disconnected", "tree")
    cases = []
    for family_index, family in enumerate(families):
        base = family_index * 9
        for offset in (0, 2, 6, 8):
            cases.append(
                {
                    "case": f"dev-{base + offset:03}",
                    "family": family,
                    "scale": 0.55 if offset < 3 else 1.5,
                    "persona_mix": 0 if offset in (0, 6) else 2,
                }
            )
    return cases


def prepare_manifests() -> None:
    module = load_d5_module()
    with zipfile.ZipFile(D5_ARCHIVE) as archive:
        archive_files = {
            name: hashlib.sha256(archive.read(name)).hexdigest()
            for name in sorted(archive.namelist())
            if not name.endswith("/")
        }
    baseline = {
        "schema": 1,
        "baseline": "D5",
        "archive": str(D5_ARCHIVE),
        "archive_sha256": file_hash(D5_ARCHIVE),
        "staged_source": str(D5_SOURCE),
        "staged_source_sha256": file_hash(D5_SOURCE),
        "archive_files": archive_files,
        "python": platform.python_version(),
        "model_class": module.ParticipantSquadModel.__name__,
    }
    protocol = {
        "schema": 1,
        "baseline": "D5",
        "candidate_order": ["T1", "L1", "U3", "L3"],
        "tail": {
            "budget_max": 6.0,
            "improvement_margin": 0.01,
            "numeric_tolerance": 1e-8,
            "structure_limit": 512,
            "seconds_soft_limit": 3.0,
            "unknown_response": "abstain if deterministic D5 suffix needs it",
        },
        "selection": "fixed before candidate runs; never select from prior wins/losses",
        "old_select54": "development regression only",
        "sealed_pools": ["round4 final108", "round5 final36"],
        "network_calls": 0,
        "glm_calls": 0,
    }
    cases = fixed_dev_cases()
    save(OUT / "baseline-manifest.json", baseline)
    save(OUT / "protocol.json", protocol)
    save(OUT / "dev-cases.json", {"schema": 1, "cases": cases, "count": len(cases)})


if __name__ == "__main__":
    prepare_manifests()
    print(json.dumps({"out": str(OUT), "cases": len(fixed_dev_cases())}, ensure_ascii=False))
