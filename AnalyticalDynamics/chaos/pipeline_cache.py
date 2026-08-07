"""Content-hash based freshness for run_all.py's pipeline stages.

Each stage writes one JSON file wrapped in a uniform envelope:

    {"stage": ..., "generated_at": ..., "hash": ..., "hash_of": ..., "payload": ...}

`hash` is a sha256 of the stage's declared inputs (its own static config plus
any upstream stage hashes it depends on) -- not of the output, and not a file
mtime. Fits here are deterministic (random_state is fixed throughout), so two
runs with the same inputs produce the same hash and the same payload; a stage
is skipped only when its previously recorded hash matches what it would compute
now. Editing a sweep's config list, or any upstream stage actually changing its
output hash, invalidates exactly the stages downstream of that edit -- nothing
else re-runs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def stage_hash(*parts) -> str:
    """sha256 of the canonical JSON of `parts`, as a short hex string."""
    blob = json.dumps(parts, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def load_if_fresh(path: Path, hash_: str) -> dict | None:
    """The stage's payload if `path` exists and was written with this hash."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if doc.get("hash") != hash_:
        return None
    return doc.get("payload")


def load_payload(path: Path):
    """Unconditional read of a stage's payload, for downstream stages that
    only need the data and check freshness through the recorded hash instead.
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))["payload"]


def write_stage(path: Path, stage: str, hash_: str, hash_of, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "stage": stage,
        "hash": hash_,
        "hash_of": hash_of,
        "payload": payload,
    }
    path.write_text(json.dumps(doc, indent=2, default=_default), encoding="utf-8")


def _default(o):
    """json.dumps fallback for numpy scalars/arrays that slip into a payload."""
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "item"):
        return o.item()
    raise TypeError(f"not JSON serialisable: {type(o)}")
