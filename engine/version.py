"""Which engine is running: a fingerprint of its code, so every scan records what produced it."""
from __future__ import annotations

import hashlib
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent


def files(root: pathlib.Path = ROOT) -> list[str]:
    out = sorted(p.relative_to(root).as_posix() for p in (root / "engine").glob("*.py"))
    out += sorted(p.relative_to(root).as_posix() for p in (root / "engine").glob("*.json") if p.name != "MANIFEST.json")
    return out


def fingerprint(root: pathlib.Path = ROOT) -> str:
    h = hashlib.sha256()
    for rel in files(root):
        h.update(rel.encode() + b"\0" + (root / rel).read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return h.hexdigest()[:12]


def info() -> dict:
    try:
        return {"fingerprint": fingerprint(), "source": "repo" if (ROOT / ".git").exists() else "dashboard copy"}
    except Exception as e:
        return {"fingerprint": None, "error": str(e)[:120]}


if __name__ == "__main__":
    print(json.dumps(info()))
