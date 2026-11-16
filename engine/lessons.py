"""The lessons registry: conclusions that outlive any one idea (a measured ceiling, a rule that did not help,
a trap, a state that works). Never trimmed; every research session starts by reading it.

  python -m engine.lessons add --state S --out O --topic T --lesson "..." --evidence "..."
"""
from __future__ import annotations

import argparse
import json
import pathlib

from .calibrate import now_iso
from .portfolio import load_state

TOPICS = ["strategy design", "grading", "portfolio", "universe", "market conditions", "execution", "operations", "research method"]


def add(state_dir, out_dir, topic: str, lesson: str, evidence: str) -> dict:
    st = load_state(state_dir)
    doc = dict(st["learning"].get("lessons") or {"id": "lessons", "items": []})
    items = list(doc.get("items") or [])
    if any(i.get("lesson", "").strip().lower() == lesson.strip().lower() for i in items):
        return {"added": False, "why": "already recorded", "count": len(items)}
    items.append({"n": len(items) + 1, "date": now_iso()[:10], "topic": topic if topic in TOPICS else "research method", "lesson": lesson.strip(), "evidence": evidence.strip()})
    doc.update({"id": "lessons", "items": items, "updated_at": now_iso()})
    out = pathlib.Path(out_dir)
    p = out / "docs" / "learning" / "lessons.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc))
    (out / "writes.json").write_text(json.dumps([{"op": "set", "collection": "learning", "doc_id": "lessons", "file_path": str(p.resolve())}], indent=1))
    return {"added": True, "count": len(items)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["add"])
    ap.add_argument("--state", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--topic", default="research method")
    ap.add_argument("--lesson", required=True)
    ap.add_argument("--evidence", default="")
    a = ap.parse_args()
    print(json.dumps(add(a.state, a.out, a.topic, a.lesson, a.evidence)))
