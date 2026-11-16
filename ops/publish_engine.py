"""Lists the engine files to publish with the dashboard (the copy scheduled runs read) and writes engine/MANIFEST.json.
Python files are served with .txt added. After any engine change: run this, then republish the dashboard with the
files it prints (index.html plus every "as" path)."""
import json, pathlib, shutil, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from engine import version

ROOT = pathlib.Path(__file__).resolve().parent.parent
stage = ROOT / "work" / "publish"
shutil.rmtree(stage, ignore_errors=True)
(stage / "engine").mkdir(parents=True)
files = version.files(ROOT)
pub = []
for rel in files:
    dst = rel + ".txt" if rel.endswith(".py") else rel
    shutil.copy(ROOT / rel, stage / dst)
    pub.append(dst)
man = {"fingerprint": version.fingerprint(ROOT), "files": files, "published": pub,
       "about": "Crypto Swing Desk engine, identical to engine/ in the GitHub repo. Scheduled runs read the \"published\" files with the Artifact tool, drop the .txt from the Python files and run them."}
(stage / "engine" / "MANIFEST.json").write_text(json.dumps(man, indent=1))
(ROOT / "ops" / "engine_manifest.json").write_text(json.dumps(man, indent=1))
out = [{"as": "index.html", "path": str(ROOT / "dashboard" / "index.html")}] + [{"as": p, "path": str(stage / p)} for p in pub + ["engine/MANIFEST.json"]]
print(json.dumps(out))
