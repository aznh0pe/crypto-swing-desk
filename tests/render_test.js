// Headless render test: loads the dashboard with a fake database and visits every tab.
// usage: node tests/render_test.js <index.html> <docs_dir>   (docs_dir holds <collection>/<id>.json files)
const fs = require("fs"), path = require("path");
const { JSDOM, VirtualConsole } = require(process.env.JSDOM || "jsdom");
const [,, page, docsDir] = process.argv;
const cols = {};
if (docsDir) for (const c of fs.readdirSync(docsDir)) { cols[c] = {}; for (const f of fs.readdirSync(path.join(docsDir, c))) cols[c][f.replace(/\.json$/, "")] = JSON.parse(fs.readFileSync(path.join(docsDir, c, f), "utf8")); }
const snapDoc = (c, id) => ({ exists: !!(cols[c] && cols[c][id]), id, data: () => cols[c][id] });
const subs = [];
const q = (c) => ({ orderBy() { return this; }, limit() { return this; }, where() { return this; },
  onSnapshot(cb) { const fire = () => cb({ docs: Object.keys(cols[c] || {}).map((id) => snapDoc(c, id)), metadata: {} }); subs.push([c, fire]); setTimeout(fire, 5); return () => {}; }, doc: (id) => db.doc(c + "/" + id) });
const emit = (c) => subs.filter(([x]) => x === c).forEach(([, f]) => setTimeout(f, 1));
const writes = [];
const db = { collection: q, doc: (p) => { const [c, id] = p.split("/"); return { onSnapshot(cb) { setTimeout(() => cb(snapDoc(c, id)), 5); return () => {}; },
  set: async (d) => { writes.push(["set", p]); (cols[c] = cols[c] || {})[id] = d; emit(c); }, update: async (d) => { writes.push(["update", p]); Object.assign(cols[c][id], d); emit(c); } }; } };
const errors = [];
const vc = new VirtualConsole(); vc.on("jsdomError", (e) => errors.push(String(e.detail || e))); vc.on("error", (...a) => errors.push(a.join(" ")));
const dom = new JSDOM(fs.readFileSync(page, "utf8"), { runScripts: "dangerously", pretendToBeVisual: true, virtualConsole: vc,
  beforeParse(w) { w.claude = { use: async (n) => (n === "db" ? db : null) }; w.scrollTo = () => {}; } });
const w = dom.window, d = w.document;
setTimeout(async () => {
  const out = {};
  for (const t of ["overview", "alerts", "log", "funds", "scans", "lab"]) {
    d.querySelector(`[data-tab="${t}"]`).dispatchEvent(new w.MouseEvent("click", { bubbles: true }));
    await new Promise((r) => setTimeout(r, 30));
    const txt = d.querySelector("#main").textContent;
    out[t] = txt.length; if (/undefined|NaN|\[object/.test(txt)) errors.push(`${t}: suspicious text: ` + (txt.match(/.{0,60}(undefined|NaN|\[object).{0,40}/) || [""])[0]);
  }
  // exercise the fill -> sale flow if there is a live ticket
  d.querySelector(`[data-tab="alerts"]`).dispatchEvent(new w.MouseEvent("click", { bubbles: true }));
  const fill = d.querySelector('[data-act="fill"]');
  if (fill) { fill.dispatchEvent(new w.MouseEvent("click", { bubbles: true })); await new Promise((r) => setTimeout(r, 20));
    d.querySelector("#modalForm").dispatchEvent(new w.Event("submit", { bubbles: true, cancelable: true })); await new Promise((r) => setTimeout(r, 1800));
    out.pos_text = (d.querySelector(".pos-card") || {}).textContent; const close = d.querySelector('[data-act="close"]'); out.position_card = !!close;
    if (close) { close.dispatchEvent(new w.MouseEvent("click", { bubbles: true })); await new Promise((r) => setTimeout(r, 20));
      d.querySelector("#cPx").value = String(Number(d.querySelector("#cPx").value || 0) || 100);
      d.querySelector("#modalForm").dispatchEvent(new w.Event("submit", { bubbles: true, cancelable: true })); await new Promise((r) => setTimeout(r, 1800));
      out.log = (() => { d.querySelector('[data-tab="log"]').dispatchEvent(new w.MouseEvent("click", { bubbles: true })); return d.querySelector("#main").textContent.slice(0, 400); })();
      out.modal_error = d.querySelector("#modalErr") ? d.querySelector("#modalErr").textContent : ""; } }
  out.writes = writes; out.chips = d.querySelector("#chips").textContent.replace(/\s+/g, " ").trim();
  console.log(JSON.stringify({ out, errors }, null, 1));
  process.exit(errors.length ? 1 : 0);
}, 150);
