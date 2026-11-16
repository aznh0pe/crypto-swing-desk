"""Assemble dashboard/index.html from dashboard/style.css and dashboard/app.js (both are inlined: one self-contained page)."""
import pathlib, re
D = pathlib.Path(__file__).resolve().parent.parent / "dashboard"
html = (D / "index.html").read_text()
css, js = (D / "style.css").read_text().rstrip() + "\n", (D / "app.js").read_text().rstrip() + "\n"
a, b = html.index("<style>") + len("<style>"), html.index("</style>")
html = html[:a] + "\n" + css + html[b:]
a, b = html.rindex("<script>") + len("<script>"), html.rindex("</script>")
html = html[:a] + "\n" + js + html[b:]
(D / "index.html").write_text(html)
print("index.html", len(html), "bytes")
