"""Inline report/data.json into report.src.html -> report/report.html (run flm.report_data first)."""

from pathlib import Path

h = Path(__file__).parent
src = (h / "report.src.html").read_text(encoding="utf-8")
data = (h / "data.json").read_text(encoding="utf-8")
(h / "report.html").write_text(src.replace("/*DATA*/", data), encoding="utf-8")
print("built report.html", len(src) + len(data), "bytes")
