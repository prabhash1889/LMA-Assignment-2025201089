"""Render the editable report to a local review page and a standalone PDF."""
from pathlib import Path
import re
import shutil
import unicodedata
import pymupdf
from markdown_it import MarkdownIt

root = Path(__file__).resolve().parents[1]
markdown = (root / "REPORT.md").read_text()
content = MarkdownIt("commonmark").enable("table").render(markdown)
content = content.replace("<table>", '<table class="table table-zebra">')
# Preserve the student's added name, roll number and course as compact metadata.
prefix, separator, remainder = content.partition("<h2>1. Architecture comparison</h2>")
authors = re.findall(r"<h2>(.*?)</h2>", prefix)
if authors:
    prefix = re.sub(r"<h2>.*?</h2>\s*", "", prefix)
    end_title = prefix.index("</h1>") + len("</h1>")
    prefix = prefix[:end_title] + '<p class="author"><strong>' + " · ".join(authors) + '</strong></p>' + prefix[end_title:]
content = prefix + separator + remainder
css = """body { font-family: sans-serif; font-size: 9.6pt; line-height: 1.3; color: #1e293b; }
h1 { font-size: 21pt; color: #0f4c5c; margin: 0 0 10pt; }
h2 { font-size: 13pt; color: #0f4c5c; margin-top: 15pt; margin-bottom: 6pt; }
h3 { font-size: 10.5pt; margin-top: 10pt; margin-bottom: 4pt; }
p { margin: 0 0 7pt; }
table { border-collapse: collapse; width: 100%; margin: 9pt 0; font-size: 8pt; }
th { color: #123b47; text-align: left; border-top: 0.8pt solid #9eb4bc; }
th, td { padding: 5pt; border-bottom: 0.5pt solid #d7e0e4; vertical-align: top; }
code { font-family: monospace; font-size: 8pt; }
a { color: #0f766e; }
.author { font-size: 9.5pt; margin-bottom: 8pt; }
img { width: 100%; height: auto; }
em { font-size: 8.2pt; }
"""
folder = root / ".lavish"
folder.mkdir(exist_ok=True)
html = folder / "report.html"
shutil.copy2(root / "results/comparison.png", folder / "comparison.png")
page_css = css + """
* { box-sizing: border-box; }
html, body { background: #f0f3f5; }
body { margin: 0; font-size: 15px; line-height: 1.65; }
main { max-width: 960px; margin: 32px auto; padding: 48px; background: white; border: 1px solid #d7e0e4; border-radius: 10px; min-width: 0; }
table { font-size: 13px; table-layout: fixed; }
td, th, p, code { overflow-wrap: anywhere; }
.toolbar { display: flex; justify-content: space-between; gap: 12px; align-items: center; margin-bottom: 28px; }
.toolbar > * { min-width: 0; }
@media (max-width: 640px) { main { margin: 0; border: 0; border-radius: 0; padding: 22px; } th, td { padding: 5px; font-size: 11px; } }
@media print { html, body { background: white; } main { margin: 0; border: 0; padding: 0; max-width: none; } .toolbar { display: none; } h2, h3 { break-after: avoid; } tr { break-inside: avoid; } }
"""
html.write_text('''<!doctype html><html lang="en" data-theme="corporate"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RAG: Baseline Retrieval vs. Re-ranking</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/daisyui@5.5.19/daisyui.css">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/daisyui@5.5.19/themes.css">
<script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4.2.4/dist/index.global.js"></script>
<style>''' + page_css + '''</style></head><body><main>
<div class="toolbar"><span class="badge badge-outline">Experiment · 18 October 2026</span>
<button class="btn btn-sm btn-outline" onclick="window.print()">Print / save PDF</button></div>
''' + content.replace('src="results/comparison.png"', 'src="comparison.png"') + "</main></body></html>")

# Use the installed PDF library directly, without a browser or a local server.
pdf_content = content.replace('<h3>Scientific research: distractors and negation</h3>',
    '<h3 style="page-break-before: always">Scientific research: distractors and negation</h3>')
pdf_content = pdf_content.replace('<h2>4. Why not cross-encode the entire collection?</h2>',
    '<h2 style="page-break-before: always">4. Why not cross-encode the entire collection?</h2>')
story = pymupdf.Story(html=pdf_content, user_css=css, em=10,
                      archive=pymupdf.Archive(str(root)))
pdf_path = root / "REPORT.pdf"
writer = pymupdf.DocumentWriter(str(pdf_path))
page_rect = pymupdf.paper_rect("a4")
body_rect = pymupdf.Rect(42, 40, page_rect.width - 42, page_rect.height - 42)
more = True
while more:
    device = writer.begin_page(page_rect)
    more, _ = story.place(body_rect)
    story.draw(device)
    writer.end_page()
writer.close()
with pymupdf.open(pdf_path) as document:
    for number, page in enumerate(document, 1):
        page.insert_text((42, page.rect.height - 23), f"RAG experiment | 18 October 2026", fontsize=8, color=(.35,.4,.45))
        page.insert_text((page.rect.width - 70, page.rect.height - 23), f"{number} / {len(document)}", fontsize=8, color=(.35,.4,.45))
    document.set_metadata({"title": "RAG: Baseline Retrieval vs. Re-ranking", "subject": "Measured comparison of dense retrieval and cross-encoder re-ranking"})
    document.saveIncr()
    text = unicodedata.normalize("NFKC", "\n".join(p.get_text() for p in document))
    for heading in ["Architecture comparison", "Failure queries", "Latency profiling", "cross-encode"]:
        assert heading in text, f"Missing report section: {heading}"
    assert any(page.get_images() for page in document), "Comparison plot is missing from the PDF."
    assert "Figure 1." in text, "Plot caption is missing from the PDF."
    print(f"Created REPORT.pdf: {len(document)} pages; REPORT.md is the editable source.")
print(f"Review page: {html.relative_to(root)}")
