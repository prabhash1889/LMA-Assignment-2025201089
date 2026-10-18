"""Build the standalone notebook from its readable percent-format source."""
from pathlib import Path
import re
import json
from pprint import pformat
import nbformat

root = Path(__file__).resolve().parents[1]
source = (root / "baseline_vs_reranking.py").read_text()
cells = []
for part in re.split(r"^# %%", source, flags=re.MULTILINE)[1:]:
    header, _, body = part.partition("\n")
    if "[markdown]" in header:
        body = "\n".join(re.sub(r"^# ?", "", line) for line in body.strip().splitlines())
        cells.append(nbformat.v4.new_markdown_cell(body))
    else:
        # Embed annotations so the notebook can run in a fresh folder with only
        # its Python dependencies; it can download the PDF itself if needed.
        if 'queries = json.loads(QUERY_PATH.read_text())' in body:
            annotations = json.loads((root / "data" / "queries.json").read_text())
            literal = pformat(annotations, width=100, sort_dicts=False)
            body = body.replace('queries = json.loads(QUERY_PATH.read_text())',
                '# Frozen annotations embedded for a standalone notebook.\nqueries = ' + literal)
        cells.append(nbformat.v4.new_code_cell(body.strip()))
notebook = nbformat.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
})
target = root / "Baseline_vs_Reranking.ipynb"
nbformat.write(notebook, target)
print(f"Wrote {target.name}: {len(cells)} cells; all implementation code is embedded.")
