"""Validate exported evidence independently of the retrieval implementation.

Use --require-answers before final submission. This validates experimental
consistency, not semantic correctness of LLM statements; review those yourself.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import nbformat
import numpy as np

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--require-answers", action="store_true")
args = parser.parse_args()

def read(name):
    return json.loads((root / name).read_text())

provenance = read("data/provenance.json")
assert hashlib.sha256((root / "data/gdpr.pdf").read_bytes()).hexdigest() == provenance["pdf_sha256"]
chunks = read("data/chunks.json")
assert len(chunks) == provenance["chunks"]
assert hashlib.sha256(json.dumps(chunks, sort_keys=True).encode()).hexdigest() == provenance["corpus_sha256"]
pages = {p["page"]: p["text"] for p in read("data/pages.json")}
by_id = {c["id"]: c for c in chunks}
for c in chunks:
    assert pages[c["page"]][c["start_char"]:c["end_char"]] == c["text"]
    assert not c["text"].startswith("4.5.2016 L 119/")

experiments = read("results/retrieval_results.json")
for e in experiments:
    b, r = e["baseline"], e["reranked"]
    assert len(b) == len(r) == 10
    assert len({c["id"] for c in b}) == 10
    assert {c["id"] for c in b} == {c["id"] for c in r}
    assert all(b[i]["cosine"] >= b[i+1]["cosine"] for i in range(9))
    assert all(r[i]["cross_score"] >= r[i+1]["cross_score"] for i in range(9))
    assert [c["baseline_rank"] for c in b] == list(range(1, 11))
    assert [c["rerank_rank"] for c in r] == list(range(1, 11))
    for c in b + r:
        assert c["text"] == by_id[c["id"]]["text"]
    labels = [{"page": e["query"]["page"], "evidence": e["query"]["evidence"]}]
    labels += e["query"].get("equivalent_evidence", [])
    for c in chunks:
        relevant = any(c["page"] == a["page"] and a["evidence"] in c["text"] for a in labels)
        assert relevant == (c["id"] in e["query"]["gold_chunk_ids"])

selected = read("results/selected_cases.json")
assert len(selected) >= 2
selected_by_id = {e["query"]["id"]: e for e in selected}
for e in selected:
    gold = set(e["query"]["gold_chunk_ids"])
    assert not any(c["id"] in gold for c in e["baseline"][:3])
    assert any(c["id"] in gold for c in e["reranked"][:3])
    assert 4 <= e["metrics"]["baseline_evidence_rank"] <= 10
    assert 1 <= e["metrics"]["reranked_evidence_rank"] <= 3

with (root / "results/latency_raw.csv").open() as f:
    timings = list(csv.DictReader(f))
for row in timings:
    b, r, total = [float(row[k]) for k in ("baseline_ms", "rerank_only_ms", "advanced_total_ms")]
    assert b > 0 and r > 0 and np.isclose(b + r, total)

pairs = read("results/answer_comparisons.json")
assert {p["query_id"] for p in pairs} == set(selected_by_id)
generated = 0
for p in pairs:
    e = selected_by_id[p["query_id"]]
    a1, a2 = p["answer_1"], p["answer_2"]
    assert a1["messages"][0] == a2["messages"][0]
    assert a1["messages"][1]["content"].split("Document excerpts:")[0] == a2["messages"][1]["content"].split("Document excerpts:")[0]
    for a, stage in ((a1, "baseline"), (a2, "reranked")):
        assert a["context_ids"] == [c["id"] for c in e[stage][:3]]
        for c in e[stage][:3]:
            assert c["text"] in a["messages"][1]["content"]
        if a["status"] == "generated":
            assert a["answer"] and a["generation_ms"] > 0
            assert not a["cited_ids_outside_context"], "Review invalid citations before submission."
            generated += 1
        else:
            assert a["answer"] is None
    if a1["status"] == a2["status"] == "generated":
        assert a1["requested_model"] == a2["requested_model"]
        assert a1["request_parameters"] == a2["request_parameters"]

notebook = nbformat.read(root / "Baseline_vs_Reranking.ipynb", as_version=4)
nbformat.validate(notebook)
code_cells = [c for c in notebook.cells if c.cell_type == "code"]
assert all(c.execution_count is not None for c in code_cells), "Notebook has unexecuted cells."
assert not any(o.output_type == "error" for c in code_cells for o in c.outputs)
summary = {"artifact_consistency_checks": "passed", "queries_checked": len(experiments),
           "selected_recoveries": len(selected), "latency_measurements": len(timings),
           "answers_generated": generated, "answers_required": 2 * len(selected),
           "notebook_code_cells_executed": len(code_cells),
           "semantic_answer_review": "Requires human assessment; not established by these checks."}
(root / "results/validation.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
if args.require_answers:
    assert generated == 2 * len(selected), "Generate both LLM answers before final submission."
