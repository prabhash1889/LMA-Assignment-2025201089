# RAG assignment: baseline vs. re-ranking

A reproducible experiment comparing dense Top 3 retrieval with cross-encoder
re-ranking of the **same Top 10**, over the original 88-page GDPR PDF.

**Start with [Baseline_vs_Reranking.ipynb](Baseline_vs_Reranking.ipynb).** It embeds
the complete pipeline implementation, explanations, tables, and saved outputs.
The PDF used as data is `data/gdpr.pdf`; `Assignment.pdf` is the specification,
not retrieval data.

The saved notebook contains a completed OpenRouter experiment using
`deepseek/deepseek-v4.1-flash`: both answers for three distinct failure cases,
their rank shifts, and repeated latency measurements. Check
`results/run_status.json` and `results/validation.json` for execution status.
The additional saved `results/context_boundary_case.json` preserves an observed
limitation: a candidate ended mid-rule and its generated answer omitted an
essential condition. Its evidence label was tightened to require the complete
clause; it is excluded from the successful examples. The initial eight real API
answers are retained across the six main answers and this two-answer case.

## Run locally

Python 3.12 is recommended. The requirements install CPU PyTorch. A GPU is not
required. First execution downloads the embedding and cross-encoder weights
from Hugging Face. Answer generation can use OpenRouter or a local Hugging Face
LLM; all retrieval and re-ranking run locally.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.lock.txt
cp .env.example .env
# Edit .env: set OPENROUTER_API_KEY and optionally OPENROUTER_MODEL.
.venv/bin/python -m ipykernel install --user --name rag-assignment \
  --display-name "RAG Assignment (Python 3.12)"
```

Open the notebook in VS Code or Jupyter and select that kernel, then **Run All**.
The notebook assumes its working directory is this repository's root. Jupyter's
UI can be installed separately with `uv pip install --python .venv/bin/python jupyterlab`.
You can also execute everything without a notebook UI:

```bash
.venv/bin/python baseline_vs_reranking.py
```

To use the unpinned direct dependencies instead of the exact recorded environment,
install `requirements.txt`. The lock file records the environment used for the
committed experiment; CPU wheels are selected using PyTorch's official index.
For CUDA, install a compatible PyTorch build separately, remove the CPU torch
entry when installing other dependencies, and set `RAG_DEVICE=cuda`.

### Alternative answer backends

```bash
# Real local LLM: downloads Qwen/Qwen2.5-1.5B-Instruct (~3 GB), slower on CPU.
RAG_LLM_BACKEND=local .venv/bin/python baseline_vs_reranking.py

# Retrieval, ranking and latency only; not a complete assignment submission.
RAG_LLM_BACKEND=none .venv/bin/python baseline_vs_reranking.py
```

OpenRouter model availability depends on the account/provider. The default is
`deepseek/deepseek-v4.1-flash`; change `OPENROUTER_MODEL` if necessary. The supplied
query pool selects three primary categories and needs six generation requests,
plus retries on rate limits or server errors. Responses are cached by the complete request.
Re-running an identical experiment reuses answers and makes no additional paid
generation request. Change the payload or delete `results/answer_cache/` to
regenerate. `.env`, caches, virtual environments and logs are excluded from Git.

## What is implemented

| Assignment requirement | Implementation / evidence |
| --- | --- |
| One complex PDF | Official original GDPR, 88 pages, URL and SHA-256 in `data/provenance.json` |
| Common chunked corpus | Page-local 210-token windows, 35-token overlap; `data/chunks.json` |
| Dense bi-encoder vector store | Pinned all-MiniLM-L6-v2, normalized NumPy matrix, exact cosine search |
| Baseline Top 10 → Top 3 → Answer 1 | Notebook sections 3 and 7 |
| Same Top 10 → cross-encoder → Top 3 → Answer 2 | Pinned ms-marco-MiniLM-L6-v2, score every query/chunk pair |
| Identical generation settings | One prompt builder and one generator; full messages saved |
| At least two actual recoveries | Required at runtime; full Top 10 and source text exported |
| Failure categories | Keyword/role distractors, exceptions, negation, and document-specific rules |
| Rank shift evidence | `results/rank_shifts.csv`, `results/selected_cases.json` |
| Compare generated answers | `results/answer_comparisons.json`, `results/answers.csv` |
| Measured baseline vs. advanced latency | `results/latency_raw.csv`, `results/latency_summary.csv`, plot |
| Reproducibility | Pinned retrieval model revisions, seed, hashes, package and hardware record |
| Student-authored report | `REPORT_WORKSHEET.md` provides prompts; write the submission yourself |

## Experimental controls and limits

- One chunk list and one initial candidate list are shared. The reranker cannot
  introduce new passages or recover evidence missing from the Top 10.
- The query pool is **exploratory stress testing**, not a random test set. It has
  54 query variants covering 18 clause targets; variants are correlated. Do not
  present its aggregate scores as population-level accuracy.
- Gold labels identify a necessary clause on a known PDF page. Every overlapping
  chunk containing that clause counts. These are clause-coverage labels, not an
  exhaustive relevance set. Read the exact source and both Top 3 contexts when
  judging whether an answer could be supported by equivalent wording elsewhere.
- All exploratory outcomes are saved, including ties, regressions and Top-10
  misses. Only real rank 4–10 → Top 3 recoveries qualify for the selected examples.
- Timings measure uncached online retrieval after warm-up. Advanced total means
  **query encoding + cosine search + cross-encoder scoring + sorting**. LLM calls,
  downloads, PDF extraction, model loading, and indexing are excluded and recorded
  separately. GPU runs synchronize the device before measuring boundaries.
- Both answers use identical model/prompt/settings; only contexts change.
  Temperature zero does not guarantee provider-level determinism.
- Citation checks detect out-of-context chunk IDs. They do not prove that an
  answer is factually supported. Qualitative conclusions require human review.
- The dataset is the original 2016 regulation, retained as a fixed experimental
  document; this project makes no claim about current consolidated law.

## Files and regeneration

```text
Baseline_vs_Reranking.ipynb   Primary, fully documented notebook with outputs
baseline_vs_reranking.py     Same code in readable percent-cell source format
data/gdpr.pdf                Fixed complex source PDF
data/queries.json            Query pool and independent necessary-clause labels
data/pages.json              Cleaned PDF text with one-based PDF page numbers
data/chunks.json             All chunks and source offsets
data/provenance.json         Source URL, hashes, token sizes and model revisions
results/                    Saved experiment evidence and generation responses
REPORT.md / REPORT.pdf      Complete report text and printable PDF for your review
REPORT_WORKSHEET.md          Prompts for writing your own final report
scripts/build_notebook.py    Rebuild notebook from percent-cell source
scripts/create_queries.py    Recreate the original annotated query pool
scripts/execute_notebook.py  Execute and save notebook outputs
requirements.lock.txt       Exact package versions used
```

After editing the Python source, regenerate and execute the notebook:

```bash
.venv/bin/python scripts/build_notebook.py
.venv/bin/python scripts/execute_notebook.py
.venv/bin/python scripts/validate_results.py --require-answers
```

In environments that prohibit even local kernel sockets, use
`scripts/execute_notebook.py --in-process`. It executes the same cells in a real
IPython shell and captures their actual tables, figures and text; it does not
simulate notebook outputs. External network access is still required for API
answer generation and initial model downloads.

Executing the notebook overwrites result files with the new measurements. The
provided PDF is bundled for stable reproduction; if absent, it is downloaded
from the official source. Changes to PDF bytes or chunk configuration invalidate
the embedding cache automatically. Gold labels are validated against extracted
text before evaluation.

## Sources

- [Assignment specification](Assignment.pdf)
- [Official original GDPR PDF, EUR-Lex](https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32016R0679)
- [Sentence Transformers: retrieve and re-rank](https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html)
- [Sentence Transformers: cross-encoders](https://www.sbert.net/examples/cross_encoder/applications/README.html)
- [Bi-encoder model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [Cross-encoder model card](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)
- [OpenRouter chat completion API](https://openrouter.ai/docs/api/api-reference/chat/send-chat-completion-request)

## Before submission

Read the notebook and the exact evidence contexts. Complete your report in your
own words using `REPORT_WORKSHEET.md`. Confirm that both generated answers are
present for at least two selected recoveries. Submit the repository link plus
your own concise report, as requested in the assignment.

### Publish the code to GitHub

The publishing helper creates a **new public repository** named
`rag-baseline-vs-reranking-gdpr` in your authenticated account. It validates the
experiment and scans the explicit bundle for your API key before copying it to
a separate Git checkout under `dist/` and uploading it. `.env`, caches, weights
and the virtual environment are excluded. It requires Git, an authenticated
GitHub CLI (`gh auth login`), and a configured Git author identity.

```bash
.venv/bin/python scripts/publish_github.py
```

An optional positional argument selects a different new repository name. The
script refuses to reuse an existing remote repository. If creation succeeds but
the push fails, the displayed local checkout under `dist/github-*/` can be used
to retry `git push -u origin main`. On success, the URL is printed and saved in
`results/github_repository.json`.
