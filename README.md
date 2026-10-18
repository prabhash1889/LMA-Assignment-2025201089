# RAG Assignment: Baseline Retrieval vs. Re-ranking

The experiment compares two retrieval workflows over the original 88-page GDPR
PDF. Both use the same chunked corpus and initial Top 10 candidates.

- **Baseline:** dense cosine retrieval → first 3 chunks → Answer 1.
- **Re-ranking:** score the same 10 candidates with a cross-encoder → best 3
  chunks → Answer 2.

Start with [Baseline_vs_Reranking.ipynb](Baseline_vs_Reranking.ipynb), which
contains the full implementation, explanations, executed outputs, and figures.
The report is available as [PDF](REPORT.pdf) and [Markdown](REPORT.md).

## Results

| Case | Critical chunk | Baseline rank | Re-ranked rank |
| --- | --- | ---: | ---: |
| Public authority and legitimate interests | `p009-c01` | 8 | 1 |
| Small organisation and processing records | `p051-c02` | 9 | 1 |
| Scientific research and purpose limitation | `p035-c02` | 7 | 1 |

Both answers were generated using `deepseek/deepseek-v4.1-flash` through
OpenRouter. The model, prompt structure and generation settings are shared;
only the supplied context changes. In these three examples, the baseline
abstained because its context was insufficient, while re-ranking supplied the
necessary evidence.

The complete exploratory pool contains 54 query variants covering 18 clause
targets. Six queries gained the necessary clause in the Top 3; thirteen missed
it in the initial Top 10. These deliberately selected stress queries do not
constitute an unbiased estimate of average retrieval accuracy.

![Evidence rank changes and retrieval latency](results/comparison.png)

Raw ranks, prompts, model responses and measured timings are included under
`results/`. An additional context-boundary example records an incomplete rule
and is excluded from the successful recovery cases.

## Setup and execution

Python 3.12 is recommended. The supplied requirements use CPU PyTorch; a GPU is
not required. The first execution downloads the embedding and cross-encoder
weights. The source PDF is bundled for reproducibility.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.lock.txt
cp .env.example .env
```

Set `OPENROUTER_API_KEY` in `.env`. The example selects the model used in the
saved experiment. Register the environment as a notebook kernel:

```bash
.venv/bin/python -m ipykernel install --user --name rag-assignment \
  --display-name "RAG Assignment (Python 3.12)"
```

Open the notebook in VS Code or Jupyter, select that kernel, and run the cells
in order from the repository root. JupyterLab can be installed separately:

```bash
uv pip install --python .venv/bin/python jupyterlab
.venv/bin/python -m jupyterlab
```

`requirements.lock.txt` records the exact package versions used.
`requirements.txt` lists the direct dependencies. Re-running the notebook
writes new measurements to the evidence files. Matching LLM requests reuse a
local response cache, avoiding additional generation requests.

To use the optional local answer model, set `RAG_LLM_BACKEND=local` in `.env`.
This downloads `Qwen/Qwen2.5-1.5B-Instruct` and can be slow on CPU. Setting
`RAG_LLM_BACKEND=none` produces retrieval-only outputs without LLM answers.

## Method and controls

- The corpus contains 391 page-local chunks with a 210-token window and
  35-token overlap. Chunk IDs preserve their PDF page.
- The bi-encoder is `sentence-transformers/all-MiniLM-L6-v2`. Normalized vectors
  are stored in a NumPy matrix; exact dot-product search equals cosine search.
- The cross-encoder is `cross-encoder/ms-marco-MiniLM-L6-v2`. It scores all ten
  query–passage pairs and sorts them by relevance. It cannot add new candidates.
- Retrieval model revisions, document hashes, seed and runtime environment are
  recorded. Necessary-clause labels also include equivalent document wording.
- Latency is measured after warm-up, ten times per selected query. Advanced
  total includes query embedding, dense search, re-ranking and sorting. PDF
  extraction, model loading, indexing and LLM generation are excluded.
- Cross-encoder logits are relevance scores, not calibrated probabilities.
  Citation consistency does not by itself establish factual correctness.
- The document is the original 2016 text, retained as a fixed experimental
  corpus. The experiment does not establish the content of current consolidated
  law.

## Submission files

| Path | Contents |
| --- | --- |
| `Baseline_vs_Reranking.ipynb` | Complete executable implementation and saved outputs |
| `REPORT.pdf`, `REPORT.md` | Report with architecture, cases, timings, scaling discussion and plot |
| `data/gdpr.pdf` | The single complex source document |
| `data/queries.json` | Annotated stress queries and evidence clauses |
| `data/provenance.json` | Source URL, hashes, model revisions and chunk configuration |
| `results/` | Retrieval results, answers, rank shifts, raw timings, environment and plot |
| `requirements*.txt`, `.env.example` | Environment and API configuration |

The notebook contains its query annotations and regenerates extraction
intermediates. Local development files, caches and credentials are excluded
from Git.

## References

- [Official original GDPR PDF, EUR-Lex](https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32016R0679)
- [Sentence Transformers: retrieve and re-rank](https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html)
- [Bi-encoder model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [Cross-encoder model card](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)
- [OpenRouter chat completion API](https://openrouter.ai/docs/api/api-reference/chat/send-chat-completion-request)
