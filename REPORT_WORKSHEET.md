# Student report worksheet — write your own submission

The assignment explicitly requires the report to be in your own words and not
AI-generated. This file is a worksheet, not the final report. Use the notebook
and saved outputs as evidence, and write your explanations and observations
yourself. Do not submit this blank worksheet as the report.

Student: ____________________

Repository URL: ____________________

Experiment date / model / device: ____________________

## 1. Architecture comparison

In your own words, explain how a bi-encoder represents the question and each
passage, and how cosine search works. Explain how a cross-encoder receives the
question and passage together and uses interactions between their tokens.

Your explanation:

> [Write here.]

## 2. Qualitative and rank shift analysis

Choose at least two **measured** restored cases from `results/selected_cases.json`.
Use `results/rank_shifts.csv` for exact chunk IDs, pages, ranks, and scores.
Read each chunk against the PDF. Use `results/answer_comparisons.json` or the
notebook outputs for the actual generated answers and supplied context.

| Field | Case 1 | Case 2 | Optional additional case |
| --- | --- | --- | --- |
| Query ID and question | [Fill] | [Fill] | [Fill] |
| Failure category | [Fill] | [Fill] | [Fill] |
| Necessary clause / PDF page / article | [Fill] | [Fill] | [Fill] |
| Critical chunk ID | [Fill] | [Fill] | [Fill] |
| Baseline rank → re-ranked rank | [Fill] | [Fill] | [Fill] |
| What Answer 1 actually said | [Fill] | [Fill] | [Fill] |
| What Answer 2 actually said | [Fill] | [Fill] | [Fill] |
| Your assessment with supporting wording | [Fill] | [Fill] | [Fill] |

Did Answer 1 omit an exception, misstate a rule, hallucinate, or explicitly
abstain? Did Answer 2 resolve the omission correctly? If Answer 1 was still
correct, say so. Check whether other baseline chunks provide equivalent
evidence; a chunk rank change alone does not prove an answer-quality gain.

Your observations:

> [Write here.]

Briefly disclose exploratory selection and mention an observed tie, regression,
or Top-10 miss from `results/query_metrics.csv` if present.

> [Write here.]

## 3. Latency profiling

Copy measured values from `results/latency_summary.csv`, not example numbers.
Include device, repeats, and whether times include generation, loading or
indexing. Use `results/environment.json` for the recorded environment.

| Query | Baseline median (ms) | Re-ranking stage median (ms) | Advanced total median (ms) | Advanced p95 (ms) |
| --- | --- | --- | --- | --- |
| [Fill] | | | | |
| [Fill] | | | | |

Your explanation of the observed quality/latency tradeoff:

> [Write here.]

## 4. Scaling question

Why not run the cross-encoder on every chunk from the start? Explain what can
be precomputed in the two architectures, how many query-dependent model passes
are needed over N chunks, and how restricting reranking to 10 candidates changes
cost. Explain the consequence when a necessary chunk is absent from those 10.

Your explanation:

> [Write here.]
