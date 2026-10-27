# 📄 DocPilot: chat with your PDFs, with citations you can verify

DocPilot is a retrieval-augmented generation (RAG) app. Upload one or more PDFs, ask questions in plain English, and get answers that **cite the exact document and page** they came from. If the answer isn't in your documents, it says so instead of making something up.

> **Live demo:** _add your Hugging Face Spaces / Streamlit Cloud link here_
> **Demo GIF:** _add a 20-second screen recording here_

## Why this project

Most "chat with PDF" demos stop at upload-and-ask. DocPilot focuses on the parts that make RAG trustworthy in practice:

| Feature | What it does |
|---|---|
| **Page-level citations** | Every claim links to the passage, document and page it was drawn from |
| **Multi-PDF search** | Ask one question across several documents at once |
| **Hybrid retrieval** | BM25 keyword search + dense embeddings, merged with Reciprocal Rank Fusion |
| **Abstention** | Two guardrails refuse to answer when evidence is weak (no hallucinated answers) |
| **Measured quality** | `evaluate.py` reports Hit@k, MRR, answer rate and abstention on a labelled question set |
| **Works without an API key** | Falls back to extractive answers, so anyone can try it for free |

## Architecture

```
 PDFs ──► pypdf (page-aware text)
            │
            ▼
      sentence-aware chunking (overlap, tagged with doc + page)
            │
            ├──► BM25 index (keywords, names, numbers)
            └──► MiniLM embeddings (meaning, paraphrase)
                          │
                          ▼
              Reciprocal Rank Fusion  ──► top-k passages
                          │
          Guardrail 1: IDF-weighted question coverage too low? ──► "not found"
                          │
                          ▼
                Claude (grounded prompt, numbered sources)
                          │
          Guardrail 2: model replies NOT_FOUND? ──► "not found"
                          │
                          ▼
                  Answer + [1][2] citations ──► Streamlit UI (expandable sources)
```

### Design decisions worth discussing

- **Why hybrid search?** Embeddings find paraphrases ("time off" ≈ "leave") but can miss exact tokens like `15,000` or `Tuesdays`. BM25 is the opposite. Fusing ranks with RRF needs no score calibration between the two systems.
- **Why IDF-weighted coverage for the first guardrail?** Plain word overlap let off-topic questions through because words like "company" and "annual" appear everywhere. Weighting by inverse document frequency makes rare terms count more, and a term that appears nowhere in the corpus (like "CEO" in a handbook that never mentions one) counts the most. Moving from plain overlap to IDF coverage raised abstention on my unanswerable set from **40% to 100%**.
- **Why chunk by sentences with overlap?** Splitting mid-sentence cuts facts in half. Overlap keeps context that straddles a boundary.
- **Why temperature 0 and numbered sources?** Reproducible answers and a clean mapping from `[n]` markers back to real passages.

## Quick start

```bash
git clone <your-repo-url> && cd docpilot
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python make_sample.py            # creates sample_docs/northwind_handbook.pdf
export ANTHROPIC_API_KEY=...     # optional; omit to use extractive mode
streamlit run app.py
```

Upload `sample_docs/northwind_handbook.pdf` and try:

- "How many days of annual leave do I get?"
- "What is the hotel cap in metro cities?"
- "Who is the CEO?" ← should refuse, because the handbook never says

## Evaluation

```bash
python evaluate.py                   # keyword vs vector vs hybrid
python evaluate.py --k 5 --chunk-size 400
```

The evaluation set (`eval/questions.json`) has 20 answerable questions, each labelled with the expected keyword and page, plus 5 unanswerable questions that must be refused.

Results on the sample handbook (chunk size 250, k = 3, keyword mode, run in a network-restricted sandbox):

| Mode | Hit@3 | MRR | Answered | Abstained |
|---|---|---|---|---|
| keyword | 95% | 0.925 | 90% | 100% |
| vector | _run locally_ | | | |
| hybrid | _run locally_ | | | |

> **To finish this table:** run `python evaluate.py` on a machine with internet access (it downloads the embedding model on first run) and paste the vector and hybrid rows. Reporting the before/after of adding embeddings is the most compelling number in this project.

**Metrics**: *Hit@k*: a retrieved passage contains the answer and comes from the correct page. *MRR*: mean reciprocal rank of the first correct passage. *Answered*: share of answerable questions the system answers. *Abstained*: share of unanswerable questions correctly refused. Answer and abstain rates run in extractive mode so they cost nothing and are fully reproducible.

## Project structure

```
docpilot/
├── app.py               # Streamlit UI: upload, settings, chat, expandable sources
├── evaluate.py          # Hit@k, MRR, answer rate, abstention
├── make_sample.py       # builds the demo PDF
├── eval/questions.json  # labelled evaluation set
└── rag/
    ├── ingest.py        # page-aware PDF parsing + chunking
    ├── retriever.py     # BM25 + embeddings + RRF + coverage scoring
    └── generator.py     # grounded prompt, citations, abstention
```

## Deploy for free

1. Push to GitHub.
2. Create a Streamlit Community Cloud or Hugging Face Space (Streamlit SDK) from the repo.
3. Add `ANTHROPIC_API_KEY` as a secret (or skip it to run in extractive mode).
4. Paste the live link at the top of this README.

## Limitations and next steps

- Scanned PDFs need OCR first (e.g. `ocrmypdf`); DocPilot reads embedded text only.
- The coverage guardrail is lexical, so heavy paraphrase can cause a valid question to be declined. Embedding-based confidence is the natural upgrade.
- Ideas: a cross-encoder re-ranker, highlighting the cited sentence inside the PDF, conversation memory, and expanding the evaluation set with LLM-judged answer correctness.

## Tech stack

Python · Streamlit · pypdf · rank-bm25 · sentence-transformers (`all-MiniLM-L6-v2`) · Anthropic Claude API
