"""DocPilot: chat with your PDFs, with page-level citations.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import os

import streamlit as st

from rag.generator import answer_question
from rag.ingest import build_chunks
from rag.retriever import HybridRetriever

st.set_page_config(page_title="DocPilot", page_icon="📄", layout="wide")

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("📄 DocPilot")
    st.caption("Ask questions across your PDFs. Every answer is cited.")

    files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

    st.subheader("Settings")
    mode = st.radio(
        "Search mode",
        ["hybrid", "keyword", "vector"],
        help="Hybrid fuses BM25 keyword search and embeddings with Reciprocal Rank Fusion.",
    )
    top_k = st.slider("Passages retrieved", 2, 10, 5)
    chunk_size = st.slider("Chunk size (characters)", 300, 1500, 800, step=100)

    env_key = os.environ.get("ANTHROPIC_API_KEY", "")
    api_key = st.text_input(
        "Anthropic API key (optional)",
        value=env_key,
        type="password",
        help="Without a key, DocPilot returns the best-matching passages directly.",
    )

# ---------------------------------------------------------------- indexing
@st.cache_resource(show_spinner="Reading and indexing your documents...")
def build_index(file_signature: tuple, chunk_size: int, _files):
    chunks = []
    for f in _files:
        chunks.extend(build_chunks(f, f.name, start_id=len(chunks), chunk_size=chunk_size))
    return HybridRetriever(chunks), chunks


st.header("Chat with your documents")

if not files:
    st.info("Upload one or more PDFs in the sidebar to get started.")
    st.stop()

signature = tuple((f.name, f.size) for f in files)
retriever, chunks = build_index(signature, chunk_size, files)

c1, c2, c3 = st.columns(3)
c1.metric("Documents", len(files))
c2.metric("Passages indexed", len(chunks))
c3.metric("Retrieval", "hybrid-ready" if retriever.has_embeddings else "keyword only")

if mode != "keyword" and not retriever.has_embeddings:
    st.warning("Embedding model unavailable here, falling back to keyword search.")
    mode = "keyword"

# ---------------------------------------------------------------- chat
if "history" not in st.session_state:
    st.session_state.history = []


def render_answer(entry: dict) -> None:
    st.markdown(entry["text"])
    if not entry["found"]:
        return
    with st.expander(f"Sources ({len(entry['sources'])})"):
        for src in entry["sources"]:
            st.markdown(f"**{src['doc']}**, page {src['page']}")
            st.caption(src["text"])


for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        render_answer(turn)

question = st.chat_input("Ask a question about your documents")
if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            hits = retriever.search(question, k=top_k, mode=mode)
            ans = answer_question(question, hits, api_key=api_key or None)
        entry = {
            "question": question,
            "text": ans.text,
            "found": ans.found,
            "sources": [
                {"doc": h.chunk.doc, "page": h.chunk.page, "text": h.chunk.text}
                for h in ans.sources
            ],
        }
        render_answer(entry)
        st.caption(f"mode: {ans.mode} · search: {mode}")
    st.session_state.history.append(entry)
