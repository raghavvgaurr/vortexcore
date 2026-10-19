"""Streamlit front end. The logic lives in src/; this file only shows it."""
import logging
import tempfile
import time
from pathlib import Path

import streamlit as st

from src import config
from src.pipeline import build_index, rag_pipeline

log = logging.getLogger(__name__)

st.set_page_config(page_title="Document Q&A", layout="centered")


@st.cache_resource(show_spinner="Reading the sample report (OCR takes a minute the first time)...")
def load_sample_index():
    # Shared between visitors, which is fine because it is a public document.
    return build_index([config.SAMPLE_PDF], force_ocr=True)


def index_uploads(files):
    """Save the uploads to a temp folder and index them. Uploaded files stay private to this session."""
    with tempfile.TemporaryDirectory() as folder:
        paths = []
        for file in files:
            if file.size > config.MAX_UPLOAD_MB * 1024 * 1024:
                raise ValueError(f"{file.name} is bigger than {config.MAX_UPLOAD_MB} MB.")
            data = file.getvalue()
            if not data.startswith(b"%PDF"):
                raise ValueError(f"{file.name} is not a PDF.")
            path = Path(folder) / Path(file.name).name
            path.write_bytes(data)
            paths.append(path)
        return build_index(paths)


def document_names(index):
    return sorted({c.metadata.get("document_id") or "document" for c in index.children})


def collect_sources(result):
    contexts = (result["expanded_context"] if result["mode"] == "single"
                else [c for part in result["results"] for c in part["expanded_context"]])
    return [{
        "file": Path(c["source"]).name if c["source"] else "unknown",
        "page": (c["page"] + 1) if isinstance(c["page"], int) else "?",  # stored 0-based
        "score": c["reranker_score"],
        "text": c["parent_text"],
    } for c in contexts]


def show_answer(entry, show_details):
    st.markdown(entry["answer"])
    if entry["sources"]:
        with st.expander(f"Sources ({len(entry['sources'])})"):
            for source in entry["sources"]:
                st.markdown(f"**{source['file']}**, page {source['page']} (reranker score {source['score']:.2f})")
                st.caption(source["text"][:500] + ("..." if len(source["text"]) > 500 else ""))
    st.caption(f"{entry['latency']:.1f}s" + (" · answered from cache" if entry["cached"] else ""))
    if show_details:
        with st.expander("Retrieval details"):
            st.write({"mode": entry["mode"], "search queries": entry["sub_queries"],
                      "rewritten query": entry["rewritten_query"]})


for key, default in (("index", None), ("history", []), ("cache", {})):
    st.session_state.setdefault(key, default)

with st.sidebar:
    st.header("Documents")
    uploads = st.file_uploader("PDF files", type="pdf", accept_multiple_files=True)
    if st.button("Index uploaded files", disabled=not uploads):
        try:
            with st.spinner("Reading and indexing..."):
                st.session_state.index = index_uploads(uploads)
            st.session_state.history = []
        except (ValueError, RuntimeError) as err:
            st.error(str(err))
    if config.SAMPLE_PDF.exists() and st.button("Use the sample report"):
        st.session_state.index = load_sample_index()
        st.session_state.history = []

    index = st.session_state.index
    selected = []
    if index is not None:
        names = document_names(index)
        st.caption(f"{len(index.children)} passages indexed")
        selected = st.multiselect("Search in", names, default=names) if len(names) > 1 else names

    st.header("Settings")
    candidate_k = st.slider("Candidates to rerank", 10, 50, config.CANDIDATE_K, step=5)
    top_k = st.slider("Passages used for the answer", 1, 10, config.TOP_K)
    show_details = st.checkbox("Show retrieval details")

st.title("Document Q&A")
index = st.session_state.index
if index is None:
    st.info("Upload a PDF in the sidebar and press **Index uploaded files** to begin.")

for entry in st.session_state.history:
    with st.chat_message("user"):
        st.write(entry["question"])
    with st.chat_message("assistant"):
        show_answer(entry, show_details)

question = st.chat_input("Ask a question about your documents", disabled=index is None)
if question:
    with st.chat_message("user"):
        st.write(question)

    all_names = document_names(index)
    document_ids = None if set(selected) == set(all_names) else set(selected)
    cache_key = (question, candidate_k, top_k, tuple(sorted(selected)))
    cached = st.session_state.cache.get(cache_key)

    with st.chat_message("assistant"):
        if not selected:
            st.warning("Select at least one document to search.")
        else:
            try:
                start = time.perf_counter()
                if cached:
                    result = cached
                else:
                    with st.spinner("Searching..."):
                        result = rag_pipeline(question, index, candidate_k, top_k,
                                              document_ids=document_ids, min_score=config.MIN_RERANK_SCORE)
                    st.session_state.cache[cache_key] = result
                entry = {
                    "question": question, "answer": result["answer"], "sources": collect_sources(result),
                    "latency": time.perf_counter() - start, "cached": bool(cached), "mode": result["mode"],
                    "sub_queries": result["sub_queries"], "rewritten_query": result.get("rewritten_query"),
                }
                st.session_state.history.append(entry)
                show_answer(entry, show_details)
            except RuntimeError as err:  # for example a missing API key
                st.error(str(err))
            except Exception as err:
                log.exception("Question failed")
                st.error(f"The request failed: {err}")