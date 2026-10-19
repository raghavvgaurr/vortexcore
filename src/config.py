"""Settings shared across the project. Change numbers here, not inside other files."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SAMPLE_PDF = DATA_DIR / "2024-wttc-introduction-to-ai.pdf"
RESULTS_DIR = ROOT / "evaluation" / "results"
MAX_UPLOAD_MB = 25

# Chunking: parents give the LLM context, children are what we search.
PARENT_CHUNK_SIZE = 1000
PARENT_CHUNK_OVERLAP = 150
CHILD_CHUNK_SIZE = 250
CHILD_CHUNK_OVERLAP = 50

# Models
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# All three LLM steps ran on this Groq model in the notebook, so keep it that way.
DECOMPOSE_MODEL = "openai/gpt-oss-120b"
REWRITE_MODEL = "openai/gpt-oss-120b"
ANSWER_MODEL = "openai/gpt-oss-120b"
LLM_TIMEOUT_SEC = 60
LLM_MAX_RETRIES = 3  # the Groq client backs off between retries (rate limits included)

# Retrieval
DENSE_K = 10
BM25_K = 10
RRF_K = 60
CANDIDATE_K = 20
TOP_K = 5
NEAR_DUPLICATE_THRESHOLD = 0.96

# Refuse to answer when the best reranker score is below this. None = never refuse
# early (the answer prompt still says "not in the context"). Set it only after
# running `python -m evaluation.ablation`, because reranker scores are raw logits.
MIN_RERANK_SCORE = None


def get_secret(name: str) -> str:
    """Read a key from the environment, then from Streamlit secrets."""
    value = os.getenv(name)
    if not value:
        try:
            import streamlit as st
            value = st.secrets.get(name)
        except Exception:
            value = None
    if not value:
        raise RuntimeError(
            f"Missing {name}. Put it in your .env file or in Streamlit secrets."
        )
    return value