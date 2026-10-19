"""One place that creates the Groq chat model, so keys and timeouts are set once."""
from functools import cache

from src.config import LLM_MAX_RETRIES, LLM_TIMEOUT_SEC, get_secret


@cache
def get_llm(model: str):
    from langchain_groq import ChatGroq

    return ChatGroq(
        model=model,
        temperature=0,
        api_key=get_secret("GROQ_API_KEY"),
        timeout=LLM_TIMEOUT_SEC,
        max_retries=LLM_MAX_RETRIES,
    )