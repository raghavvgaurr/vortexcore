"""Turn the user's question into search queries: decompose multi-part questions, rewrite single ones."""
from langchain_core.prompts import ChatPromptTemplate

from src.config import DECOMPOSE_MODEL, REWRITE_MODEL
from src.llm import get_llm

rewrite_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a query-rewriting component in a Retrieval-Augmented
Generation (RAG) system.

Rewrite the user's question into a clear, self-contained query
that is suitable for searching a document collection.

Rules:
1. Preserve the user's original intent.
2. Preserve all named entities, dates, numbers, and constraints.
3. Preserve every part of a multi-part question.
4. Resolve unclear references only when the supplied question
   provides enough information.
5. Do not answer the question.
6. Do not invent facts, entities, dates, or assumptions.
7. Do not add information merely because it seems likely.
8. Return only the rewritten question, without explanations
   or quotation marks.
"""
    ),
    ("human", "Original question: {question}")
])

decomposition_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a query decomposition component in a Retrieval-Augmented
Generation (RAG) system.

Determine whether the user's question contains multiple distinct
information requirements.

Rules:
1. If the question contains only one information requirement,
   return exactly one query.
2. If it contains multiple independent information requirements,
   split it into separate self-contained search queries.
3. Preserve all named entities, dates, numbers, and constraints.
4. Do not answer the question.
5. Do not invent facts.
6. Each query must be understandable independently.
7. Return ONLY the queries, one per line.
8. Do not number the queries.
9. Do not add explanations.

User question:
{question}
"""
    )
])


def rewrite_query(question: str) -> str:
    """Rewrite a question while preserving its original intent."""
    question = question.strip()
    if not question:
        raise ValueError("Question cannot be empty.")

    chain = rewrite_prompt | get_llm(REWRITE_MODEL)
    rewritten = chain.invoke({"question": question}).content.strip()

    if not rewritten:
        raise ValueError("The query rewriter returned an empty result.")
    return rewritten


def decompose_query(question: str) -> list[str]:
    question = question.strip()
    if not question:
        raise ValueError("Question cannot be empty.")

    chain = decomposition_prompt | get_llm(DECOMPOSE_MODEL)
    response = chain.invoke({"question": question})

    queries = [line.strip() for line in response.content.splitlines() if line.strip()]
    if not queries:
        raise ValueError("Query decomposition returned no queries.")
    return queries