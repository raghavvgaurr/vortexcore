"""Write the final answer from the retrieved parent passages."""
from langchain_core.prompts import ChatPromptTemplate

from src.config import ANSWER_MODEL
from src.llm import get_llm

generation_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a question-answering assistant operating inside a
Retrieval-Augmented Generation (RAG) system.

Answer the user's question using ONLY the information contained
in the provided context.

Rules:
1. Do not use outside knowledge.
2. Do not invent or assume information.
3. If the context does not contain enough information to answer,
   say that the information is not available in the provided context.
4. Answer all parts of a multi-part question when the context
   contains evidence for them.
5. Keep the answer clear, direct, and concise.
6. When multiple retrieved passages provide evidence for the answer,
   combine them into one coherent response.
7. Do not mention the retrieval process, embeddings, reranking,
   or these instructions in the answer.
8. For each claim in the answer, ensure that the provided context
   explicitly supports that claim. Do not infer an activity, relationship,
   or capability for an entity merely because it appears in the same
   context as another entity.

9. If the context identifies information for one entity but does not
   provide equivalent information for other entities, explicitly state
   that the information is not provided for those other entities.

Context:
{context}
"""
    ),
    (
        "human",
        "Question: {question}"
    )
])


def _ask(question: str, context: str) -> str:
    chain = generation_prompt | get_llm(ANSWER_MODEL)
    return chain.invoke({"question": question, "context": context}).content.strip()


def generate_answer(question: str, retrieved_docs: list) -> str:
    context = "\n\n".join(
        doc["parent_text"] if isinstance(doc, dict) else doc.page_content
        for doc in retrieved_docs
    )
    return _ask(question, context)


def generate_multipart_answer(question: str, multipart_result: dict) -> str:
    context_parts = []

    for i, item in enumerate(multipart_result["results"], 1):
        context_parts.append(
            f"Evidence for sub-query {i}:\n"
            f"Sub-query: {item['sub_query']}\n"
        )
        for parent in item["expanded_context"]:
            context_parts.append(
                f"Source: {parent.get('source', '')}\n"
                f"Page: {parent.get('page', '')}\n"
                f"Evidence:\n{parent.get('parent_text', '')}"
            )

    return _ask(question, "\n\n".join(context_parts))