"""
DocuMind Answer Generator
--------------------------
Takes re-ranked document chunks and a user query, assembles a grounded
prompt, and calls the configured LLM to produce a cited answer.

Supports:
  • OpenAI GPT-4o / GPT-4o-mini  (default)
  • Mistral via LangChain

The prompt is engineered for enterprise Q&A:
  - Grounded in retrieved context only (reduces hallucination)
  - Citations reference [1], [2], … inline
  - Falls back gracefully when context is insufficient
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from src.utils.exceptions import LLMContextTooLongError, LLMError, LLMRateLimitError
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Prompt templates ─────────────────────────────────────────────────

SYSTEM_PROMPT = """You are DocuMind, an enterprise document Q&A assistant.
Answer questions ONLY using the provided context passages.
Rules:
1. Cite every factual claim using [N] notation where N is the passage number.
2. If the answer is not found in the context, say "I don't have enough information in the provided documents to answer this question."
3. Be concise but complete. Use bullet points for lists.
4. Never fabricate information not found in the context.
5. At the end, include a "Sources" section listing which passages you cited."""

HUMAN_PROMPT_TEMPLATE = """Context passages:
{context}

Question: {question}

Answer (with citations):"""


def _build_context(chunks: List[Tuple[Document, float]]) -> Tuple[str, List[Dict]]:
    """
    Format retrieved chunks into a numbered context string for the prompt.

    Returns:
        context_str: Formatted context to inject into the prompt.
        sources:     List of source dicts for the API response metadata.
    """
    parts = []
    sources = []

    for i, (doc, score) in enumerate(chunks, start=1):
        filename = doc.metadata.get("filename", "unknown")
        page = doc.metadata.get("page", "?")
        text = doc.page_content.strip()

        parts.append(f"[{i}] (Source: {filename}, page {page})\n{text}")
        sources.append(
            {
                "citation_number": i,
                "filename": filename,
                "page": page,
                "chunk_id": doc.metadata.get("chunk_id", ""),
                "relevance_score": round(float(score), 4),
                "preview": text[:120] + ("…" if len(text) > 120 else ""),
            }
        )

    return "\n\n---\n\n".join(parts), sources


# ── Generator class ──────────────────────────────────────────────────

class AnswerGenerator:
    """
    Generates grounded answers from retrieved context using an LLM.

    Args:
        llm_provider:  "openai" or "mistral".
        model_name:    Model identifier (e.g. "gpt-4o-mini").
        api_key:       Provider API key.
        temperature:   Sampling temperature (0 = deterministic).
        max_tokens:    Max tokens in the LLM response.
    """

    def __init__(
        self,
        llm_provider: str = "openai",
        model_name: str = "gpt-4o-mini",
        api_key: str = "",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> None:
        self.llm_provider = llm_provider
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._llm = None

    def _get_llm(self):
        """Lazily initialise the LangChain LLM wrapper."""
        if self._llm is not None:
            return self._llm

        if self.llm_provider == "openai":
            from langchain_groq import ChatGroq

            self._llm = ChatGroq(
                model=self.model_name,
                api_key=self.api_key,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        elif self.llm_provider == "mistral":
            from langchain_community.chat_models import ChatMistralAI

            self._llm = ChatMistralAI(
                model=self.model_name,
                api_key=self.api_key,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        else:
            raise LLMError(
                f"Unsupported LLM provider: '{self.llm_provider}'",
                details={"supported": ["openai", "mistral"]},
            )

        logger.info(
            "LLM initialised",
            provider=self.llm_provider,
            model=self.model_name,
        )
        return self._llm

    def generate(
        self,
        query: str,
        chunks: List[Tuple[Document, float]],
    ) -> Dict:
        """
        Generate a grounded, cited answer for the user query.

        Args:
            query:  User's question.
            chunks: Re-ranked (Document, score) pairs from the retriever.

        Returns:
            Dict with keys:
                answer   — LLM response text with [N] citations
                sources  — list of source metadata dicts
                model    — model identifier used
                provider — LLM provider used

        Raises:
            LLMError, LLMRateLimitError, LLMContextTooLongError
        """
        if not chunks:
            return {
                "answer": "No relevant documents were found to answer your question.",
                "sources": [],
                "model": self.model_name,
                "provider": self.llm_provider,
            }

        context_str, sources = _build_context(chunks)
        human_content = HUMAN_PROMPT_TEMPLATE.format(
            context=context_str,
            question=query,
        )

        logger.info(
            "Generating answer",
            query=query[:80],
            num_chunks=len(chunks),
            model=self.model_name,
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=human_content),
        ]

        try:
            llm = self._get_llm()
            response = llm.invoke(messages)
            answer_text = response.content
        except Exception as exc:
            error_str = str(exc).lower()
            if "rate limit" in error_str or "429" in error_str:
                raise LLMRateLimitError(
                    "LLM API rate limit exceeded. Please try again shortly."
                ) from exc
            if "context_length" in error_str or "maximum context" in error_str:
                raise LLMContextTooLongError(
                    "The retrieved context is too long for the model. "
                    "Try asking a more specific question."
                ) from exc
            raise LLMError(
                f"LLM call failed: {exc}",
                details={"provider": self.llm_provider, "model": self.model_name},
            ) from exc

        logger.info("Answer generated", answer_length=len(answer_text))

        return {
            "answer": answer_text,
            "sources": sources,
            "model": self.model_name,
            "provider": self.llm_provider,
        }
