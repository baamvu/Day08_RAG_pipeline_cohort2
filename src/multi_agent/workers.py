"""
Worker Agents — Each worker handles a specific stage of the RAG pipeline.

Workers:
    1. RetrieverWorker  — semantic + lexical search → RRF merge
    2. RerankerWorker   — cross-encoder reranking
    3. GeneratorWorker  — LLM call with context + citation
"""

import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dotenv import load_dotenv

load_dotenv()


@dataclass
class WorkerResult:
    """Standardized output from a worker."""
    worker_name: str
    success: bool
    data: Any = None
    error: str = ""
    metadata: dict = field(default_factory=dict)


class BaseWorker(ABC):
    """Abstract base class for all workers."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def execute(self, query: str, context: dict = None) -> WorkerResult:
        pass


class RetrieverWorker(BaseWorker):
    """
    Worker 1: Retrieval
    Runs semantic search + lexical search in parallel, merges with RRF.
    """

    @property
    def name(self) -> str:
        return "RetrieverWorker"

    def execute(self, query: str, context: dict = None) -> WorkerResult:
        from src.task5_semantic_search import semantic_search
        from src.task6_lexical_search import lexical_search
        from src.task7_reranking import rerank_rrf

        top_k = context.get("top_k", 10) if context else 10

        try:
            dense_results = semantic_search(query, top_k=top_k)
            sparse_results = lexical_search(query, top_k=top_k)

            non_empty = [r for r in [dense_results, sparse_results] if r]
            if not non_empty:
                return WorkerResult(
                    worker_name=self.name,
                    success=True,
                    data=[],
                    metadata={"dense_count": 0, "sparse_count": 0},
                )

            merged = rerank_rrf(non_empty, top_k=top_k)
            for item in merged:
                item["source"] = "hybrid"

            return WorkerResult(
                worker_name=self.name,
                success=True,
                data=merged,
                metadata={
                    "dense_count": len(dense_results),
                    "sparse_count": len(sparse_results),
                    "merged_count": len(merged),
                },
            )
        except Exception as e:
            return WorkerResult(worker_name=self.name, success=False, error=str(e))


class RerankerWorker(BaseWorker):
    """
    Worker 2: Reranking
    Re-scores and re-orders candidates using cross-encoder.
    """

    @property
    def name(self) -> str:
        return "RerankerWorker"

    def execute(self, query: str, context: dict = None) -> WorkerResult:
        from src.task7_reranking import rerank

        candidates = context.get("candidates", []) if context else []
        top_k = context.get("top_k", 5) if context else 5

        if not candidates:
            return WorkerResult(
                worker_name=self.name,
                success=True,
                data=[],
                metadata={"input_count": 0},
            )

        try:
            reranked = rerank(query, candidates, top_k=top_k)
            for item in reranked:
                if "source" not in item:
                    item["source"] = "hybrid"

            return WorkerResult(
                worker_name=self.name,
                success=True,
                data=reranked,
                metadata={"input_count": len(candidates), "output_count": len(reranked)},
            )
        except Exception as e:
            return WorkerResult(worker_name=self.name, success=False, error=str(e))


class GeneratorWorker(BaseWorker):
    """
    Worker 3: Generation
    Generates answer with citation from retrieved context using LLM.
    """

    @property
    def name(self) -> str:
        return "GeneratorWorker"

    def execute(self, query: str, context: dict = None) -> WorkerResult:
        from src.task10_generation import (
            reorder_for_llm,
            format_context,
            SYSTEM_PROMPT,
            TEMPERATURE,
            TOP_P,
        )

        chunks = context.get("chunks", []) if context else []

        if not chunks:
            return WorkerResult(
                worker_name=self.name,
                success=True,
                data={"answer": "Không tìm thấy thông tin.", "sources": []},
                metadata={"chunks_used": 0},
            )

        try:
            reordered = reorder_for_llm(chunks)
            formatted_context = format_context(reordered)
            user_msg = f"Context:\n{formatted_context}\n\n---\n\nQuestion: {query}"

            from openai import OpenAI

            mimo_key = os.getenv("MIMO_API_KEY", "")
            mimo_base = os.getenv("MIMO_BASE_URL", "")
            mimo_model = os.getenv("MIMO_MODEL", "mimo-v2.5-pro")

            if mimo_key and mimo_base:
                client = OpenAI(api_key=mimo_key, base_url=mimo_base)
                model = mimo_model
            else:
                raise ValueError("No API key configured")

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_tokens=1024,
            )

            answer = response.choices[0].message.content or ""

            return WorkerResult(
                worker_name=self.name,
                success=True,
                data={"answer": answer, "sources": chunks},
                metadata={"chunks_used": len(chunks), "model": model},
            )
        except Exception as e:
            return WorkerResult(worker_name=self.name, success=False, error=str(e))
