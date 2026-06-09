"""
Supervisor Agent — Orchestrates the RAG pipeline using workers.

The Supervisor:
    1. Receives user query
    2. Plans execution (decides which workers to call)
    3. Delegates to RetrieverWorker → RerankerWorker → GeneratorWorker
    4. Handles fallback logic (PageIndex if hybrid results are poor)
    5. Aggregates and returns final result

Execution Flow:
    Query → Supervisor.plan()
               │
               ├→ RetrieverWorker.execute(query)
               │      → semantic + lexical → RRF merge → candidates
               │
               ├→ RerankerWorker.execute(query, candidates)
               │      → cross-encoder rerank → top_k results
               │
               ├→ [If score < threshold] PageIndex fallback
               │
               └→ GeneratorWorker.execute(query, chunks)
                      → LLM call → answer with citation
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.multi_agent.workers import (
    BaseWorker,
    RetrieverWorker,
    RerankerWorker,
    GeneratorWorker,
    WorkerResult,
)


@dataclass
class PipelineResult:
    """Final result from the Supervisor pipeline."""
    answer: str
    sources: list
    retrieval_source: str  # "hybrid" or "pageindex"
    worker_trace: list = field(default_factory=list)  # execution log


class Supervisor:
    """
    Supervisor Agent — orchestrates the multi-agent RAG pipeline.

    Manages a team of specialized workers and decides the execution plan
    based on the query and intermediate results.
    """

    def __init__(
        self,
        top_k: int = 5,
        score_threshold: float = 0.3,
        use_reranking: bool = True,
    ):
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.use_reranking = use_reranking

        # Initialize workers
        self.retriever = RetrieverWorker()
        self.reranker = RerankerWorker()
        self.generator = GeneratorWorker()

    def plan(self, query: str) -> list[str]:
        """
        Plan the execution steps based on the query.
        Returns list of worker names to execute in order.
        """
        steps = ["RetrieverWorker"]
        if self.use_reranking:
            steps.append("RerankerWorker")
        steps.append("GeneratorWorker")
        return steps

    def _run_pageindex_fallback(self, query: str) -> list[dict]:
        """Run PageIndex as fallback when hybrid results are poor."""
        try:
            from src.task8_pageindex_vectorless import pageindex_search
            return pageindex_search(query, top_k=self.top_k)
        except Exception:
            return []

    def run(self, query: str) -> PipelineResult:
        """
        Execute the full RAG pipeline using workers.

        Steps:
            1. RetrieverWorker: semantic + lexical → RRF merge
            2. RerankerWorker: cross-encoder rerank (if enabled)
            3. Check threshold → PageIndex fallback if needed
            4. GeneratorWorker: LLM generation with context
        """
        trace = []

        # Step 1: Retrieve
        plan = self.plan(query)
        trace.append({"step": "plan", "workers": plan})

        retriever_result = self.retriever.execute(
            query, context={"top_k": self.top_k * 2}
        )
        trace.append({
            "step": "retrieve",
            "worker": retriever_result.worker_name,
            "success": retriever_result.success,
            "metadata": retriever_result.metadata,
        })

        candidates = retriever_result.data if retriever_result.success else []

        # Step 2: Rerank
        if self.use_reranking and candidates:
            reranker_result = self.reranker.execute(
                query, context={"candidates": candidates, "top_k": self.top_k}
            )
            trace.append({
                "step": "rerank",
                "worker": reranker_result.worker_name,
                "success": reranker_result.success,
                "metadata": reranker_result.metadata,
            })
            candidates = reranker_result.data if reranker_result.success else candidates

        # Step 3: Check threshold → PageIndex fallback
        retrieval_source = "hybrid"
        best_score = candidates[0]["score"] if candidates else 0

        if not candidates or best_score < self.score_threshold:
            fallback_results = self._run_pageindex_fallback(query)
            trace.append({
                "step": "fallback",
                "worker": "PageIndex",
                "triggered": True,
                "reason": f"score {best_score:.3f} < threshold {self.score_threshold}",
                "results_count": len(fallback_results),
            })
            if fallback_results:
                candidates = fallback_results
                retrieval_source = "pageindex"
        else:
            trace.append({
                "step": "fallback",
                "worker": "PageIndex",
                "triggered": False,
                "reason": f"score {best_score:.3f} >= threshold {self.score_threshold}",
            })

        # Step 4: Generate
        generator_result = self.generator.execute(
            query, context={"chunks": candidates}
        )
        trace.append({
            "step": "generate",
            "worker": generator_result.worker_name,
            "success": generator_result.success,
            "metadata": generator_result.metadata,
        })

        if generator_result.success:
            return PipelineResult(
                answer=generator_result.data["answer"],
                sources=generator_result.data["sources"],
                retrieval_source=retrieval_source,
                worker_trace=trace,
            )
        else:
            return PipelineResult(
                answer=f"Generation failed: {generator_result.error}",
                sources=candidates,
                retrieval_source=retrieval_source,
                worker_trace=trace,
            )


if __name__ == "__main__":
    supervisor = Supervisor(top_k=5, use_reranking=True)

    queries = [
        "Hình phạt cho tội tàng trữ ma tuý?",
        "Châu Việt Cường bị xử lý như nào?",
    ]

    for q in queries:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        print("=" * 60)
        result = supervisor.run(q)
        print(f"A: {result.answer[:200]}...")
        print(f"Sources: {len(result.sources)} | Via: {result.retrieval_source}")
        print(f"Trace: {len(result.worker_trace)} steps")
        for step in result.worker_trace:
            print(f"  - {step['step']}: {step.get('worker', '?')} ({step.get('success', step.get('triggered', '?'))})")
