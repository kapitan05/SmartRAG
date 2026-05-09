import asyncio
from typing import Any, List, Optional

from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool
from langchain_qdrant import (
    FastEmbedSparse,
    QdrantVectorStore,
    RetrievalMode,
)
from qdrant_client import QdrantClient, models

from src.api.schemas import SECSearchSchema

# Global lock to prevent CPU/RAM spikes on the cloud server
# when the Planner calls this tool multiple times in parallel.
SEARCH_LOCK = asyncio.Lock()


def make_sec_search_tool(
    qdrant_client: QdrantClient,
    embeddings: Embeddings,
    sparse_embeddings: FastEmbedSparse,
) -> BaseTool:

    @tool(args_schema=SECSearchSchema, response_format="content_and_artifact")
    async def search_sec_reports(
        query: str,
        ticker: Optional[List[str]] = None,
        years: Optional[List[int]] = None,
        quarters: Optional[List[str]] = None,
        config: RunnableConfig = None,  # type: ignore
    ) -> tuple[str, list[Any]]:
        """Use this tool to search for financial information, risks,
        and metrics in SEC 10-K/10-Q reports.
        """

        config_safe = config or {}

        k = config_safe.get("configurable", {}).get("retriever_k", 4)
        search_algorithm = config_safe.get("configurable", {}).get(
            "search_algorithm", "hybrid"
        )
        collection = config_safe["configurable"].get("collection_name", "sec_reports")

        if search_algorithm == "dense":
            mode = RetrievalMode.DENSE
        elif search_algorithm == "bm25":
            mode = RetrievalMode.SPARSE
        else:
            mode = RetrievalMode.HYBRID

        # Qdrant Metadata Filters
        must_conditions: List[models.Condition] = []

        if ticker:
            must_conditions.append(
                models.FieldCondition(
                    key="metadata.ticker", match=models.MatchAny(any=ticker)
                )
            )
        if years:
            string_years = [str(y) for y in years]
            must_conditions.append(
                models.FieldCondition(
                    key="metadata.year", match=models.MatchAny(any=string_years)
                )
            )
        if quarters:
            must_conditions.append(
                models.FieldCondition(
                    key="metadata.quarter", match=models.MatchAny(any=quarters)
                )
            )

        qdrant_filter: Optional[models.Filter] = (
            models.Filter(must=must_conditions) if must_conditions else None
        )

        # Init vector store
        # (this does NOT do any network calls, just sets up the wrapper around Qdrant)
        vector_store = QdrantVectorStore(
            client=qdrant_client,
            collection_name=collection,
            embedding=embeddings,
            sparse_embedding=sparse_embeddings,
            sparse_vector_name="bm25",
            retrieval_mode=mode,
        )

        async with SEARCH_LOCK:
            docs = await vector_store.asimilarity_search(
                query, k=k, filter=qdrant_filter
            )

        if not docs:
            return "No documents found matching the applied filters.", []

        context_texts = []
        for doc in docs:
            text = doc.page_content
            meta = doc.metadata
            fallback_source = (
                f"{meta.get('year', '')} {meta.get('quarter', '')}"
                f" {meta.get('ticker', '')}.pdf"
            ).strip()
            source_file = meta.get("source", fallback_source)
            context_texts.append(f"{text}\n\n[SOURCE: {source_file}]")

        # llm content is the concatenation of all retrieved contexts,
        #  separated by --- for clarity.
        llm_content = "\n\n---\n\n".join(context_texts)

        # 4. Возвращаем кортеж: (Текст для LLM, Сырые документы для msg.artifact)
        return llm_content, docs

    return search_sec_reports
