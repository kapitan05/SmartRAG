from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


def make_sec_search_tool(
    qdrant_client: QdrantClient, embeddings: Embeddings
) -> BaseTool:

    vector_store = QdrantVectorStore(
        client=qdrant_client, collection_name="sec_reports", embedding=embeddings
    )

    # ВАЖНО: Указываем, что инструмент возвращает и текст, и артефакт
    @tool(response_format="content_and_artifact")
    async def search_sec_reports(
        query: str,
        config: RunnableConfig,  # <- LangChain сам передаст сюда конфиг!
    ) -> tuple[str, list[Any]]:
        """Используй этот инструмент для поиска финансовой информации и рисков в отчетах SEC 10-K."""

        # 1. Извлекаем наш динамический k (по умолчанию 4, если не передали)
        k = config.get("configurable", {}).get("retriever_k", 4)

        # 2. Ищем k документов
        docs = await vector_store.asimilarity_search(query, k=k)

        context_texts = []
        for doc in docs:
            text = doc.page_content
            source_file = doc.metadata.get("source", "Unknown")
            context_texts.append(f"{text}\n\n[SOURCE: {source_file}]")

        # 3. Это строка, которую будет читать LLM-агент
        llm_content = "\n\n---\n\n".join(context_texts)

        # 4. Возвращаем кортеж: (Текст для LLM, Сырые документы для msg.artifact)
        return llm_content, docs

    return search_sec_reports
