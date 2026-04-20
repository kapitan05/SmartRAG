from langchain_core.embeddings import Embeddings
from langchain_core.tools import BaseTool, tool
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


def make_sec_search_tool(
    qdrant_client: QdrantClient, embeddings: Embeddings
) -> BaseTool:
    """Фабрика для создания инструмента поиска."""

    # Создаем удобную обертку LangChain над сырым клиентом Qdrant
    vector_store = QdrantVectorStore(
        client=qdrant_client, collection_name="sec_reports", embedding=embeddings
    )

    @tool
    def search_sec_reports(query: str) -> str:
        """Используй этот инструмент для поиска финансовой информации и рисков в отчетах SEC 10-K."""

        # LangChain сам переведет query в вектор и сделает правильный поиск!
        docs = vector_store.similarity_search(query, k=4)

        context_texts = []
        for doc in docs:
            # LangChain сам парсит payload, отдавая нам готовый объект Document
            company = doc.metadata.get("company", "Unknown")
            text = doc.page_content
            context_texts.append(f"[Отчет: {company}]\n{text}")

        return "\n\n---\n\n".join(context_texts)

    return search_sec_reports
