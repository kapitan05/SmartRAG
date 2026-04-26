# src/agent/builder.py
from typing import Any

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph.state import CompiledStateGraph
from qdrant_client import QdrantClient  # Импортируй свой класс графа

from src.agent.workflow import RAGWorkflow
from src.core.config import settings
from src.tools.sec_search import make_sec_search_tool


def build_rag_graph() -> CompiledStateGraph[Any, Any, Any, Any]:
    """
    Единая точка сборки (Factory) для нашего графа.
    Гарантирует, что API и тесты используют идентичные настройки.
    """
    # data base + models + tools
    qdrant_client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small", api_key=settings.openai_api_key
    )
    llm = ChatOpenAI(
        model="gpt-4o-mini", temperature=0.0, api_key=settings.openai_api_key
    )

    sec_search_tool = make_sec_search_tool(qdrant_client, embeddings)

    # graph workflow
    workflow = RAGWorkflow(llm=llm, tools=[sec_search_tool])
    return workflow.compile()
