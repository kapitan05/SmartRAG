from typing import Any, cast

from fastapi import Request
from langgraph.graph.state import CompiledStateGraph
from motor.motor_asyncio import AsyncIOMotorCollection


def get_chat_history_collection(request: Request) -> AsyncIOMotorCollection[Any]:
    """Извлекаем коллекцию MongoDB из состояния приложения."""
    return cast(
        AsyncIOMotorCollection[Any],
        request.app.state.mongo_client["rag_db"]["chat_history"],
    )


def get_graph(request: Request) -> CompiledStateGraph[Any, Any, Any]:
    """Извлекаем скомпилированный граф LangGraph."""
    return cast(CompiledStateGraph[Any, Any, Any], request.app.state.graph)
