from typing import Any, cast

from fastapi import Request
from langgraph.graph.state import CompiledStateGraph
from motor.motor_asyncio import (
    AsyncIOMotorClient,
    AsyncIOMotorCollection,
    AsyncIOMotorDatabase,
)


def get_database(request: Request) -> AsyncIOMotorDatabase[Any]:
    """
    Dependency to get the MongoDB database instance.
    Used for accessing multiple collections (e.g., users, settings).
    """
    client = cast(AsyncIOMotorClient[Any], request.app.state.mongo_client)
    return client["rag_db"]


def get_chat_history_collection(request: Request) -> AsyncIOMotorCollection[Any]:
    """Dependency to get the MongoDB collection for chat history."""
    db = get_database(request)
    return db["chat_history"]


def get_graph(request: Request) -> CompiledStateGraph[Any, Any, Any]:
    """Dependency to get the compiled LangGraph."""
    return cast(CompiledStateGraph[Any, Any, Any], request.app.state.graph)
