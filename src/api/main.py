from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph.state import CompiledStateGraph
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection
from qdrant_client import QdrantClient

from src.agent.builder import build_rag_graph
from src.agent.workflow import RAGWorkflow
from src.api.dependencies import get_chat_history_collection, get_graph
from src.api.schemas import ChatRequest, ChatResponse
from src.core.config import settings
from src.prompts.system import AGENT_SYSTEM_PROMPT
from src.tools.sec_search import make_sec_search_tool

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:

    mongo_client: AsyncIOMotorClient[Any] = AsyncIOMotorClient(settings.mongo_uri)
    app.state.mongo_client = mongo_client

    app.state.graph = build_rag_graph()
    yield

    mongo_client.close()


app = FastAPI(lifespan=lifespan, title="Enterprise RAG Copilot")


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    collection: AsyncIOMotorCollection[Any] = Depends(get_chat_history_collection),
    graph: CompiledStateGraph[Any, Any, Any] = Depends(get_graph),
) -> ChatResponse:
    try:
        cursor = collection.find({"user_id": request.user_id}).sort("_id", -1).limit(5)
        history_docs = await cursor.to_list(length=5)

        messages: list[Any] = [SystemMessage(content=AGENT_SYSTEM_PROMPT)]
        for doc in reversed(history_docs):
            messages.append(HumanMessage(content=doc["query"]))
            messages.append(AIMessage(content=doc["answer"]))

        messages.append(HumanMessage(content=request.query))

        # Используем инжектированный граф
        final_state = await graph.ainvoke(
            {"messages": messages, "approved": False, "revisions": 0}
        )

        ai_answer = str(final_state["messages"][-1].content)
        revisions_made = final_state.get("revisions", 1) - 1

        await collection.insert_one(
            {"user_id": request.user_id, "query": request.query, "answer": ai_answer}
        )

        return ChatResponse(answer=ai_answer, revisions_needed=revisions_made)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/chat/context/{user_id}")
async def clear_context(
    user_id: str,
    collection: AsyncIOMotorCollection[Any] = Depends(get_chat_history_collection),
) -> dict[str, Any]:
    """Удаляет всю историю диалога для конкретного пользователя."""
    result = await collection.delete_many({"user_id": user_id})
    return {"status": "success", "deleted_count": result.deleted_count}


@app.get("/api/chat/history/{user_id}")
async def get_history(
    user_id: str,
    limit: int = 5,
    collection: AsyncIOMotorCollection[Any] = Depends(get_chat_history_collection),
) -> list[dict[str, str]]:
    """Возвращает последние сообщения для отображения при загрузке UI."""
    cursor = collection.find({"user_id": user_id}).sort("_id", 1)
    docs = await cursor.to_list(length=limit)
    return [{"query": d["query"], "answer": d["answer"]} for d in docs]
