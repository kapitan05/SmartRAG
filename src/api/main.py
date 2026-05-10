import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph.state import CompiledStateGraph
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

from src.agent.builder import build_rag_graph
from src.api.dependencies import get_chat_history_collection, get_graph
from src.api.schemas import ChatRequest, ChatResponse
from src.core.config import settings
from src.prompts.system import AGENT_SYSTEM_PROMPT

load_dotenv()

# prometheus+grafana metrics
LLM_TOKENS_USED = Counter(
    "rag_llm_tokens_total",
    "Total tokens consumed by LLM",
    ["model_name"],
)

RAG_GENERATION_TIME = Histogram(
    "rag_generation_duration_seconds", "Time taken for LLM to generate answer"
)

RAG_REVISIONS_COUNT = Histogram(
    "rag_agent_revisions_total",
    "Number of self-correction loops the LangGraph agent needed",
)


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
    start_time = time.time()
    try:
        cursor = collection.find({"user_id": request.user_id}).sort("_id", -1).limit(5)
        history_docs = await cursor.to_list(length=5)

        messages: list[Any] = [SystemMessage(content=AGENT_SYSTEM_PROMPT)]
        for doc in reversed(history_docs):
            messages.append(HumanMessage(content=doc["query"]))
            messages.append(AIMessage(content=doc["answer"]))

        messages.append(HumanMessage(content=request.query))

        # invoke the RAG graph with the full message history and current query
        final_state = await graph.ainvoke(
            {"messages": messages, "approved": False, "revisions": 0}
        )

        ai_message = final_state["messages"][-1]
        ai_answer = str(final_state["messages"][-1].content)
        revisions_made = final_state.get("revisions", 1) - 1
        RAG_REVISIONS_COUNT.observe(revisions_made)

        # grafana metrics
        metadata = getattr(ai_message, "response_metadata", {})
        token_usage = metadata.get("token_usage", {})
        total_tokens = token_usage.get("total_tokens", 0)
        model_name = metadata.get("model", "unknown_model")
        if total_tokens > 0:
            LLM_TOKENS_USED.labels(model_name=model_name).inc(total_tokens)

        # total generation time
        RAG_GENERATION_TIME.observe(time.time() - start_time)

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
    """Delete all chat history for a user to reset context."""
    result = await collection.delete_many({"user_id": user_id})
    return {"status": "success", "deleted_count": result.deleted_count}


@app.get("/api/chat/history/{user_id}")
async def get_history(
    user_id: str,
    limit: int = 5,
    collection: AsyncIOMotorCollection[Any] = Depends(get_chat_history_collection),
) -> list[dict[str, str]]:
    """Returns the last messages for display when loading the UI."""
    cursor = collection.find({"user_id": user_id}).sort("_id", 1)
    docs = await cursor.to_list(length=limit)
    return [{"query": d["query"], "answer": d["answer"]} for d in docs]


# metrics collection
Instrumentator().instrument(app).expose(app)
