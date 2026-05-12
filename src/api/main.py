import time
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Any, AsyncGenerator

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from motor.motor_asyncio import (
    AsyncIOMotorClient,
    AsyncIOMotorCollection,
    AsyncIOMotorDatabase,
)
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

from src.agent.builder import build_rag_graph
from src.api.auth import (
    create_access_token,
    get_current_user,
    get_password_hash,
    verify_password,
)
from src.api.dependencies import get_chat_history_collection, get_database, get_graph
from src.api.schemas import ChatRequest, ChatResponse, Token
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
    current_user: str = Depends(get_current_user),
    collection: AsyncIOMotorCollection[Any] = Depends(get_chat_history_collection),
    graph: CompiledStateGraph[Any, Any, Any] = Depends(get_graph),
) -> ChatResponse:
    start_time = time.time()
    try:
        cursor = collection.find({"user_id": current_user}).sort("_id", -1).limit(5)
        history_docs = await cursor.to_list(length=5)

        messages: list[Any] = [SystemMessage(content=AGENT_SYSTEM_PROMPT)]
        for doc in reversed(history_docs):
            messages.append(HumanMessage(content=doc["query"]))
            messages.append(AIMessage(content=doc["answer"]))

        messages.append(HumanMessage(content=request.query))

        should_use_critic = (
            request.use_critic
            if request.use_critic is not None
            else settings.USE_CRITIC_BY_DEFAULT
        )
        should_use_planner = (
            request.use_planner
            if request.use_planner is not None
            else settings.USE_PLANNER_BY_DEFAULT
        )

        graph_config: RunnableConfig = {
            "configurable": {
                "use_critic": should_use_critic,
                "use_planner": should_use_planner,
                "user_id": current_user,
            }
        }
        initial_state = {
            "messages": messages,
            "approved": False,
            "revisions": 0,
        }

        final_state = await graph.ainvoke(initial_state, config=graph_config)

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
            {"user_id": current_user, "query": request.query, "answer": ai_answer}
        )

        return ChatResponse(answer=ai_answer, revisions_needed=revisions_made)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/chat/context")
async def clear_context(
    current_user: str = Depends(get_current_user),
    collection: AsyncIOMotorCollection[Any] = Depends(get_chat_history_collection),
) -> dict[str, Any]:
    """Delete all chat history for a user to reset context."""
    result = await collection.delete_many({"user_id": current_user})
    return {"status": "success", "deleted_count": result.deleted_count}


@app.get("/api/chat/history")
async def get_history(
    limit: int = 5,
    current_user: str = Depends(get_current_user),
    collection: AsyncIOMotorCollection[Any] = Depends(get_chat_history_collection),
) -> list[dict[str, str]]:
    """Returns the last messages for display when loading the UI."""
    cursor = collection.find({"user_id": current_user}).sort("_id", 1)
    docs = await cursor.to_list(length=limit)
    return [{"query": d["query"], "answer": d["answer"]} for d in docs]


@app.post("/api/register", status_code=status.HTTP_201_CREATED)
async def register_user(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncIOMotorDatabase[Any] = Depends(get_database),
) -> dict[str, str]:
    users_collection = db["users"]
    existing_user = await users_collection.find_one({"username": form_data.username})
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already registered")

    hashed_password = get_password_hash(form_data.password)
    await users_collection.insert_one(
        {"username": form_data.username, "hashed_password": hashed_password}
    )
    return {"msg": "User created successfully"}


@app.post("/api/token", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncIOMotorDatabase[Any] = Depends(get_database),
) -> Token:
    users_collection = db["users"]
    user = await users_collection.find_one({"username": form_data.username})

    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Generate Token
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]},
        expires_delta=access_token_expires,
    )
    return Token(access_token=access_token, token_type="bearer")


# metrics collection
Instrumentator().instrument(app).expose(app)
