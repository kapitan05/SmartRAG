from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from fastapi import Depends, FastAPI, HTTPException
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph.state import CompiledStateGraph
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection
from qdrant_client import QdrantClient

from src.agent.workflow import RAGWorkflow
from src.api.dependencies import get_chat_history_collection, get_graph
from src.api.schemas import ChatRequest, ChatResponse
from src.core.config import settings
from src.prompts.system import AGENT_SYSTEM_PROMPT
from src.tools.sec_search import make_sec_search_tool


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # 1. Инициализация ресурсов
    qdrant_client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    mongo_client: AsyncIOMotorClient[Any] = AsyncIOMotorClient(settings.mongo_uri)
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=settings.openai_api_key,
    )
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,
        api_key=settings.openai_api_key,
    )

    # 2. Сборка графа
    sec_search_tool = make_sec_search_tool(qdrant_client, embeddings)
    workflow = RAGWorkflow(llm=llm, tools=[sec_search_tool])
    compiled_graph = workflow.compile()

    # 3. Сохранение в app.state (Никаких глобальных переменных!)
    app.state.mongo_client = mongo_client
    app.state.graph = compiled_graph

    yield

    # 4. Очистка ресурсов
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
