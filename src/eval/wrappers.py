from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from src.agent.builder import build_rag_graph

eval_graph = build_rag_graph()


async def rag_eval_wrapper(inputs: dict[str, Any]) -> dict[str, Any]:
    """
    Обертка над LangGraph для системы эвалюации LangSmith.
    Принимает вопрос, прогоняет через агента и возвращает ответ + контекст.
    """
    question: str = inputs["question"]

    # Уникальный ID для каждого тестового прогона, чтобы изолировать контекст
    thread_id = "eval_run_thread"
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    # Формируем состояние для LangGraph
    initial_state = {
        "user_id": "evaluator_bot",
        "query": question,
        "messages": [HumanMessage(content=question)],
    }

    # Выполняем граф
    result_state = await eval_graph.ainvoke(initial_state, config=config)

    # ИЗВЛЕКАЕМ ДАННЫЕ ДЛЯ DEEPEVAL
    # Убедись, что твой граф сохраняет найденные документы (например, в result_state["documents"])
    retrieved_docs = result_state.get("documents", [])

    return {
        "answer": result_state["messages"][-1].content
        if "messages" in result_state
        else "No answer found",
        # DeepEval требует массив строк для параметра retrieval_context
        "retrieved_docs": [doc.page_content for doc in retrieved_docs],
    }
