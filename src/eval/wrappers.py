from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from src.agent.builder import build_rag_graph

eval_graph = build_rag_graph()


async def rag_eval_wrapper(inputs: dict[str, Any]) -> dict[str, Any]:
    """
    Wrapper for running the RAG evaluation graph.
    This function will be called by LangSmith's evaluation framework.
    It takes a question as input, runs the RAG graph,
    and extracts both the final answer and the retrieved documents for evaluation.
    """
    question: str = inputs["question"]

    # unique thread_id for LangSmith
    thread_id = "eval_run_thread"
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    # graph state with initial question
    initial_state = {
        "user_id": "evaluator_bot",
        "query": question,
        "messages": [HumanMessage(content=question)],
    }

    result_state = await eval_graph.ainvoke(initial_state, config=config)

    messages = result_state.get("messages", [])

    retrieved_texts = []
    for msg in messages:
        if getattr(msg, "type", "") == "tool":
            retrieved_texts.append(str(msg.content))

    if "answer" in result_state:
        final_answer = result_state["answer"]
    elif messages:
        final_answer = messages[-1].content
    else:
        final_answer = "Error: Could not extract answer from state."

    return {
        "answer": final_answer,
        "retrieved_docs": retrieved_texts,
    }
