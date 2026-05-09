import uuid
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from src.agent.builder import build_rag_graph

_cached_graph = None


def get_eval_graph() -> Any:
    global _cached_graph
    if _cached_graph is None:
        _cached_graph = build_rag_graph()
    return _cached_graph


def make_rag_eval_wrapper(config_overrides: dict[str, Any]) -> Any:
    """
    FACTORY FUNCTION: This wraps the evaluation logic and bakes in the
    experiment's specific configurations (e.g., retriever_k, use_planner, prompt_version).
    """

    async def rag_eval_wrapper(inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Wrapper for running the RAG evaluation graph.
        This function will be called by LangSmith's evaluation framework.
        It takes a question as input, runs the RAG graph,
        and extracts both the final answer and the retrieved documents for evaluation.
        """
        graph_eval = get_eval_graph()

        question: str = inputs["question"]
        # unique thread_id for LangSmith
        thread_id = str(uuid.uuid4())
        merged_config = {**config_overrides, "thread_id": thread_id}
        config: RunnableConfig = {"configurable": merged_config}

        # graph state with initial question
        initial_state = {
            "user_id": "evaluator_bot",
            "query": question,
            "messages": [HumanMessage(content=question)],
        }

        result_state = await graph_eval.ainvoke(initial_state, config=config)

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

    return rag_eval_wrapper
