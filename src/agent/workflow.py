from typing import Any, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from src.agent.state import AgentState, CriticFeedback
from src.prompts.system import CRITIC_SYSTEM_PROMPT


class RAGWorkflow:
    def __init__(self, llm: BaseChatModel, tools: list[BaseTool]):
        self.llm = llm
        self.tools = tools
        # Предварительно связываем LLM с тулзами и создаем LLM для критика
        self.llm_with_tools = self.llm.bind_tools(self.tools, parallel_tool_calls=False)
        self.critic_llm = self.llm.with_structured_output(CriticFeedback)

    def agent_node(self, state: AgentState) -> dict[str, Any]:
        """Узел основного Агента."""
        response = self.llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def critic_node(self, state: AgentState) -> dict[str, Any]:
        """Узел Критика (Reflexion pattern)."""
        messages = state["messages"]
        draft_answer = str(messages[-1].content)
        revisions = state.get("revisions", 0)

        if revisions >= 2:
            return {"approved": True}

        context_msgs = [m for m in messages if getattr(m, "type", "") == "tool"]
        context = (
            "\n".join(str(m.content) for m in context_msgs)
            if context_msgs
            else "Поиск в базе не производился."
        )

        user_prompt = f"КОНТЕКСТ:\n{context}\n\nОТВЕТ АГЕНТА:\n{draft_answer}"

        raw_feedback = self.critic_llm.invoke(
            [
                SystemMessage(content=CRITIC_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
        )
        feedback = cast(CriticFeedback, raw_feedback)

        if feedback.approved:
            return {"approved": True, "revisions": revisions + 1}

        feedback_msg = HumanMessage(
            content=f"КРИТИК ОТКЛОНИЛ ОТВЕТ. Ошибки: {feedback.issues}. "
            f"Перепиши ответ, используя строго предоставленный контекст."
        )
        return {
            "messages": [feedback_msg],
            "approved": False,
            "revisions": revisions + 1,
        }

    @staticmethod
    def should_continue_agent(state: AgentState) -> str:
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "tools"
        return "critic"

    @staticmethod
    def should_continue_critic(state: AgentState) -> str:
        if state.get("approved"):
            return END
        return "agent"

    def compile(self) -> CompiledStateGraph[Any, Any, Any]:
        """Сборка графа."""
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", self.agent_node)
        workflow.add_node("tools", ToolNode(self.tools))
        workflow.add_node("critic", self.critic_node)

        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent", self.should_continue_agent, {"tools": "tools", "critic": "critic"}
        )
        workflow.add_edge("tools", "agent")
        workflow.add_conditional_edges(
            "critic", self.should_continue_critic, {"agent": "agent", END: END}
        )

        return workflow.compile()
