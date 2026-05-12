import uuid
from typing import Any, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

# from agent import state
from src.agent.state import AgentState, CriticFeedback
from src.api.schemas import SearchPlan
from src.prompts.system import (
    AGENT_SYSTEM_PROMPT,
    CRITIC_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    PROMPT_VERSIONS,
)
from src.prompts.templates import CRITIC_EVALUATION_TEMPLATE


class RAGWorkflow:
    def __init__(self, llm: BaseChatModel, tools: list[BaseTool]):
        self.llm = llm
        self.tools = tools
        self.llm_with_tools = self.llm.bind_tools(self.tools, parallel_tool_calls=False)
        self.critic_llm = self.llm.with_structured_output(CriticFeedback)

    def agent_node(self, state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        """Main Agent node."""
        messages = state["messages"]
        configurable = config.get("configurable", {})
        prompt_version_name = configurable.get("prompt_version", "v3_baseline")
        prompt_text = PROMPT_VERSIONS.get(prompt_version_name, AGENT_SYSTEM_PROMPT)
        system_message = SystemMessage(content=prompt_text)
        messages_to_send = [system_message] + messages

        response = self.llm_with_tools.invoke(messages_to_send)

        return {"messages": [response]}

    def critic_node(self, state: AgentState) -> dict[str, Any]:
        """Critic node."""
        messages = state["messages"]
        draft_answer = str(messages[-1].content)
        revisions = state.get("revisions", 0)

        if revisions >= 2:
            return {"approved": True}
        # user query extraction
        user_queries = [
            m.content
            for m in messages
            if isinstance(m, HumanMessage) and "CRITIC_FEEDBACK:" not in str(m.content)
        ]
        original_question = user_queries[-1] if user_queries else "Unknown Query"

        context_msgs = [m for m in messages if getattr(m, "type", "") == "tool"]
        context = (
            "\n\n".join(str(m.content) for m in context_msgs)
            if context_msgs
            else "No external context was retrieved."
        )

        user_prompt = CRITIC_EVALUATION_TEMPLATE.format(
            question=original_question, context=context, draft_answer=draft_answer
        )

        raw_feedback = self.critic_llm.invoke(
            [
                SystemMessage(content=CRITIC_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
        )
        feedback = cast(CriticFeedback, raw_feedback)

        if feedback.approved:
            return {"approved": True, "revisions": revisions}

        feedback_msg = HumanMessage(
            content=f"CRITIC_FEEDBACK: Your previous answer was rejected.\n"
            f"Identified Issues: {feedback.issues}\n"
            f"Action Required: Rewrite the answer to directly address the user's "
            f"question, strictly using ONLY the provided context. Do not hallucinate."
        )

        return {
            "messages": [feedback_msg],
            "approved": False,
            "revisions": revisions + 1,
        }

    def planner_node(self, state: AgentState) -> dict[str, Any]:
        """Intercepts the question and decomposes it into parallel tool calls."""
        messages = state["messages"]
        user_query = messages[-1].content

        # Bind the structured output schema to the LLM
        planner_llm = self.llm.with_structured_output(SearchPlan)

        # get the decomposed plan
        plan = cast(
            SearchPlan,
            planner_llm.invoke(
                [
                    SystemMessage(content=PLANNER_SYSTEM_PROMPT),
                    HumanMessage(content=str(user_query)),
                ]
            ),
        )

        # a standard Tool Call list
        tool_calls = []
        for q in plan.queries:
            tool_calls.append(
                {
                    "name": "search_sec_reports",
                    "args": q.model_dump(exclude_none=True),
                    "id": str(uuid.uuid4()),
                }
            )

        # create an AIMessage pretending the Agent requested these tools
        ai_msg = AIMessage(content="", tool_calls=tool_calls)

        return {"messages": [ai_msg]}

    @staticmethod
    def route_start(state: AgentState, config: RunnableConfig) -> str:
        """Dynamically routes to the planner or directly to
        the agent based on config.
        """
        use_planner = config.get("configurable", {}).get("use_planner", False)

        if use_planner:
            return "planner"
        return "agent"

    @staticmethod
    def should_continue_agent(state: AgentState, config: RunnableConfig) -> str:
        last_message = state["messages"][-1]

        if getattr(last_message, "tool_calls", None):
            return "tools"

        use_critic = config.get("configurable", {}).get("use_critic", True)

        if use_critic:
            return "critic"

        return END

    @staticmethod
    def should_continue_critic(state: AgentState) -> str:
        if state.get("approved"):
            return END
        return "agent"

    def compile(self) -> CompiledStateGraph[Any, Any, Any]:
        """Compile the LangGraph."""
        workflow = StateGraph(AgentState)
        workflow.add_node("planner", self.planner_node)
        workflow.add_node("agent", self.agent_node)
        workflow.add_node("tools", ToolNode(self.tools))
        workflow.add_node("critic", self.critic_node)

        workflow.set_conditional_entry_point(
            self.route_start, {"planner": "planner", "agent": "agent"}
        )

        workflow.add_edge("planner", "tools")

        workflow.add_conditional_edges(
            "agent",
            self.should_continue_agent,
            {"tools": "tools", "critic": "critic", END: END},
        )
        workflow.add_edge("tools", "agent")
        workflow.add_conditional_edges(
            "critic", self.should_continue_critic, {"agent": "agent", END: END}
        )

        return workflow.compile()
