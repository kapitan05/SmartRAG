from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class CriticFeedback(BaseModel):
    """Structured output for the Critic Agent."""

    approved: bool = Field(description="Approved or not.")
    issues: list[str] = Field(
        default_factory=list, description="List of identified issues."
    )


class AgentState(TypedDict):
    """The State of our LangGraph."""

    messages: Annotated[list[BaseMessage], add_messages]
    approved: bool
    revisions: int
