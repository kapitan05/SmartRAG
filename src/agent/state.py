from enum import Enum
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class CriticDecision(str, Enum):
    APPROVED = "approved"
    REVISE = "revise"
    RESEARCH = "research"


class CriticFeedback(BaseModel):
    decision: CriticDecision = Field(description="The routing decision for the agent.")
    issues: list[str] = Field(description="Specific issues found. Empty if approved.")


class AgentState(TypedDict):
    """The State of our LangGraph."""

    messages: Annotated[list[BaseMessage], add_messages]
    revisions: int
    critic_decision: CriticDecision
