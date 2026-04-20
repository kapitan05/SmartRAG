from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class CriticFeedback(BaseModel):
    """Structured output for the Critic Agent."""

    approved: bool = Field(
        description="Одобрен ли ответ (True если нет выдуманных фактов)."
    )
    issues: list[str] = Field(
        default_factory=list, description="Список найденных ошибок."
    )


class AgentState(TypedDict):
    """The State of our LangGraph."""

    messages: Annotated[list[Any], add_messages]
    approved: bool
    revisions: int
