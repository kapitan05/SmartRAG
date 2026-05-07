from typing import List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    user_id: str
    query: str


class ChatResponse(BaseModel):
    answer: str
    revisions_needed: int


class SECSearchSchema(BaseModel):
    query: str = Field(
        ...,
        description="Semantic query for vector search (e.g., 'reasons for "
        "revenue decline', 'supply chain risks'). Do NOT include years, quarters, "
        "or company tickers in this string.",
    )
    ticker: Optional[List[str]] = Field(
        default=None,
        description="List of company tickers to filter by. Valid examples:"
        " ['MSFT', 'NVDA']. Leave as None if no specific company is mentioned.",
    )
    years: Optional[List[int]] = Field(
        default=None,
        description="List of years to filter by (e.g., [2022, 2023])."
        "Leave as None if the year is not specified or not relevant.",
    )
    quarters: Optional[List[str]] = Field(
        default=None,
        description="List of quarters to filter by (e.g., "
        "['Q1', 'Q2', 'Q3', 'Q4']). Leave as None if "
        "the quarter is not specified.",
    )
