from pydantic import BaseModel


class ChatRequest(BaseModel):
    user_id: str
    query: str


class ChatResponse(BaseModel):
    answer: str
    revisions_needed: int
