from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    session_id: str
    temperature: float = 0.7

class ChatResponse(BaseModel):
    content: str
    is_final: bool = False