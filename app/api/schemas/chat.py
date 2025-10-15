from app.api.schemas.base import APIBase


class ChatRequest(APIBase):
    message: str
    session_id: str
    temperature: float = 0.7

class ChatResponse(APIBase):
    content: str
    is_final: bool = False