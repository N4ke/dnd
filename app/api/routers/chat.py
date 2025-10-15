from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.api.services.llm_service import LLMService
from app.api.schemas.chat import ChatRequest, ChatResponse
from app.config.settings import settings


router = APIRouter(prefix="/chat")

@router.websocket("/ws")
async def chat_websocket(websocket: WebSocket):
    await websocket.accept()
    llm_service = LLMService.get_instance(settings)
    
    try:
        while True:
            data = await websocket.receive_json()
            request = ChatRequest(**data)
            response = await llm_service.process_request(request)
            await websocket.send_json(response.model_dump())
    except WebSocketDisconnect:
        pass

@router.post("/ask")
async def chat_ask(request: ChatRequest) -> ChatResponse:
    llm_service = LLMService.get_instance(settings)
    response = await llm_service.process_request(request)
    return response