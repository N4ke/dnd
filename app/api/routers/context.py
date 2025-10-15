from fastapi import APIRouter, Request
from app.api.services.context import ContextService


router = APIRouter()

@router.get("/context/{session_id}")
async def get_context(session_id: str, request: Request):
    context_service = ContextService(request.app.state.redis)
    return await context_service.load_context(session_id)
