from fastapi import APIRouter
from app.api.services.context import ContextService


router = APIRouter()

@router.get("/context/{session_id}")
async def get_context(session_id: str):
    return await ContextService().load_context(session_id)
