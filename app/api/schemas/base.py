from pydantic import BaseModel
from typing import Dict, Any


class APIBase(BaseModel):
    message: str
    details: Dict[str, Any] = {}

class ContextState(APIBase):
    history: list
    world_state: dict
    character_stats: dict
