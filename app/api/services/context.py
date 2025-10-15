from redis.asyncio import Redis
import json
from app.config.settings import settings


class ContextService:
    def __init__(self, redis: Redis):
        self.redis = redis
    
    async def save_context(self, session_id: str, history: list):
        await self.redis.setex(
            f"context:{session_id}",
            settings.SESSION_TTL,
            json.dumps(history, ensure_ascii=False)
        )
    
    async def load_context(self, session_id: str) -> list:
        data = await self.redis.get(f"context:{session_id}")
        return json.loads(data) if data else []