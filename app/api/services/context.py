from redis.asyncio import Redis
import json
from app.config.settings import settings


class ContextService:
    def __init__(self):
        self.redis = Redis.from_url(settings.REDIS_URL)
    
    async def save_context(self, session_id: str, history: list):
        await self.redis.setex(
            f"context:{session_id}",
            settings.SESSION_TTL,
            str(history)
        )
    
    async def load_context(self, session_id: str) -> list:
        data = await self.redis.get(f"context:{session_id}")
        return json.loads(data.decode()) if data else []