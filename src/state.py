# src/state.py
import os, json, time
from typing import Dict, Any

class InMemoryState:
    def __init__(self):
        self._m: Dict[str, Dict[str, Any]] = {}
    def hgetall(self, key: str) -> Dict[str, Any]:
        return self._m.get(key, {}).copy()
    def hmset(self, key: str, data: Dict[str, Any]):
        self._m.setdefault(key, {}).update(data)

def _try_redis():
    url = os.getenv("REDIS_URL") or os.getenv("REDIS_HOST")
    if not url: return None
    try:
        import redis  # pip install redis
        if os.getenv("REDIS_URL"):
            r = redis.from_url(os.getenv("REDIS_URL"), decode_responses=True)
        else:
            r = redis.Redis(host=os.getenv("REDIS_HOST","localhost"),
                            port=int(os.getenv("REDIS_PORT","6379")),
                            decode_responses=True)
        r.ping()
        return r
    except Exception:
        return None

class ConvState:
    """
    Conversation state with Redis if available; else in-memory.
    Stores:
      - user_turn_idx
      - prev_bot_text
      - no_progress_count
      - bot_repeat_count
      - ema_score
      - consecutive_high
    """
    def __init__(self, ttl_seconds: int = 86400):
        self.ttl = ttl_seconds
        self.rdb = _try_redis()
        self.mem = InMemoryState()

    def _key(self, conversation_id: str) -> str:
        return f"conv:{conversation_id}"

    def load(self, conversation_id: str) -> Dict[str, Any]:
        key = self._key(conversation_id)
        if self.rdb:
            data = self.rdb.hgetall(key) or {}
        else:
            data = self.mem.hgetall(key)
        # coerce types and provide defaults
        return {
            "user_turn_idx": int(data.get("user_turn_idx", 0)),
            "prev_bot_text": data.get("prev_bot_text", ""),
            "no_progress_count": float(data.get("no_progress_count", 0.0)),
            "bot_repeat_count": float(data.get("bot_repeat_count", 0.0)),
            "ema_score": float(data.get("ema_score", 0.0)),
            "consecutive_high": int(data.get("consecutive_high", 0)),
        }

    def save(self, conversation_id: str, state: Dict[str, Any]):
        key = self._key(conversation_id)
        payload = {
            "user_turn_idx": int(state.get("user_turn_idx", 0)),
            "prev_bot_text": state.get("prev_bot_text", ""),
            "no_progress_count": float(state.get("no_progress_count", 0.0)),
            "bot_repeat_count": float(state.get("bot_repeat_count", 0.0)),
            "ema_score": float(state.get("ema_score", 0.0)),
            "consecutive_high": int(state.get("consecutive_high", 0)),
            "updated_at": int(time.time()),
        }
        if self.rdb:
            self.rdb.hset(key, mapping={k: str(v) for k,v in payload.items()})
            self.rdb.expire(key, self.ttl)
        else:
            self.mem.hmset(key, payload)
