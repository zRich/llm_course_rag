import json
import time
import hashlib
import logging
from typing import Any, Callable, Optional

try:
    import redis  # type: ignore
    REDIS_AVAILABLE = True
except Exception:
    REDIS_AVAILABLE = False

from src.config.settings import settings

logger = logging.getLogger(__name__)


def _now() -> float:
    return time.time()


def canonicalize_text(text: str) -> str:
    """规范化文本用于缓存键：去首尾空白、折叠空白、统一大小写。"""
    if not text:
        return ""
    s = " ".join(text.strip().split())
    return s.lower()


def make_key(payload: Any, namespace: Optional[str] = None) -> str:
    """稳定生成缓存键（SHA256）。payload 可为 dict/list/tuple/str。"""
    ns = namespace or getattr(settings, "cache_namespace", "lab03")
    try:
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    except Exception:
        raw = str(payload)
    h = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"{ns}:{h}"


def with_jitter(ttl: int) -> int:
    ratio = float(getattr(settings, "cache_jitter_ratio", 0.15))
    jitter = int(ttl * ratio)
    # ±jitter 范围，避免雪崩
    return max(1, ttl - jitter + int(jitter * (hashlib.sha256(str(_now()).encode()).digest()[0] / 255)))


class MemoryCache:
    def __init__(self):
        self.store: dict[str, tuple[float, Any]] = {}

    def get(self, key: str) -> Optional[Any]:
        item = self.store.get(key)
        if not item:
            return None
        exp, val = item
        if exp < _now():
            # 过期清理
            self.store.pop(key, None)
            return None
        return val

    def set(self, key: str, value: Any, ttl: int) -> None:
        self.store[key] = (_now() + with_jitter(int(ttl)), value)

    def get_or_set(self, key: str, ttl: int, supplier: Callable[[], Any]) -> Any:
        val = self.get(key)
        if val is not None:
            logger.info(f"cache_hit memory key={key[:16]}...")
            return val
        val = supplier()
        self.set(key, val, ttl)
        logger.info(f"cache_set memory key={key[:16]}... ttl={ttl}")
        return val


class RedisCache:
    def __init__(self):
        if not REDIS_AVAILABLE:
            raise RuntimeError("redis not available")
        self.client = redis.StrictRedis(
            host=getattr(settings, "redis_host", "127.0.0.1"),
            port=int(getattr(settings, "redis_port", 6379)),
            db=int(getattr(settings, "redis_db", 0)),
            password=getattr(settings, "redis_password", None),
            decode_responses=True,
        )

    def get(self, key: str) -> Optional[Any]:
        v = self.client.get(key)
        if v is None:
            return None
        try:
            return json.loads(v)
        except Exception:
            return v

    def set(self, key: str, value: Any, ttl: int) -> None:
        ttl_j = with_jitter(int(ttl))
        try:
            payload = json.dumps(value, ensure_ascii=False)
        except Exception:
            payload = str(value)
        self.client.setex(key, ttl_j, payload)

    def get_or_set(self, key: str, ttl: int, supplier: Callable[[], Any]) -> Any:
        val = self.get(key)
        if val is not None:
            logger.info(f"cache_hit redis key={key[:16]}...")
            return val
        val = supplier()
        self.set(key, val, ttl)
        logger.info(f"cache_set redis key={key[:16]}... ttl={ttl}")
        return val


def get_cache():
    if not getattr(settings, "cache_enabled", True):
        return MemoryCache()  # 关闭时返回内存，但不命中（由调用方决定是否绕过）
    provider = getattr(settings, "cache_provider", "memory").lower()
    if provider == "redis" and REDIS_AVAILABLE:
        try:
            return RedisCache()
        except Exception as e:
            logger.warning(f"RedisCache 初始化失败，回退内存: {e}")
            return MemoryCache()
    return MemoryCache()