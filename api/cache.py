"""
推理缓存

实现推理结果的内存缓存
"""

import hashlib
import json
import time
from typing import Optional, Any
from datetime import datetime, timedelta


class InferenceCache:
    """
    推理结果缓存

    使用内存缓存存储推理结果，支持 TTL 过期
    """

    def __init__(self, ttl: int = 3600, max_size: int = 100):
        """
        Args:
            ttl: 缓存过期时间（秒）
            max_size: 最大缓存条目数
        """
        self._cache: dict[str, tuple[Any, float]] = {}
        self._ttl = ttl
        self._max_size = max_size

    def _make_key(self, file_path: str, model_name: str, threshold: float) -> str:
        """生成缓存键"""
        content = f"{file_path}:{model_name}:{threshold}"
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, file_path: str, model_name: str, threshold: float) -> Optional[Any]:
        """
        获取缓存结果

        Args:
            file_path: 文件路径
            model_name: 模型名称
            threshold: 阈值

        Returns:
            缓存结果，如果不存在或已过期返回 None
        """
        key = self._make_key(file_path, model_name, threshold)

        if key not in self._cache:
            return None

        result, timestamp = self._cache[key]

        # 检查是否过期
        if time.time() - timestamp > self._ttl:
            del self._cache[key]
            return None

        return result

    def set(self, file_path: str, model_name: str, threshold: float, result: Any):
        """
        设置缓存

        Args:
            file_path: 文件路径
            model_name: 模型名称
            threshold: 阈值
            result: 推理结果
        """
        # 如果缓存已满，删除最旧的条目
        if len(self._cache) >= self._max_size:
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]

        key = self._make_key(file_path, model_name, threshold)
        self._cache[key] = (result, time.time())

    def invalidate(self, file_path: str = None, model_name: str = None):
        """
        使缓存失效

        Args:
            file_path: 如果指定，只删除该文件的缓存
            model_name: 如果指定，只删除该模型的缓存
        """
        if file_path is None and model_name is None:
            self._cache.clear()
            return

        # 删除匹配的缓存
        keys_to_delete = []
        for key, (result, _) in self._cache.items():
            if file_path and result.get("file_path") == file_path:
                keys_to_delete.append(key)
            if model_name and result.get("model_name") == model_name:
                keys_to_delete.append(key)

        for key in keys_to_delete:
            del self._cache[key]

    def cleanup_expired(self):
        """清理过期缓存"""
        current_time = time.time()
        expired_keys = [
            key for key, (_, timestamp) in self._cache.items()
            if current_time - timestamp > self._ttl
        ]

        for key in expired_keys:
            del self._cache[key]

        return len(expired_keys)

    def get_stats(self) -> dict:
        """获取缓存统计"""
        total = len(self._cache)
        expired = sum(
            1 for _, (_, timestamp) in self._cache.items()
            if time.time() - timestamp > self._ttl
        )

        return {
            "total": total,
            "expired": expired,
            "active": total - expired,
            "max_size": self._max_size,
            "ttl": self._ttl,
        }


# 全局缓存单例
_inference_cache: Optional[InferenceCache] = None


def get_inference_cache() -> InferenceCache:
    """获取推理缓存单例"""
    global _inference_cache

    if _inference_cache is None:
        from .config import get_settings
        settings = get_settings()
        _inference_cache = InferenceCache(ttl=settings.cache_ttl)

    return _inference_cache
