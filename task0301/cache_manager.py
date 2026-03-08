# task0301/cache_manager.py
# 作用：RAG 生成结果的缓存策略
#   - 基于 query + context 的 SHA256 哈希键
#   - LRU 大小限制（OrderedDict 实现）
#   - TTL 时效控制（医学知识不能永久缓存）
#   - 温度过滤（只缓存低温度的确定性结果）

import hashlib
import time
import threading
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple


class GenerationCache:
    """
    内存 LRU + TTL 生成结果缓存。

    Args:
        maxsize:         最大缓存条目数（默认 200）
        ttl_seconds:     缓存有效期（秒）。默认 3600（1小时）
        temp_threshold:  只缓存 temperature <= 此值的结果（默认 0.4）
    """

    def __init__(
        self,
        maxsize: int = 200,
        ttl_seconds: float = 3600.0,
        temp_threshold: float = 0.4,
    ):
        self.maxsize = maxsize
        self.ttl_seconds = ttl_seconds
        self.temp_threshold = temp_threshold

        # OrderedDict 用于 LRU：最近访问的移到末尾，淘汰队首
        # 存储格式：key -> (value, timestamp, temperature)
        self._store: OrderedDict[str, Tuple[Any, float, float]] = OrderedDict()
        self._lock = threading.Lock()

        # 统计信息
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._skipped_temp = 0

    # --------------------------------------------------
    # 缓存键生成
    # --------------------------------------------------

    @staticmethod
    def make_key(query: str, context: str = "") -> str:
        """
        基于查询 + 上下文内容生成 SHA256 缓存键。
        保证相同输入 → 相同 key，不同输入 → 不同 key。
        """
        raw = (query or "").strip() + "\x00" + (context or "").strip()
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    # --------------------------------------------------
    # 内部：过期检查
    # --------------------------------------------------

    def _is_expired(self, timestamp: float) -> bool:
        return (time.time() - timestamp) > self.ttl_seconds

    # --------------------------------------------------
    # 读缓存
    # --------------------------------------------------

    def get(self, key: str) -> Optional[Any]:
        """
        读取缓存。命中且未过期返回 value，否则返回 None。
        命中时将条目移到 OrderedDict 末尾（LRU 更新）。
        """
        with self._lock:
            if key not in self._store:
                self._misses += 1
                return None

            value, timestamp, _ = self._store[key]

            # TTL 检查
            if self._is_expired(timestamp):
                del self._store[key]
                self._misses += 1
                return None

            # LRU：移到末尾
            self._store.move_to_end(key)
            self._hits += 1
            return value

    # --------------------------------------------------
    # 写缓存
    # --------------------------------------------------

    def set(self, key: str, value: Any, temperature: float = 0.0) -> bool:
        """
        写入缓存。
        - 若 temperature > temp_threshold，直接跳过（不缓存高随机性结果）
        - 若已满，淘汰最久未访问的条目（LRU 队首）
        返回是否成功写入。
        """
        # 温度过滤
        if temperature > self.temp_threshold:
            self._skipped_temp += 1
            return False

        with self._lock:
            if key in self._store:
                # 已存在 → 更新并移到末尾
                self._store.move_to_end(key)
                self._store[key] = (value, time.time(), temperature)
                return True

            # 容量检查 → LRU 淘汰
            if len(self._store) >= self.maxsize:
                evicted_key, _ = self._store.popitem(last=False)
                self._evictions += 1

            self._store[key] = (value, time.time(), temperature)
            return True

    # --------------------------------------------------
    # 统计信息
    # --------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        """返回缓存使用统计"""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = round(self._hits / total, 4) if total > 0 else 0.0

            # 扫描过期条目数（不删除）
            now = time.time()
            expired_count = sum(
                1 for (_, ts, _) in self._store.values()
                if (now - ts) > self.ttl_seconds
            )

            return {
                "size":             len(self._store),
                "maxsize":          self.maxsize,
                "hits":             self._hits,
                "misses":           self._misses,
                "hit_rate":         hit_rate,
                "evictions":        self._evictions,
                "skipped_by_temp":  self._skipped_temp,
                "expired_entries":  expired_count,
                "ttl_seconds":      self.ttl_seconds,
                "temp_threshold":   self.temp_threshold,
            }

    # --------------------------------------------------
    # 工具方法
    # --------------------------------------------------

    def clear(self):
        """清空所有缓存"""
        with self._lock:
            self._store.clear()

    def evict_expired(self) -> int:
        """主动清理所有过期条目，返回清理数量"""
        with self._lock:
            expired_keys = [
                k for k, (_, ts, _) in self._store.items()
                if self._is_expired(ts)
            ]
            for k in expired_keys:
                del self._store[k]
            return len(expired_keys)

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"GenerationCache(size={s['size']}/{s['maxsize']}, "
            f"hit_rate={s['hit_rate']:.2%}, "
            f"ttl={s['ttl_seconds']}s, "
            f"temp_threshold={s['temp_threshold']})"
        )


# ==========================================
# 简单自测
# ==========================================

if __name__ == "__main__":
    cache = GenerationCache(maxsize=5, ttl_seconds=5, temp_threshold=0.4)

    key1 = cache.make_key("what is metformin?", "context_abc")
    key2 = cache.make_key("what is metformin?", "context_abc")   # 同一 key
    key3 = cache.make_key("what is aspirin?",   "context_xyz")

    assert key1 == key2, "相同输入应生成相同 key"
    assert key1 != key3, "不同输入应生成不同 key"

    # 写入低温度结果（应成功）
    ok = cache.set(key1, {"answer": "Metformin is a biguanide..."}, temperature=0.3)
    assert ok, "低温度结果应写入成功"

    # 读取（命中）
    val = cache.get(key1)
    assert val is not None, "应命中缓存"
    print("HIT:", val)

    # 写入高温度结果（应跳过）
    ok = cache.set(key3, {"answer": "Aspirin..."}, temperature=0.9)
    assert not ok, "高温度结果不应写入"

    # 未命中
    val2 = cache.get(key3)
    assert val2 is None, "高温度结果未写入，应返回 None"

    # TTL 测试
    import time
    print("等待 6s 测试 TTL 过期...")
    time.sleep(6)
    val3 = cache.get(key1)
    assert val3 is None, "TTL 过期后应返回 None"

    print("所有测试通过！")
    print(cache.stats())
