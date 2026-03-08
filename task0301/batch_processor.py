# task0301/batch_processor.py
# 作用：基于 ThreadPoolExecutor 的并行批量 RAG 生成处理
#   - 并发数根据 CPU 核心数自动调整
#   - 单任务失败不影响其他任务
#   - 保证输入输出顺序一致
#   - 集成缓存：先查缓存，命中则跳过 LLM 调用

import os
import time
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class BatchProcessor:
    """
    并行批量处理器。

    Args:
        generate_fn:    生成函数，签名 fn(query: str) -> Dict[str, Any]
        max_workers:    最大线程数（None 则自动根据 CPU 核心数设定）
        cache:          可选的 GenerationCache 实例，若提供则优先查缓存
        timeout:        单个任务最大等待时间（秒），None 表示不限
    """

    def __init__(
        self,
        generate_fn: Callable[[str], Dict[str, Any]],
        max_workers: Optional[int] = None,
        cache=None,
        timeout: Optional[float] = None,
    ):
        self.generate_fn = generate_fn
        self.timeout = timeout
        self.cache = cache

        # 自动确定线程数：不超过 CPU 逻辑核心数，但至少为 1
        cpu_count = os.cpu_count() or 1
        if max_workers is None:
            # 对于 I/O 密集型（LLM API 调用）可以适当放大；这里取核数
            self.max_workers = min(cpu_count, 8)
        else:
            self.max_workers = max(1, max_workers)

        logger.info(f"BatchProcessor 初始化: max_workers={self.max_workers}")

    def _run_single(self, query: str) -> Dict[str, Any]:
        """
        执行单个任务：先查缓存，命中则直接返回；否则调用 generate_fn。
        异常时返回带 error 字段的占位结果。
        """
        # 1) 查缓存
        if self.cache is not None:
            cache_key = self.cache.make_key(query)
            cached = self.cache.get(cache_key)
            if cached is not None:
                logger.info(f"[CACHE HIT] query='{query[:60]}'")
                return {**cached, "_cache_hit": True}

        # 2) 调用 LLM 生成
        try:
            t0 = time.time()
            result = self.generate_fn(query)
            elapsed = round(time.time() - t0, 2)
            result["_elapsed"] = elapsed
            result["_cache_hit"] = False

            # 3) 写缓存（低温度结果）
            if self.cache is not None:
                # 从生成结果或 pipeline 配置中获取 temperature
                temperature = result.get("_temperature", 0.0)
                cache_key = self.cache.make_key(query)
                self.cache.set(cache_key, result, temperature=temperature)

            return result

        except Exception as e:
            logger.error(f"[TASK ERROR] query='{query[:60]}'\n{traceback.format_exc()}")
            return {
                "query":    query,
                "answer":   "",
                "error":    str(e),
                "_cache_hit": False,
                "_elapsed": 0.0,
            }

    def run(self, queries: List[str]) -> List[Dict[str, Any]]:
        """
        并行处理一批 queries，保证返回结果顺序与输入一致。

        Returns:
            List[Dict]，每条对应一个 query 的生成结果（含 _elapsed, _cache_hit, error 字段）
        """
        if not queries:
            return []

        n = len(queries)
        logger.info(f"BatchProcessor.run: {n} queries, max_workers={self.max_workers}")

        results: List[Optional[Dict[str, Any]]] = [None] * n

        # 用 index -> future 映射，保证顺序
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx: Dict[Future, int] = {
                executor.submit(self._run_single, q): i
                for i, q in enumerate(queries)
            }

            for future in as_completed(future_to_idx, timeout=self.timeout):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    # future.result() 本身抛异常（理论上 _run_single 内部已捕获）
                    logger.error(f"[FUTURE ERROR] idx={idx}: {e}")
                    results[idx] = {
                        "query":    queries[idx],
                        "answer":   "",
                        "error":    str(e),
                        "_cache_hit": False,
                        "_elapsed": 0.0,
                    }

        # 兜底：填充未完成的任务（timeout 导致）
        for i in range(n):
            if results[i] is None:
                results[i] = {
                    "query":    queries[i],
                    "answer":   "",
                    "error":    "timeout",
                    "_cache_hit": False,
                    "_elapsed": 0.0,
                }

        return results

    def print_summary(self, results: List[Dict[str, Any]]):
        """打印批量处理汇总"""
        print("\n" + "=" * 60)
        print(f"批量处理汇总 ({len(results)} 条)")
        print("=" * 60)
        total_time = 0.0
        cache_hits = 0
        errors = 0

        for i, r in enumerate(results, start=1):
            elapsed = r.get("_elapsed", 0.0)
            hit = r.get("_cache_hit", False)
            err = r.get("error", "")
            ans_preview = (r.get("answer") or "")[:80].replace("\n", " ")

            total_time += elapsed
            if hit:
                cache_hits += 1
            if err:
                errors += 1

            flag = "[CACHE]" if hit else ("[ERROR]" if err else "[OK]   ")
            print(f"  #{i:02d} {flag} {elapsed:.1f}s | {r.get('query', '')[:55]}")
            if err:
                print(f"       Error: {err[:80]}")
            elif ans_preview:
                print(f"       Answer: {ans_preview}...")

        print("-" * 60)
        print(f"  总耗时（串行等价）: {total_time:.1f}s  |  Cache命中: {cache_hits}/{len(results)}  |  错误: {errors}")
        print("=" * 60)


# ==========================================
# 简单自测
# ==========================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    call_count = [0]   # 用列表避免闭包问题

    def fake_generate(query: str) -> Dict[str, Any]:
        """模拟生成函数（延迟 0.5s）"""
        call_count[0] += 1
        time.sleep(0.5)
        if "fail" in query.lower():
            raise ValueError("模拟失败！")
        return {"query": query, "answer": f"答案: {query[:20]}...", "_temperature": 0.3}

    from cache_manager import GenerationCache
    cache = GenerationCache(maxsize=100, ttl_seconds=60, temp_threshold=0.4)

    processor = BatchProcessor(generate_fn=fake_generate, max_workers=4, cache=cache)

    queries = [
        "What is metformin?",
        "Does aspirin reduce MI risk?",
        "What are statin benefits?",
        "fail this query",          # 故意触发异常
        "What is metformin?",       # 重复 → 应命中缓存
    ]

    t_start = time.time()
    results = processor.run(queries)
    t_end = time.time()

    processor.print_summary(results)
    print(f"\n实际总耗时（并行）: {t_end - t_start:.2f}s（串行预计 ≥{len(queries)*0.5:.1f}s）")
    print(f"LLM 实际调用次数: {call_count[0]}（应为 4，第5条命中缓存）")
    print("\n缓存状态:", cache.stats())
