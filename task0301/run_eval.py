# task0301/run_eval.py
# 作用：集成验证脚本
#   1. 单次生成 + 答案评估
#   2. 缓存命中测试
#   3. 批量并行处理（含汇总）
#
# 用法：
#   cd d:\learn\steamcode\coding
#   conda activate med_rag
#   python task0301/run_eval.py

import os
import sys
import json
import time
import logging

# ====== 路径配置 ======
BASE_DIR = os.path.dirname(os.path.abspath(__file__))       # task0301
PROJECT_DIR = os.path.normpath(os.path.join(BASE_DIR, "..")) # 项目根目录

sys.path.insert(0, BASE_DIR)
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "task0202"))
sys.path.insert(0, os.path.join(PROJECT_DIR, "task0216"))

# ====== 日志 ======
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(BASE_DIR, "run_eval.log"), encoding="utf-8"),
    ]
)
logger = logging.getLogger(__name__)

# ====== 引入本周新模块 ======
from answer_evaluator import AnswerEvaluator
from cache_manager import GenerationCache
from batch_processor import BatchProcessor

# ====== 引入已有 Pipeline ======
from task0224.generation_pipeline import MedicalGenerationPipeline, TEST_QUERIES

# ====== 参考答案（Ground Truth）======
# 双语参考答案：同时包含英文和中文关键词
# 这样无论 LLM 用英文还是中文回答，关键信息匹配都能产生有效召回
REFERENCE_ANSWERS = {
    TEST_QUERIES[0]: (
        # 英文部分
        "Metformin reduces cardiovascular risk in type 2 diabetes mellitus patients by approximately 33% "
        "(95% CI). Standard dose is 500-2000mg per day for 6-12 months. "
        "Mechanism: AMPK pathway activation. Side effects include GI adverse events such as nausea. "
        # 中文部分（确保与中文生成答案能产生交集）
        "二甲双胍可降低2型糖尿病患者约33%的心血管风险。"
        "推荐剂量500毫克至2000毫克，每日服用，持续6个月至12个月。"
        "机制：激活AMPK通路（AMPK pathway）。"
        "副作用包括胃肠道不良反应（adverse events）。"
    ),
    TEST_QUERIES[1]: (
        "Aspirin reduces the risk of myocardial infarction (MI) by 25-30% in high-risk patients. "
        "Recommended dose is 75-100mg per day long-term. "
        "Risk of adverse events includes gastrointestinal bleeding. "
        "阿司匹林可降低高风险患者心肌梗死风险约25%至30%。"
        "推荐剂量100毫克每日一次，长期服用。"
        "副作用包括胃肠道出血，需评估获益与风险。"
    ),
    TEST_QUERIES[2]: (
        "Statins reduce LDL cholesterol and cardiovascular mortality by 20-35% in coronary artery disease patients. "
        "Mechanism: HMG-CoA reductase inhibition. "
        "High-intensity therapy is recommended long-term. Side effects include myopathy. "
        "他汀类药物可降低冠心病患者心血管死亡率约20%至35%。"
        "机制：抑制HMG-CoA还原酶（HMG-CoA reductase inhibition）。"
        "推荐长期高强度治疗方案。副作用包括肌肉毒性（myopathy）。"
    ),
}


# ==========================================
# 全局初始化（只初始化一次，避免重复加载模型）
# ==========================================

def build_pipeline() -> MedicalGenerationPipeline:
    logger.info("初始化 MedicalGenerationPipeline ...")
    return MedicalGenerationPipeline(
        llm_model="deepseek-r1:8b",
        ollama_url="http://localhost:11434",
        max_context_tokens=3000,
        enable_evaluation=True,
        enable_review=True,
    )


# ==========================================
# Part 1：单次生成 + 评估
# ==========================================

def demo_single_with_eval(pipeline: MedicalGenerationPipeline, evaluator: AnswerEvaluator):
    print("\n" + "=" * 70)
    print("【Part 1】单次生成 + 答案评估")
    print("=" * 70)

    query = TEST_QUERIES[0]
    logger.info(f"Query: {query}")

    # 生成
    result = pipeline.generate(query)
    answer = result.get("answer", "")

    print(f"\nQuery: {query}")
    print(f"\n[生成答案] ({len(answer.split())} words)")
    print(answer[:600] + ("..." if len(answer) > 600 else ""))

    # 评估（传入参考答案，启用 ROUGE + 关键信息召回 + 幻觉检测）
    reference = REFERENCE_ANSWERS.get(query, "")
    eval_result = evaluator.evaluate(prediction=answer, reference=reference)
    evaluator.print_report(eval_result, query=query)

    # 若有第一个指标，保存评估结果到文件
    save_path = os.path.join(BASE_DIR, "eval_single.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump({"query": query, "answer": answer, "evaluation": eval_result}, f,
                  ensure_ascii=False, indent=2)
    logger.info(f"评估结果已保存: {save_path}")

    return answer


# ==========================================
# Part 2：缓存命中测试
# ==========================================

def demo_cache(pipeline: MedicalGenerationPipeline, cache: GenerationCache):
    print("\n" + "=" * 70)
    print("【Part 2】缓存命中测试")
    print("=" * 70)

    query = TEST_QUERIES[0]

    def cached_generate(q: str):
        """带缓存的生成包装函数"""
        key = cache.make_key(q)
        cached = cache.get(key)
        if cached is not None:
            logger.info(f"[CACHE HIT] '{q[:60]}'")
            return {**cached, "_cache_hit": True}

        result = pipeline.generate(q)
        result["_temperature"] = 0.3   # pipeline 使用混合温度，这里视为低温度结果
        ok = cache.set(key, result, temperature=0.3)
        result["_cache_hit"] = False
        logger.info(f"[CACHE SET] ok={ok}, query='{q[:60]}'")
        return result

    # 第一次调用（未命中）
    print("\n--- 第1次调用（预期：MISS）---")
    t0 = time.time()
    r1 = cached_generate(query)
    t1 = time.time()
    print(f"[{'CACHE HIT' if r1.get('_cache_hit') else 'CACHE MISS'}] 耗时: {t1 - t0:.2f}s")

    # 第二次调用（应命中）
    print("\n--- 第2次调用（预期：HIT）---")
    t0 = time.time()
    r2 = cached_generate(query)
    t1 = time.time()
    print(f"[{'CACHE HIT' if r2.get('_cache_hit') else 'CACHE MISS'}] 耗时: {t1 - t0:.4f}s")

    print("\n缓存统计:")
    stats = cache.stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")


# ==========================================
# Part 3：批量并行处理
# ==========================================

def demo_batch(pipeline: MedicalGenerationPipeline,
               cache: GenerationCache,
               evaluator: AnswerEvaluator):
    print("\n" + "=" * 70)
    print("【Part 3】批量并行处理")
    print("=" * 70)

    def generate_fn(query: str):
        result = pipeline.generate(query)
        result["_temperature"] = 0.3
        return result

    processor = BatchProcessor(
        generate_fn=generate_fn,
        max_workers=2,    # 保守设置，避免 Ollama 负载过高
        cache=cache,
    )

    queries = TEST_QUERIES  # 3 条预设 query

    print(f"\n处理 {len(queries)} 条 queries（max_workers={processor.max_workers}）...")
    t_start = time.time()
    results = processor.run(queries)
    t_total = time.time() - t_start

    # 打印汇总
    processor.print_summary(results)
    print(f"\n实际并行总耗时: {t_total:.2f}s")

    # 对每条结果做评估
    print("\n--- 批量评估结果 ---")
    all_evals = []
    for r in results:
        if r.get("error"):
            print(f"  [SKIP] {r['query'][:60]} — error: {r.get('error')}")
            continue
        ref = REFERENCE_ANSWERS.get(r.get("query", ""), "")
        eval_res = evaluator.evaluate(r.get("answer", ""), reference=ref)
        s = eval_res["summary"]
        print(f"  Query: {r['query'][:60]}")
        print(f"    rouge1={s['rouge1_f']:.4f}  rougeL={s['rougeL_f']:.4f}  "
              f"recall={s['key_info_recall']:.4f}  "
              f"halluc={s['halluc_score']:.4f}({s['halluc_risk']})  "
              f"cache={r.get('_cache_hit', False)}")
        all_evals.append({
            "query":      r.get("query"),
            "evaluation": eval_res,
            "elapsed":    r.get("_elapsed", 0.0),
            "cache_hit":  r.get("_cache_hit", False),
        })

    # 保存批量评估结果
    save_path = os.path.join(BASE_DIR, "eval_batch.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(all_evals, f, ensure_ascii=False, indent=2)
    logger.info(f"批量评估结果已保存: {save_path}")

    print(f"\n缓存最终状态: hits={cache.stats()['hits']}, "
          f"misses={cache.stats()['misses']}, "
          f"size={cache.stats()['size']}")


# ==========================================
# 主入口
# ==========================================

def main():
    print("\n" + "=" * 70)
    print("  task0301 集成验证：评估 + 缓存 + 批量处理")
    print("=" * 70)

    # 初始化共享组件
    pipeline = build_pipeline()
    evaluator = AnswerEvaluator()
    cache = GenerationCache(
        maxsize=200,
        ttl_seconds=3600,    # 1小时有效期
        temp_threshold=0.4,
    )

    # 运行三个演示部分
    try:
        demo_single_with_eval(pipeline, evaluator)
    except Exception as e:
        logger.error(f"Part 1 失败: {e}", exc_info=True)

    try:
        demo_cache(pipeline, cache)
    except Exception as e:
        logger.error(f"Part 2 失败: {e}", exc_info=True)

    try:
        demo_batch(pipeline, cache, evaluator)
    except Exception as e:
        logger.error(f"Part 3 失败: {e}", exc_info=True)

    print("\n✅ run_eval.py 运行完毕。结果文件:")
    print(f"  - {os.path.join(BASE_DIR, 'eval_single.json')}")
    print(f"  - {os.path.join(BASE_DIR, 'eval_batch.json')}")
    print(f"  - {os.path.join(BASE_DIR, 'run_eval.log')}")


if __name__ == "__main__":
    main()
