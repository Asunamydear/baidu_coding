# task0224/generation_pipeline.py
# 作用：生成模块第二部分
#   1. LLMGenerator  - 本地 Ollama LLM 集成（单次/批量生成，JSON 提取）
#   2. MedicalGenerationPipeline - 完整多阶段生成流水线
#      query -> 上下文组装 -> 证据评估 -> 草稿生成 -> 批判审查 -> 最终答案 -> 后处理
#   3. 测试主入口 - 运行示例 query 并 log 关键指标

import os
import sys
import re
import json
import time
import logging
from typing import Dict, Any, List, Optional

import requests

# ====== 路径配置，保证能 import 同级/上级模块 ======
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # task0224
PROJECT_DIR = os.path.normpath(os.path.join(BASE_DIR, ".."))  # 项目根目录

sys.path.append(PROJECT_DIR)
sys.path.append(os.path.join(PROJECT_DIR, "task0202"))
sys.path.append(os.path.join(PROJECT_DIR, "task0216"))

# 引入已有模块
from task0125.rag_retrieval import process_medical_query
from task0202.multipath_retriever import MultiPathRetriever
from task0202.reranker import SimpleReranker, RERANK_MODEL

# 引入 context_builder（task0216 中我们命名的文件）
# 注意：context_builder.py 和 tast.py 都存在，这里我们选 context_builder
try:
    sys.path.insert(0, os.path.join(PROJECT_DIR, "task0216"))
    from context_builder import ContextAssembler, DocumentChunk, MEDICAL_PROMPTS, PromptStage
except ImportError:
    from tast import ContextAssembler, DocumentChunk, MEDICAL_PROMPTS, PromptStage

# ====== 日志配置 ======
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(BASE_DIR, "generation.log"), encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)


# ==========================================
# 1. 本地 LLM 集成（Ollama）
# ==========================================

class LLMGenerator:
    """
    通过 Ollama REST API 调用本地 LLM 进行生成。
    Args:
        model_name: Ollama 模型名称（如 "deepseek-r1:8b" / "llama3:8b"）
        base_url:   Ollama 服务地址（默认 http://localhost:11434）
        timeout:    请求超时时间（秒）
    """

    def __init__(
        self,
        model_name: str = "deepseek-r1:8b",
        base_url: str = "http://localhost:11434",
        timeout: int = 120
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        # 初始化时测试链接是否可用
        self._test_connection()

    def _test_connection(self):
        """测试 Ollama 服务是否在线"""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                logger.info(f"Ollama 连接成功，可用模型: {models}")
                if self.model_name not in models:
                    logger.warning(f"模型 '{self.model_name}' 未在 Ollama 中找到，请先运行: ollama pull {self.model_name}")
            else:
                logger.warning(f"Ollama 响应异常: {resp.status_code}")
        except Exception as e:
            logger.warning(f"Ollama 连接失败（{e}），请确认服务已启动：ollama serve")

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """从文本中提取 JSON 部分，并尝试修复常见格式错误"""
        # 找最外层的 { ... }
        match = re.search(r'\{[\s\S]*\}', text)
        if not match:
            return None

        raw = match.group(0)

        # 常见修复：补充缺失的右括号
        open_cnt = raw.count('{')
        close_cnt = raw.count('}')
        if open_cnt > close_cnt:
            raw += '}' * (open_cnt - close_cnt)

        # 移除尾部多余逗号（JSON 不允许）
        raw = re.sub(r',\s*([}\]])', r'\1', raw)

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.debug(f"JSON 解析失败，原始文本:\n{raw[:300]}")
            return None

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        require_json: bool = False
    ) -> Dict[str, Any]:
        """
        单次生成。
        使用 prompt（+ system_prompt），按照 temperature/max_tokens 合成请求。
        如果 require_json=True，则在 system 提示中追加 JSON 格式要求，并尝试解析结果。

        返回:
            {
              "success": bool,
              "text": str,            # 原始输出
              "parsed": dict | None,  # 解析到的 JSON（require_json=True 时有效）
              "model": str,
              "elapsed": float
            }
        """
        # 1) 根据模型格式构建系统提示
        sys_content = system_prompt or ""
        if require_json:
            sys_content += (
                "\n\n请将你的回答严格以 JSON 格式输出（不要输出代码块，只输出纯 JSON）。"
                "确保所有键名使用英文双引号，所有字段完整。"
            )

        # 2) 构建 Ollama API 请求体（/api/chat，支持 messages 格式）
        messages = []
        if sys_content.strip():
            messages.append({"role": "system", "content": sys_content.strip()})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }

        start = time.time()
        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout
            )
            elapsed = time.time() - start

            if resp.status_code != 200:
                logger.error(f"LLM API 错误: {resp.status_code} - {resp.text[:200]}")
                return {"success": False, "text": "", "parsed": None, "model": self.model_name, "elapsed": elapsed}

            data = resp.json()
            raw_text = data.get("message", {}).get("content", "")

            parsed = None
            if require_json:
                parsed = self._extract_json(raw_text)
                if parsed is None:
                    logger.warning("JSON 提取失败，将以纯文本返回")

            return {
                "success": True,
                "text": raw_text,
                "parsed": parsed,
                "model": self.model_name,
                "elapsed": round(elapsed, 2)
            }

        except requests.exceptions.Timeout:
            elapsed = time.time() - start
            logger.error(f"LLM 请求超时（{self.timeout}s）")
            return {"success": False, "text": "", "parsed": None, "model": self.model_name, "elapsed": elapsed}
        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"LLM 请求异常: {e}")
            return {"success": False, "text": "", "parsed": None, "model": self.model_name, "elapsed": elapsed}

    def batch_generate(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 512,
        require_json: bool = False
    ) -> List[Dict[str, Any]]:
        """
        批量生成（顺序调用，适合小批量；Ollama 暂不支持真正并发批量）
        """
        results = []
        for i, prompt in enumerate(prompts):
            logger.info(f"批量生成 {i+1}/{len(prompts)} ...")
            result = self.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                require_json=require_json
            )
            results.append(result)
        return results


# ==========================================
# 2. 完整医学生成流水线
# ==========================================

class MedicalGenerationPipeline:
    """
    多阶段 RAG 生成流水线：
      检索 -> 上下文组装 -> 证据评估 -> 草稿生成 -> 批判审查 -> 最终答案 -> 后处理
    """

    def __init__(
        self,
        llm_model: str = "deepseek-r1:8b",
        ollama_url: str = "http://localhost:11434",
        max_context_tokens: int = 3000,
        enable_evaluation: bool = True,
        enable_review: bool = True,
    ):
        # 初始化所有组件
        logger.info("初始化 MedicalGenerationPipeline ...")

        # 上下文组装器
        self.context_assembler = ContextAssembler(max_context_tokens=max_context_tokens)

        # 检索组件
        self.retriever = MultiPathRetriever()
        self.reranker = SimpleReranker(RERANK_MODEL)

        # LLM 生成器
        self.llm = LLMGenerator(model_name=llm_model, base_url=ollama_url)

        # 是否启用可选阶段
        self.enable_evaluation = enable_evaluation
        self.enable_review = enable_review

        logger.info("Pipeline 初始化完成。")

    def _retrieve_docs(self, query_info: Dict[str, Any], top_n: int = 20, top_k: int = 10) -> List[Dict]:
        """执行多路召回 + 重排序，返回 top_k 文档列表"""
        result = self.retriever.retrieve(
            query_info=query_info,
            top_k_vector=max(5, top_n),
            top_k_keyword=max(5, top_n),
            fusion_strategy="rrf"
        )
        fused = result["fused_results"][:top_n]
        reranked = self.reranker.rerank(
            query=query_info["clean_query"],
            candidates=fused,
            top_k=top_k
        )
        return reranked

    def _fill_template(self, template: str, **kwargs) -> str:
        """安全填充提示词模板（忽略缺失的 key）"""
        try:
            return template.format(**kwargs)
        except KeyError as e:
            logger.warning(f"提示词模板缺少变量: {e}")
            return template

    def _filter_context_by_evaluation(
        self,
        context_result: Dict[str, Any],
        evaluation_text: str
    ) -> str:
        """
        基于评估结果筛选上下文，保留被提及的文档块。
        策略：扫描评估文本中出现的 [Doc N] 标记，只保留被引用的块。
        """
        selected = context_result["selected_chunks"]
        original_text = context_result["context_text"]

        # 找出评估中提到的 Doc 编号
        mentioned = set(re.findall(r'\[Doc\s*(\d+)\]', evaluation_text, re.IGNORECASE))

        if not mentioned:
            # 评估没有引用任何文档，回退到原始上下文
            return original_text

        # 重新按被评估提及的序号过滤
        kept_parts = []
        for idx, chunk in enumerate(selected, start=1):
            if str(idx) in mentioned:
                kept_parts.append(f"[Doc {idx} | Source: {chunk.source}]\n{chunk.text}\n")

        if not kept_parts:
            return original_text

        return "\n".join(kept_parts)

    def _post_process(self, text: str, selected_chunks: List) -> str:
        """生成后处理：确保段落格式整洁"""
        if not text:
            return text
        # 段落美化：单换行变双换行
        text = re.sub(r'(?<!\n)\n(?!\n)', '\n\n', text.strip())
        return text

    def _format_sources(self, selected_chunks: List) -> List[Dict[str, Any]]:
        """将选中的文档块格式化为来源引用列表"""
        sources = []
        for i, chunk in enumerate(selected_chunks, start=1):
            sources.append({
                "rank": i,
                "doc_id": chunk.chunk_id,
                "source": chunk.source,
                "relevance_score": round(chunk.relevance_score, 4),
                "text_snippet": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text
            })
        return sources

    def generate(self, query: str, top_n: int = 20, top_k: int = 10) -> Dict[str, Any]:
        """
        主生成入口，执行完整多阶段 RAG 流水线。
        """
        pipeline_start = time.time()
        stage_times = {}
        stage_success = {}
        token_counts = {}

        logger.info(f"=== 开始生成流水线 | query: {query[:100]} ===")

        # -------- 阶段 0：Query 增强 --------
        t0 = time.time()
        query_info = process_medical_query(query)
        if not query_info.get("ok"):
            return {"query": query, "answer": f"查询拒绝: {query_info.get('reason')}", "error": True}
        stage_times["query_enhancement"] = round(time.time() - t0, 2)

        # -------- 阶段 1：上下文组装 --------
        t0 = time.time()
        retrieved_docs = self._retrieve_docs(query_info, top_n=top_n, top_k=top_k)
        context_result = self.context_assembler.assemble_context(retrieved_docs)
        context_text = context_result["context_text"]
        stage_times["context_assembly"] = round(time.time() - t0, 2)
        token_counts["context"] = context_result["metadata"]["estimated_tokens"]
        logger.info(f"上下文组装完成: {context_result['metadata']['chunks_selected']} 块, "
                    f"{token_counts['context']} tokens")

        # -------- 阶段 2：检索结果评估（可选）--------
        evidence_evaluation = ""
        if self.enable_evaluation:
            t0 = time.time()
            stage = MEDICAL_PROMPTS["evidence_evaluator"]
            user_prompt = self._fill_template(
                stage.user_prompt_template, question=query, context=context_text
            )
            eval_result = self.llm.generate(
                prompt=user_prompt,
                system_prompt=stage.system_prompt,
                temperature=stage.temperature,
                max_tokens=stage.max_tokens
            )
            stage_times["evidence_evaluation"] = round(time.time() - t0, 2)
            stage_success["evidence_evaluation"] = eval_result["success"]
            evidence_evaluation = eval_result["text"] if eval_result["success"] else ""

            # 使用评估结果筛选上下文（基于文档引用标记过滤）
            if evidence_evaluation:
                context_text = self._filter_context_by_evaluation(context_result, evidence_evaluation)
                logger.info(f"评估后上下文长度: {len(context_text)} chars")

        # -------- 阶段 3：草稿生成 --------
        t0 = time.time()
        stage = MEDICAL_PROMPTS["answer_generator"]
        user_prompt = self._fill_template(
            stage.user_prompt_template, question=query, context=context_text
        )
        draft_result = self.llm.generate(
            prompt=user_prompt,
            system_prompt=stage.system_prompt,
            temperature=stage.temperature,
            max_tokens=stage.max_tokens
        )
        stage_times["draft_generation"] = round(time.time() - t0, 2)
        stage_success["draft_generation"] = draft_result["success"]
        draft_answer = draft_result["text"] if draft_result["success"] else ""
        token_counts["draft"] = len(draft_answer.split())
        logger.info(f"草稿生成完成 (耗时 {stage_times['draft_generation']}s)")

        # -------- 阶段 4：批判性审查（可选）--------
        review_feedback = ""
        if self.enable_review and draft_answer:
            t0 = time.time()
            stage = MEDICAL_PROMPTS["critical_reviewer"]
            user_prompt = self._fill_template(
                stage.user_prompt_template,
                question=query,
                context=context_text,
                draft_answer=draft_answer
            )
            review_result = self.llm.generate(
                prompt=user_prompt,
                system_prompt=stage.system_prompt,
                temperature=stage.temperature,
                max_tokens=stage.max_tokens
            )
            stage_times["critical_review"] = round(time.time() - t0, 2)
            stage_success["critical_review"] = review_result["success"]
            review_feedback = review_result["text"] if review_result["success"] else ""
            logger.info(f"批判性审查完成 (耗时 {stage_times['critical_review']}s)")

        # -------- 阶段 5：最终答案生成 --------
        t0 = time.time()
        stage = MEDICAL_PROMPTS["final_assembler"]

        # 如果成功完成审查，使用审查意见；否则直接用草稿
        if review_feedback and review_feedback.strip() != "通过":
            user_prompt = self._fill_template(
                stage.user_prompt_template,
                review_feedback=review_feedback,
                draft_answer=draft_answer
            )
            final_result = self.llm.generate(
                prompt=user_prompt,
                system_prompt=stage.system_prompt,
                temperature=stage.temperature,
                max_tokens=stage.max_tokens
            )
            final_answer_raw = final_result["text"] if final_result["success"] else draft_answer
            stage_success["final_assembly"] = final_result.get("success", False)
        else:
            # 审查通过或未启用审查，直接使用草稿
            final_answer_raw = draft_answer
            stage_success["final_assembly"] = True

        stage_times["final_assembly"] = round(time.time() - t0, 2)

        # -------- 阶段 6：后处理 --------
        final_answer = self._post_process(final_answer_raw, context_result["selected_chunks"])
        token_counts["final_answer"] = len(final_answer.split())

        # -------- 计算总时间并组装结果 --------
        total_time = round(time.time() - pipeline_start, 2)

        result = {
            "query": query,
            "answer": final_answer,
            "context_metadata": context_result["metadata"],
            "generation_metrics": {
                "total_time_seconds": total_time,
                "stage_times": stage_times,
                "token_counts": token_counts,
                "stage_success": stage_success,
            },
            "intermediate_results": {
                "evidence_evaluation": evidence_evaluation,
                "draft_answer": draft_answer,
                "review_feedback": review_feedback,
            },
            "sources": self._format_sources(context_result["selected_chunks"]),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        logger.info(f"=== 生成完成 | 总耗时: {total_time}s | 答案长度: {token_counts['final_answer']} words ===")
        return result


# ==========================================
# 3. 测试主入口
# ==========================================

TEST_QUERIES = [
    "What is the effect of metformin on cardiovascular outcomes in type 2 diabetes patients?",
    "Does aspirin reduce the risk of myocardial infarction in high-risk patients?",
    "What are the benefits of statins in patients with coronary artery disease?",
]


def print_result(result: Dict[str, Any]):
    """打印生成结果"""
    print("\n" + "-" * 70)
    print(f"Query: {result['query']}")
    print(f"Time: {result['timestamp']}")
    print("-" * 70)

    metrics = result.get("generation_metrics", {})
    print(f"\n[生成指标]")
    print(f"  总耗时: {metrics.get('total_time_seconds', '-')}s")
    print(f"  各阶段: {metrics.get('stage_times', {})}")
    print(f"  token数: {metrics.get('token_counts', {})}")
    print(f"  各阶段是否成功: {metrics.get('stage_success', {})}")

    ctx_meta = result.get("context_metadata", {})
    print(f"\n[上下文]")
    print(f"  检索到: {ctx_meta.get('total_chunks_retrieved', '-')} 块")
    print(f"  去重后: {ctx_meta.get('unique_chunks_after_dedup', '-')} 块")
    print(f"  选入: {ctx_meta.get('chunks_selected', '-')} 块")
    print(f"  估算 tokens: {ctx_meta.get('estimated_tokens', '-')}")
    print(f"  来源: {ctx_meta.get('chunk_sources', {})}")

    print(f"\n[引用来源 Top 3]")
    for src in result.get("sources", [])[:3]:
        print(f"  Doc {src['rank']}: doc_id={src['doc_id']}, score={src['relevance_score']}")
        print(f"    {src['text_snippet'][:120]}")

    inter = result.get("intermediate_results", {})
    if inter.get("evidence_evaluation"):
        print(f"\n[证据评估]")
        print(inter["evidence_evaluation"][:400])
    if inter.get("review_feedback"):
        print(f"\n[审查意见]")
        print(inter["review_feedback"][:400])

    print(f"\n[最终答案]")
    print(result.get("answer", "（无答案）"))
    print("-" * 70)


def main():
    print("== Medical RAG Generation Pipeline ==")
    print(f"Rerank model: {RERANK_MODEL}")

    # 初始化流水线（可在这里调整 enable_evaluation / enable_review）
    pipeline = MedicalGenerationPipeline(
        llm_model="deepseek-r1:8b",
        ollama_url="http://localhost:11434",
        max_context_tokens=3000,
        enable_evaluation=True,
        enable_review=True,
    )

    # 选择运行模式：交互 or 批量测试
    mode = input("\n运行模式（input=输入模式 / test=批量测试，默认 input）: ").strip().lower()

    if mode == "test":
        # 批量跑预设 query
        all_metrics = []
        for q in TEST_QUERIES:
            logger.info(f"\n--- 测试 Query: {q} ---")
            result = pipeline.generate(q)
            print_result(result)

            # 记录关键指标
            m = result.get("generation_metrics", {})
            all_metrics.append({
                "query": q[:60],
                "total_time_s": m.get("total_time_seconds"),
                "answer_words": m.get("token_counts", {}).get("final_answer"),
                "stage_success": m.get("stage_success"),
            })

        print("\n\n===== 批量测试汇总 =====")
        for row in all_metrics:
            print(f"  [{row['total_time_s']}s] [{row['answer_words']} words] {row['query']}")
            print(f"  stages: {row['stage_success']}")

        # 保存结果 JSON
        save_path = os.path.join(BASE_DIR, "test_results.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, ensure_ascii=False, indent=2)
        print(f"\n指标已保存至: {save_path}")

    else:
        # 交互模式：用户手动输入 query
        while True:
            query = input("\n输入医学问题（q 退出）: ").strip()
            if not query or query.lower() == "q":
                break
            result = pipeline.generate(query)
            print_result(result)


if __name__ == "__main__":
    main()
