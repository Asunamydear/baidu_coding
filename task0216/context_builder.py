import os
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from transformers import AutoTokenizer

# ==========================================
# 第一部分：上下文组装器
# ==========================================

@dataclass
class DocumentChunk:
    """文档块数据类"""
    text: str
    metadata: Dict[str, Any]
    relevance_score: float
    source: str
    chunk_id: str

class ContextAssembler:
    def __init__(self, tokenizer_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", max_context_tokens: int = 4000):
        # 1. 加载tokenizer
        # 使用快速加载方式加载模型的分词器用于控制输入给LLM的窗口大小
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        self.max_context_tokens = max_context_tokens

    def estimate_tokens(self, text: str) -> int:
        """2. 估算文本token数量"""
        if not text:
            return 0
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _calculate_jaccard_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似性（Jaccard相似性），用于去重"""
        set1 = set(text1.lower().split())
        set2 = set(text2.lower().split())
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0

    def _analyze_sources(self, chunks: List[DocumentChunk]) -> Dict[str, int]:
        """统计不同数据源的命中次数"""
        sources = {}
        for chunk in chunks:
            sources[chunk.source] = sources.get(chunk.source, 0) + 1
        return sources

    def _truncate_text(self, text: str, max_tokens: int) -> str:
        """从后向前寻找句子截断点，实现在完整段落处截断（如后10%文本中找到句点）"""
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) <= max_tokens:
            return text
            
        truncated_tokens = tokens[:max_tokens]
        truncated_text = self.tokenizer.decode(truncated_tokens)
        
        # 尝试在后10%文本中寻找句号或句点（确保在完整段落/句子处截断）
        search_len = max(10, int(len(truncated_text) * 0.1))
        tail = truncated_text[-search_len:]
        
        # 匹配英文或中文句号
        match = re.search(r'[.。](?!.*[.。])', tail)
        if match:
            cut_idx = len(truncated_text) - search_len + match.end()
            return truncated_text[:cut_idx]
            
        return truncated_text

    def assemble_context(self, retrieved_docs: List[Dict[str, Any]], similarity_threshold: float = 0.8) -> Dict[str, Any]:
        """
        3. 组装上下文核心逻辑
        """
        # (1) 转换文档格式（Dict -> DocumentChunk)
        chunks = []
        for doc in retrieved_docs:
            chunks.append(DocumentChunk(
                text=doc.get("text", doc.get("text_head", "")),
                metadata=doc.get("metadata", {}),
                relevance_score=float(doc.get("rerank_score", doc.get("fusion_score", 0.0))),
                source=doc.get("source", "unknown"),
                chunk_id=str(doc.get("doc_id", doc.get("uid", "unknown")))
            ))
            
        # (2) 按照相关性排序 (从高到低)
        chunks.sort(key=lambda x: x.relevance_score, reverse=True)

        # (3) 计算文本相似性并去重，同时优先选择高相关性但不密集的来源（考虑多样化）
        unique_chunks = []
        source_counts = {}
        max_per_source = 3  # 多样性策略：假设同一来源最多选用3块，以降低同质化
        
        for chunk in chunks:
            # 考虑多样化（如果已经有太多来自同一来源的文档，直接拒绝或降级）
            if source_counts.get(chunk.source, 0) >= max_per_source:
                continue

            # Jaccard 相似度去重（若和已选中的块非常相似，则跳过）
            is_dup = False
            for selected in unique_chunks:
                sim = self._calculate_jaccard_similarity(chunk.text, selected.text)
                if sim > similarity_threshold:
                    is_dup = True
                    break
            
            if not is_dup:
                unique_chunks.append(chunk)
                source_counts[chunk.source] = source_counts.get(chunk.source, 0) + 1

        # (4) 构建上下文字符串并检查长度限制，确保即使被断开也是在段落终止处
        selected_chunks = []
        final_context_parts = []
        current_tokens = 0

        for idx, chunk in enumerate(unique_chunks):
            chunk_str = f"[Doc {idx+1} | Source: {chunk.source}]\n{chunk.text}\n"
            chunk_tokens = self.estimate_tokens(chunk_str)

            if current_tokens + chunk_tokens > self.max_context_tokens:
                # 若剩余容量仅够一点点，则用特定算法找到最后句点对它截断保留
                remaining_tokens = self.max_context_tokens - current_tokens
                if remaining_tokens > 50:  # 大于50tokens才有截断保留的价值
                    truncated_str = self._truncate_text(chunk_str, remaining_tokens)
                    final_context_parts.append(truncated_str)
                    selected_chunks.append(chunk)
                break
                
            final_context_parts.append(chunk_str)
            selected_chunks.append(chunk)
            current_tokens += chunk_tokens

        final_context = "\n".join(final_context_parts)

        # (5) 补充元数据
        context_metadata = {
            "total_chunks_retrieved": len(retrieved_docs),
            "unique_chunks_after_dedup": len(unique_chunks),
            "chunks_selected": len(selected_chunks),
            "estimated_tokens": self.estimate_tokens(final_context),
            "chunk_sources": self._analyze_sources(selected_chunks)
        }
        
        # 返回最终组装结果
        return {
            "context_text": final_context,
            "metadata": context_metadata,
            "selected_chunks": selected_chunks
        }


# ==========================================
# 第二部分：医学提示词工程模板
# ==========================================

@dataclass
class PromptStage:
    name: str
    system_prompt: str
    user_prompt_template: str
    temperature: float
    max_tokens: int

MEDICAL_PROMPTS: Dict[str, PromptStage] = {

    # 阶段 1：证据评估器
    "evidence_evaluator": PromptStage(
        name="证据评估器",
        system_prompt=(
            "你是一个严谨的临床医学文献评估助手。你需要评估提供的文献片段与用户问题的相关性，"
            "并提取其中直接支持、反驳或补充该问题的核心医学事实（如RCT研究设计、干预手段、患者结局等）。"
            "请保持客观并忽略一切无关信息；若找不到任何相关证据，请直接明确声明没有任何直接证据。"
        ),
        user_prompt_template=(
            "【用户问题】\n{question}\n\n"
            "【检索到的候选文献片段】\n{context}\n\n"
            "任务：请评估上述文献，提取关键的医学证据事实，并为每条提取的证据打标签评定证据等级（强/中/弱）。"
        ),
        temperature=0.1,
        max_tokens=600
    ),

    # 阶段 2：答案生成器
    "answer_generator": PromptStage(
        name="答案生成器",
        system_prompt=(
            "你是一位专业的数字化临床医疗顾问。请基于下面提供并经过核实的医学证据，为用户的问题生成准确、"
            "清晰、以证据为基础的回答。您必须在此回答中直接引用医学事实的具体文献来源（如[Doc 1]）。"
        ),
        user_prompt_template=(
            "【相关核实医学证据】\n{context}\n\n"
            "【用户问题】：{question}\n\n"
            "请遵循以下结构来撰写回答：\n"
            "1. 核心结论（简明扼要的一段话）\n"
            "2. 详细分析（说明研究背景、论证过程与支撑证据，多引用文献）\n"
            "3. 临床建议或注意事项（如果证据不足导致无法回答，请明确告知风险）"
        ),
        temperature=0.3,
        max_tokens=1024
    ),

    # 阶段 3：批判性审查器
    "critical_reviewer": PromptStage(
        name="批判性审查器",
        system_prompt=(
            "你的身份是一个极其严格的医学同行评审人（Peer Reviewer）。你需要对草拟的医学回答进行批判性审查，"
            "核心检查该回答是否做出了“未被支持的越界推断/幻觉推断”、夸大功效、缺乏引用标记，或者隐瞒了安全性风险。"
        ),
        user_prompt_template=(
            "【原始用户问题】\n{question}\n\n"
            "【基础支撑证据】\n{context}\n\n"
            "【草稿回答（待审查）】\n{draft_answer}\n\n"
            "任务：审查草稿回答。请列举出该草稿不严谨、没有证据支撑、或具备误导性的陈述，并给出修改意见。"
            "如果不存任何需要修改的地方，只需完全回复“通过”。"
        ),
        temperature=0.0,
        max_tokens=600
    ),

    # 阶段 4：最终组装器
    "final_assembler": PromptStage(
        name="最终组装器",
        system_prompt=(
            "你是一个医学出版级别的最终编辑。你需要根据“草稿回答”与相应的“审查意见”，修订并优化出一份完美连贯的最终回答。"
            "排版要易读、清晰、格式专业，确保面向具有医学背景的读者时不出专业纰漏，面向普通读者的部分易懂易读。"
        ),
        user_prompt_template=(
            "【同行评审反馈意见】\n{review_feedback}\n\n"
            "【原始草稿回答】\n{draft_answer}\n\n"
            "任务：请整合审查意见来纠正草稿中的问题，并输出最终的循证医学回答。保留相关引用序号标记。"
        ),
        temperature=0.2,
        max_tokens=1200
    ),
}

if __name__ == "__main__":
    # 简单测试代码（供后续调试）
    print("上下文组装器与提示词模块加载成功。")
