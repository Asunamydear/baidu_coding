# Task 0216 – RAG 生成模块：上下文组装与提示词工程

---

### question ###

本周任务是在完成多路检索与重排序（Task0202）基础上：

构建 RAG 生成模块第一部分，包括：

1. 上下文组装器（Context Assembler）
2. 医学提示词工程模板（Medical Prompt Engineering）

目标：

- 对检索到的文档进行去重与多样性筛选
- 控制上下文长度，适配 LLM 输入限制
- 设计多阶段医学问答提示词结构

---

## 本周文件结构说明

```
task0216/
│
├── context_builder.py   （主文件）
└── tast.py              （同内容草稿，已重命名）
```

---

## 1. DocumentChunk（数据类）

用于统一表示一个检索到的文档块：

```python
@dataclass
class DocumentChunk:
    text: str
    metadata: Dict[str, Any]
    relevance_score: float
    source: str
    chunk_id: str
```

作用：

- 将来自不同检索路径（vector / bm25）的文档块统一格式
- 携带相关性分数，便于后续排序与筛选

---

## 2. ContextAssembler（上下文组装器）

功能：

- 加载 tokenizer，精准估算 token 数量
- 对检索结果转换格式、去重、排序、截断

核心处理流程：

```
输入文档列表
    ↓
转换为 DocumentChunk
    ↓
按 relevance_score 排序
    ↓
Jaccard 相似度去重（阈值 0.8）
    ↓
多样性控制（同来源最多选 3 块）
    ↓
按 max_context_tokens 截断
（在后 10% 文本中寻找句号截断）
    ↓
返回 context_text + metadata + selected_chunks
```

Jaccard 相似度计算示例：

```python
intersection = len(set1 & set2)
union = len(set1 | set2)
similarity = intersection / union
```

输出结构：

```python
{
    "context_text": "...",
    "metadata": {
        "total_chunks_retrieved": 10,
        "unique_chunks_after_dedup": 7,
        "chunks_selected": 5,
        "estimated_tokens": 1394,
        "chunk_sources": {"vector": 3, "bm25": 2}
    },
    "selected_chunks": [...]
}
```

作用：

- 防止重复内容占用上下文窗口
- 保证来源多样性，避免单一文档主导
- 在 token 限制内尽可能保留完整句子

---

## 3. PromptStage（提示词阶段数据类）

```python
@dataclass
class PromptStage:
    name: str
    system_prompt: str
    user_prompt_template: str
    temperature: float
    max_tokens: int
```

---

## 4. MEDICAL_PROMPTS（医学提示词字典）

设计四个阶段，以字典形式存储，支持动态按 key 调用：

```python
MEDICAL_PROMPTS: Dict[str, PromptStage] = {
    "evidence_evaluator": ...,
    "answer_generator": ...,
    "critical_reviewer": ...,
    "final_assembler": ...,
}
```

各阶段说明：

| 阶段               | 名称         | temperature | max_tokens | 作用                                   |
| ------------------ | ------------ | ----------- | ---------- | -------------------------------------- |
| evidence_evaluator | 证据评估器   | 0.1         | 600        | 评估文献相关性，提取核心证据并标注等级 |
| answer_generator   | 答案生成器   | 0.3         | 1024       | 基于证据生成结构化医学回答             |
| critical_reviewer  | 批判性审查器 | 0.0         | 600        | 检查草稿中的幻觉推断和不严谨陈述       |
| final_assembler    | 最终组装器   | 0.2         | 1200       | 整合审查意见，输出最终回答             |

调用方式示例：

```python
stage = MEDICAL_PROMPTS["answer_generator"]
prompt = stage.user_prompt_template.format(question=q, context=ctx)
```

---

## 本周完成内容总结

- 定义 DocumentChunk 数据类，统一文档块格式
- 实现 ContextAssembler，支持去重、多样性控制、截断
- 设计 PromptStage 数据类，规范化各阶段提示词结构
- 实现四阶段医学提示词模板，以字典形式组织

---

## 当前系统状态

```
Task0118: 向量索引构建       完成
Task0125: 查询理解增强       完成
Task0202: 多路检索 + 重排序  完成
Task0216: 上下文组装 + 提示词 完成
```

---

## 下一阶段

接入 LLM 生成模块，完成多阶段生成流水线：

```
上下文 → 证据评估 → 草稿生成 → 批判审查 → 最终答案
```
