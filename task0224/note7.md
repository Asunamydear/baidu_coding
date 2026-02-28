# Task 0224 – RAG 生成模块：LLM 集成与多阶段生成流水线

---

### question ###

本周任务是在完成上下文组装与提示词工程（Task0216）基础上：

构建 RAG 生成模块第二部分，包括：

1. 本地 LLM 集成（Ollama）
2. 完整多阶段生成流水线
3. 测试主入口与指标日志

目标：

- 将检索结果接入本地 LLM 完成问答生成
- 多阶段串联（评估 → 草稿 → 审查 → 最终答案）
- 记录每次生成的关键指标

---

## 本周文件结构说明

```
task0224/
│
├── generation_pipeline.py   （主文件）
├── generation.log           （运行日志，自动生成）
└── test_results.json        （批量测试结果，自动生成）
```

---

## 1. LLMGenerator（本地 LLM 集成）

通过 Ollama REST API 调用本地大模型。

初始化参数：

```python
LLMGenerator(
    model_name="deepseek-r1:8b",
    base_url="http://localhost:11434",
    timeout=120
)
```

初始化时自动测试连接，打印可用模型列表。

核心方法：

```python
generate(
    prompt,
    system_prompt,
    temperature,
    max_tokens,
    require_json   # True 时自动提取并修复 JSON
)
```

JSON 提取逻辑：

```python
# 从文本中匹配最外层 { ... }
# 自动补全缺失的右括号
# 移除尾部多余逗号
```

`batch_generate()` 对多个 prompt 顺序调用 `generate()`，适合小批量生成。

---

## 2. MedicalGenerationPipeline（多阶段生成流水线）

整合所有组件，实现从 query 到最终答案的完整流程。

初始化组件：

```python
self.context_assembler = ContextAssembler(...)
self.retriever = MultiPathRetriever()
self.reranker = SimpleReranker(...)
self.llm = LLMGenerator(...)
```

完整生成流程：

```
Query
    ↓
阶段 0：Query 增强（调用 task0125）
    ↓
阶段 1：多路召回 + Rerank + 上下文组装
    ↓
阶段 2：证据评估（可选，enable_evaluation）
        使用 [Doc N] 标记过滤上下文
    ↓
阶段 3：草稿生成
    ↓
阶段 4：批判性审查（可选，enable_review）
    ↓
阶段 5：最终答案生成
        审查通过 → 直接用草稿
        有意见  → 调用 final_assembler 重写
    ↓
阶段 6：后处理（段落格式整洁）
    ↓
组装输出结果
```

最终输出结构：

```python
{
    "query": "...",
    "answer": "...",
    "context_metadata": {...},
    "generation_metrics": {
        "total_time_seconds": 139.31,
        "stage_times": {...},
        "token_counts": {...},
        "stage_success": {...}
    },
    "intermediate_results": {
        "evidence_evaluation": "...",
        "draft_answer": "...",
        "review_feedback": "..."
    },
    "sources": [...],
    "timestamp": "2026-02-28 12:34:03"
}
```

---

## 3. 测试主入口

支持两种模式：

```
运行模式（input=输入模式 / test=批量测试，默认 input）:
```

- **input 模式**：手动输入医学问题，实时查看结果
- **test 模式**：自动运行预设 3 条 query，保存指标至 `test_results.json`

运行方式：

```bash
conda activate med_rag
python task0224/generation_pipeline.py
```

---

## Example Run

输入：

```
Does metformin reduce cardiovascular risk in T2DM patients?
```

输出指标：

```
[生成指标]
  总耗时: 139.31s
  各阶段: {'context_assembly': 6.52, 'evidence_evaluation': 41.76,
           'draft_generation': 56.53, 'critical_review': 34.49, 'final_assembly': 0.0}
  token数: {'context': 1394, 'draft': 82, 'final_answer': 108}
  各阶段是否成功: {'evidence_evaluation': True, 'draft_generation': True, ...}
```

说明：

- `final_assembly: 0.0s` 表示审查通过，直接使用草稿
- 上下文 5 块，向量 3 篇 + BM25 2 篇

---

## 本周完成内容总结

- 实现 LLMGenerator，支持 Ollama 本地调用与 JSON 自动提取
- 构建 MedicalGenerationPipeline，完成六阶段生成流水线
- 证据评估与批判审查均可通过参数开关控制
- 编写测试主入口，支持交互和批量测试两种模式
- 生成日志自动写入 generation.log

---

## 当前系统状态

```
Task0118: 向量索引构建        完成
Task0125: 查询理解增强        完成
Task0202: 多路检索 + 重排序   完成
Task0216: 上下文组装 + 提示词 完成
Task0224: LLM 集成 + 生成流水线 完成
```

系统已具备完整的 RAG 端对端能力：

```
用户问题 → 检索 → 上下文组装 → 多阶段 LLM 生成 → 最终答案
```

---

## 下一阶段

对生成质量进行评估：

```
生成答案 → ROUGE / BERTScore 评估 → 指标分析
```
