## 百度实习资料汇总

**项目简介**：基于 PubMed RCT20k 数据集的医疗问答 RAG (Retrieval-Augmented Generation) 系统，覆盖数据分析 → 向量索引 → 多路检索 → LLM 生成 → 答案评估全链路。

- 每周 task 单独放一个文件夹；数据集过大，已加入 `.gitignore` 不上传
- 模型：BGE-small-en（嵌入） + BGE-reranker-base（重排序） + DeepSeek-R1:8b（生成，Ollama 本地）

---

## 系统整体流程

```
PubMed RCT20k 数据集（176,102 条句子）
         ↓
   task0106  数据分析（字段/token分布/质量）
         ↓
   task0110  文本分块 → chunks.jsonl（16,679 块）
         ↓
   task0118  向量化 → ChromaDB 索引 + BM25 倒排索引
         ↓
用户医学问题
         ↓
   task0125  查询增强（实体识别 / 同义词 / 缩写扩展）
         ↓
   task0202  多路检索（向量 + BM25）→ RRF 融合 → BGE Reranker 重排
         ↓
   task0216  上下文组装（去重 / token 控制） + 四阶段提示词
         ↓
   task0224  LLM 多阶段生成：证据评估 → 草稿 → 审查 → 最终答案
         ↓
   task0301  答案评估 + 缓存 + 批量处理
         ↓
      最终答案 + 四维度评估报告
```

---

## Task 0106 – 数据分析

**文件**：`task0106/question1.py`、`question2and3.py`、`question2_2.py`

**数据集**：PubMed RCT20k，每条记录为临床试验摘要中的一个句子，字段：
`abstract_id`（唯一标识）、`label`（BACKGROUND/OBJECTIVE/METHODS/RESULTS/CONCLUSIONS）、`text`、`sentence_id`

**分析结果**：

| 指标                  | 数值                 |
| --------------------- | -------------------- |
| 摘要总数              | 15,000               |
| 平均 token 长度       | 378.58               |
| 中位数 token 长度     | 365                  |
| 最短 / 最长           | 109 / 1247           |
| text 缺失率           | 0.0（无需清洗）      |
| 极短文本（<20词）数量 | 328（0.19%，可忽略） |
| 超过 512 token 的摘要 | 1292 篇（8.61%）     |

**分割策略**：
- ≤512 token：整篇直接作为一个 chunk
- \>512 token：按 IMRaD 标签（BACKGROUND/METHODS/RESULTS 等）分组，超长语义段再按多句划窗切割，每块不超过 512 token

---

## Task 0110 – 文本分块

**文件**：`task0110/question1.py`、`question2.py`、`question5.py`

**输入**：PubMed 原始句子数据
**输出**：`chunks.jsonl`（16,679 个分块，每块含文本和元数据）

分块逻辑：按摘要分组 → 按 IMRaD 标签合并相邻句子 → 超长段滑动窗口切割 → 去重 → 保存

---

## Task 0118 – 向量索引构建

**文件**：`task0118/build_index.py`、`text_verify.py`

**输入**：`chunks.jsonl`
**输出**：ChromaDB 向量索引（持久化到本地）

- 嵌入模型：`BAAI/bge-small-en-v1.5`
- 向量库：ChromaDB（支持余弦相似度检索）
- BM25 倒排索引同步构建（在 task0202 中初始化使用）

---

## Task 0125 – 查询理解与增强

**文件**：`task0125/rag_retrieval.py`

**输入**：原始用户问题（字符串）
**输出**：增强后的查询结构（含同义词、缩写扩展、检索 filter）

```python
query_info = process_medical_query("What is the effect of metformin on cardiovascular outcomes?")
# 输出：{
#   "original": "...",
#   "expanded_terms": ["metformin", "biguanide", "T2DM", "type 2 diabetes"],
#   "filters": {"label": ["RESULTS", "CONCLUSIONS"]}
# }
```

---

## Task 0202 – 多路检索 + 融合 + 重排序

**文件**：`task0202/bm25_index.py`、`multipath_retriever.py`、`reranker.py`、`pipeline_retrieve.py`

**输入**：增强 query 结构
**输出**：结构化证据列表（含排名、来源、相关性分数）

### 检索架构

```
增强 Query
    ├── 向量检索（ChromaDB + BGE）→ Top-K 文档
    └── BM25 检索（倒排索引）   → Top-K 文档
                ↓
          RRF 融合（score += 1/(rank+k)）
                ↓
        BGE Reranker 重排序
                ↓
      证据列表（rerank_score 排序）
```

- **BM25** 公式：`idf * tf*(k1+1) / (tf + k1*(1-b + b*doc_len/avg_len))`
- **融合**：RRF / Weighted（向量0.7 + BM25×0.3）
- **重排模型**：`BAAI/bge-reranker-base`，`(query, doc)` 对打分

**示例输出（rerank 分数分布）**：
```
Top1: 3.01  → "Large prospective studies have demonstrated that metformin treatment improves cardiovascular prognosis."
Top2: 2.15
Top3: 2.08
...
弱相关: -1.58
```

**结果 JSON**：
```json
{
  "query": "...",
  "references": [
    { "rank": 1, "doc_id": "...", "rerank_score": 3.01, "source": "vector", "text": "..." }
  ]
}
```

---

## Task 0216 – 上下文组装 + 提示词工程

**文件**：`task0216/context_builder.py`

**输入**：重排后的文档列表
**输出**：格式化上下文字符串 + 四阶段 Prompt 模板

- 去重（基于文本哈希）
- 多样性筛选（避免同一来源文档堆叠）
- Token 预算控制（≤3000 tokens）
- 四阶段 Prompt：证据评估 → 草稿生成 → 批判审查 → 最终答案

---

## Task 0224 – LLM 集成与多阶段生成

**文件**：`task0224/generation_pipeline.py`

**输入**：上下文 + Prompt
**输出**：最终答案文本 + 生成指标

### 生成流程

```
Query
  ↓ 上下文组装（5块, 1287 tokens）
  ↓ 阶段1：证据评估（~34s，评估哪些文档有用）
  ↓ 阶段2：草稿生成（~47s）
  ↓ 阶段3：批判审查（~30s，审查草稿，提出改进意见）
  ↓ 阶段4：最终答案（审查通过→直接用草稿；有意见→重写）
最终答案
```

**示例生成指标**（`run_eval.log` 摘录）：
```
[INFO] 上下文组装完成: 5 块, 1287 tokens
[INFO] 草稿生成完成 (耗时 47.22s)
[INFO] 批判性审查完成 (耗时 29.7s)
[INFO] === 生成完成 | 总耗时: 117.92s | 答案长度: 303 words ===
```

**输出结构**：
```python
{
    "query": "...",
    "answer": "...",
    "generation_metrics": {
        "total_time_seconds": 117.92,
        "stage_times": {"context_assembly": 6.52, "evidence_evaluation": 34.0,
                        "draft_generation": 47.22, "critical_review": 29.7, "final_assembly": 0.0},
        "token_counts": {"context": 1287, "draft": 82, "final_answer": 303}
    },
    "sources": [...]
}
```

> `final_assembly: 0.0s` 说明审查通过，直接使用草稿，无需重写

---

## Task 0301 – 答案评估 + 缓存 + 批量处理

**文件**：`task0301/answer_evaluator.py`、`cache_manager.py`、`batch_processor.py`、`run_eval.py`

### 评估模块（`AnswerEvaluator`）

四个评估维度：

| 维度         | 方法                    | 输出                           |
| ------------ | ----------------------- | ------------------------------ |
| ROUGE 相似性 | `rouge_similarity()`    | rouge1/2/L 的 F1/P/R           |
| 关键信息召回 | `key_info_recall()`     | 6类正则提取，存在性匹配 recall |
| 幻觉检测     | `hallucination_score()` | [0,1] 风险分 + 等级            |
| 可读性评估   | `readability_score()`   | 平均句长、长句比例、等级       |

**关键信息类别**（中英文正则兼容）：

| 类别       | 提取目标                              |
| ---------- | ------------------------------------- |
| percentage | `33%`、`百分之三十`                   |
| dosage     | `500mg`、`500毫克`、`每日两次`        |
| time_range | `12 weeks`、`3个月`、`6-12个月`       |
| safety     | `adverse event`、`副作用`、`不良反应` |
| treatment  | `recommend`、`建议`、`治疗方案`       |
| mechanism  | `AMPK pathway`、`激活`、`通路`        |

**幻觉信号**（触发则计分）：
- `研究表明 / studies show` → 来源模糊
- `已被证明 / has been proven` → 缺乏限定
- `100%` → 医学中极少绝对
- `完全安全/有效 / all patients` → 过度绝对化

### 缓存模块（`GenerationCache`）

```python
cache = GenerationCache(maxsize=200, ttl_seconds=3600, temp_threshold=0.4)
```

| 机制     | 实现                                     |
| -------- | ---------------------------------------- |
| 缓存键   | `sha256(query + context)`                |
| LRU 淘汰 | `OrderedDict.move_to_end + popitem`      |
| TTL 过期 | 写入时记录时间戳，读时判断老化           |
| 温度过滤 | `temperature > 0.4` 的高随机性结果不写入 |
| 线程安全 | `threading.Lock()` 保护所有读写          |

### 批量处理模块（`BatchProcessor`）

```python
processor = BatchProcessor(generate_fn, max_workers=2, cache=cache)
results   = processor.run(queries)     # 顺序保证，单任务失败隔离
```

---

### 评估报告（`eval_batch.json`）

三条 TEST_QUERIES 的批量评估结果：

| Query                | rouge1-F | rougeL-F | recall | halluc     | elapsed | cache  |
| -------------------- | -------- | -------- | ------ | ---------- | ------- | ------ |
| metformin 心血管结局 | 0.0714   | 0.0714   | 0.0    | 0.0 (none) | 0.0s    | ✅ HIT  |
| aspirin 心梗风险     | 0.1212   | 0.0909   | 1.0    | 0.0 (none) | 186.86s | ❌ MISS |
| statin 冠心病收益    | 0.0759   | 0.0506   | 0.0    | 0.15 (low) | 214.53s | ❌ MISS |

- `metformin` 命中缓存（Part 2 已写入），耗时 0 秒
- `statin` 触发 `vague_citation_zh`（研究表明），幻觉分 0.15
- `aspirin` 关键信息召回 1.0，安全类关键词完全覆盖

### 运行日志（`run_eval.log` 关键节点）

```
[INFO] 初始化 MedicalGenerationPipeline ...
[INFO] Load pretrained SentenceTransformer: BAAI/bge-small-en-v1.5
[INFO] Ollama 连接成功，可用模型: ['deepseek-r1:8b']

# Part 1 – 单次生成 + 四维度评估
[INFO] 上下文组装完成: 5 块, 1287 tokens
[INFO] 草稿生成完成 (耗时 47.22s)
[INFO] 批判性审查完成 (耗时 29.7s)
[INFO] === 生成完成 | 总耗时: 117.92s | 答案长度: 303 words ===
[INFO] 评估结果已保存: eval_single.json

# Part 2 – 缓存测试
[INFO] [CACHE SET] ok=True, query='What is the effect of metformin...'
[INFO] [CACHE HIT] 'What is the effect of metformin...'   ← 第2次查询命中，耗时0s

# Part 3 – 批量并行（2线程）
[INFO] BatchProcessor 初始化: max_workers=2
[INFO] BatchProcessor.run: 3 queries, max_workers=2
[INFO] [CACHE HIT] query='What is the effect of metformin...'   ← 0秒返回
[INFO] === 开始生成流水线 | query: Does aspirin... ===
[INFO] === 开始生成流水线 | query: What are the benefits of statins... ===
[INFO] 草稿生成完成 (耗时 72.02s)   ← 两条并行跑，几乎同时完成
[INFO] 草稿生成完成 (耗时 90.62s)
[INFO] === 生成完成 | 总耗时: 186.86s ===
[INFO] === 生成完成 | 总耗时: 214.53s ===
[INFO] 批量评估结果已保存: eval_batch.json
```

---

## 运行方式

```bash
# 1. 激活环境
conda activate med_rag

# 2. 确认 Ollama 服务在线（另开终端）
ollama serve

# 3. 运行集成验证
cd d:\learn\steamcode\coding
python task0301/run_eval.py
```

---

## 系统状态

```
Task0106: 数据分析                    ✅ 完成
Task0110: 文本分块                    ✅ 完成
Task0118: 向量索引构建                 ✅ 完成
Task0125: 查询理解增强                 ✅ 完成
Task0202: 多路检索 + 融合 + 重排序      ✅ 完成
Task0216: 上下文组装 + 提示词           ✅ 完成
Task0224: LLM 集成 + 多阶段生成流水线   ✅ 完成
Task0301: 答案评估 + 缓存 + 批量处理    ✅ 完成
```
