# Task 0202 – RAG 检索系统：Multi-Path 检索与重排序流水线

---

### question ###

本周任务是在完成查询理解（Task0125）与向量索引构建（Task0118）基础上：

构建完整的 RAG 检索流水线，包括：

1. 多路检索（Vector + BM25）
2. 融合策略（Fusion）
3. 重排序器（Reranker）
4. 最终输出结构化证据列表

目标：

- 提升检索精度
- 优化排序相关性
- 为后续大模型生成提供高质量证据输入

---

## 本周文件结构说明

本周共包含 4 个核心文件：

```
task0202/
│
├── bm25_index.py
├── multipath_retriever.py
├── reranker.py
└── pipeline_retrieve.py
```

---

## 1. bm25_index.py  
关键词检索模块（Keyword Retrieval）

功能：

- 构建 BM25 倒排索引
- 对所有 chunk 进行分词
- 计算文档长度与平均长度
- 支持 BM25 相关性打分
- 返回 TopK 文档

核心逻辑示例：

```python
# 分词
tokens = text.lower().split()

# 计算 BM25 分数
score = idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avg_len))
```

作用：

- 提供精确关键词匹配能力
- 弥补向量检索在精确词匹配上的不足
- 与向量检索形成互补

---

## 2. multipath_retriever.py  
多路检索与融合模块（MultiPath Retriever）

功能：

- 调用向量检索（ChromaDB）
- 调用 BM25 检索
- 支持三种融合策略：
  - simple
  - rrf（Reciprocal Rank Fusion）
  - weighted（向量权重更高）

核心接口：

```python
retrieve(
    query_info,
    top_k_vector,
    top_k_keyword,
    fusion_strategy
)
```

融合策略示例：

RRF：

```python
score += 1 / (rank + k)
```

Weighted：

```python
fusion_score = 0.7 * vector_score + 0.3 * normalized_bm25_score
```

输出结构示例：

```python
{
    "uid": "...",
    "doc_id": "...",
    "source": "vector" / "bm25",
    "fusion_score": ...
}
```

作用：

- 提升召回率
- 保留不同检索路径的优势
- 输出融合后的候选文档列表

---

## 3. reranker.py  
重排序模块（Re-Ranking）

模型：

```
BAAI/bge-reranker-base
```

功能：

- 对 (query, document) 进行相关性打分
- 输出 rerank_score
- 重新排序候选文档

加载方式：

```python
self.tokenizer = AutoTokenizer.from_pretrained(model_name)
self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
```

推理流程：

```python
tokenize → forward → logits → relevance score
```

作用：

- 修正融合阶段排序偏差
- 降低弱相关文档排名
- 提升高相关证据至 Top

---

## 4. pipeline_retrieve.py  
完整检索流水线（Full Retrieval Pipeline）

整体流程：

```
Query
  ↓
Query增强（Task0125）
  ↓
MultiPath 检索（Vector + BM25）
  ↓
Fusion
  ↓
Reranker
  ↓
输出 TopK 证据列表
  ↓
保存 retrieval_output.json
```

核心调用：

```python
process_medical_query()
MultiPathRetriever.retrieve()
SimpleReranker.rerank()
```

最终输出结构示例：

```json
{
  "query": "...",
  "references": [
    {
      "rank": 1,
      "doc_id": "...",
      "rerank_score": 3.01,
      "source": "vector",
      "text": "..."
    }
  ]
}
```

作用：

- 提供标准化证据输入
- 为后续 LLM 生成阶段准备结构化数据
- 构成完整 RAG 检索系统

---

## Example Run

输入：

```
metformin cardiovascular disease
```

Reranked Top1：

```
Large prospective studies have demonstrated that metformin treatment improves cardiovascular prognosis.
```

分数分布示例：

```
3.01
2.15
2.08
1.54
1.11
...
-1.58
```

说明：

- 强相关文档得分明显更高
- 弱相关或无关内容被压低
- 排序发生优化

---

## 本周完成内容总结

- 实现 BM25 关键词检索
- 构建 MultiPath 多路检索框架
- 实现三种融合策略
- 集成 BGE Reranker 模型
- 完成完整检索流水线
- 输出结构化证据 JSON

---

## 当前系统状态

```
Task0118: 向量索引构建 完成
Task0125: 查询理解增强 完成
Task0202: 多路检索 + 融合 + 重排序 完成
```

系统已具备：

- 高召回能力
- 高相关性排序
- 可扩展生成阶段接口

---

## 下一阶段

接入生成模块（LLM Answer Generation）

```
evidence → prompt构造 → LLM生成 → 引用证据
```

RAG 检索系统核心部分已完成。

