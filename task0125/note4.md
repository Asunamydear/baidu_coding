# Task 0125 – RAG 检索系统：查询理解与增强

---

### question ###

哈啰同学，以下是本周开发 task：

**检索系统开发第一部分：**

在完成向量化索引后，构建 RAG 系统的智能检索系统。

- **输入**：用户自然语言查询  
  （如：“二甲双胍对心血管疾病有何影响？”）
- **输出**：精准筛选的、包含多维度证据的参考文档列表  
- **目标**：检索结果与查询高度相关、证据充分，支持后续生成高质量答案

---

### 本周核心处理：查询理解与增强

本周重点实现 **Query Understanding & Augmentation**，包括：

- 医学实体识别（drug）
- 医学缩写 / 同义词扩展
- 查询基础清洗
- 构建多种检索查询
  - 向量检索查询（BGE 指令前缀）
  - 关键词检索查询
- 从自然语言中抽取过滤条件（filters）

#### 静态医学同义词（示例）

```python
MEDICAL_SYNONYMS = {
    "mi": ["myocardial infarction", "heart attack"]
}
```

#### 医学实体识别规则（示例）

```python
MEDICAL_PATTERNS = {
    "drug": r"\b(aspirin|metformin|atorvastatin|warfarin|insulin)\b"
}
```

---

### 向量检索查询构造

采用 BGE 模型推荐的检索指令前缀：

```python
base_query = f"Represent this question for searching relevant passages: {query}"
```

---

### 本周产出内容

- 完成查询理解与增强模块代码
- 实现医学实体识别与同义词扩展
- 构建向量查询与关键词查询
- 从查询中抽取 filters，并映射为 Chroma 的 `where` 条件
- 与已有向量索引完成集成，用于 RAG 检索阶段

---

### Example Run（示例）

**输入查询：**

```text
metformin effect on cardiovascular disease short
```

**查询理解结果：**

```text
entities     = {'drug': ['metformin']}
filters      = {'token_count': ('<=', 250)}
where_filter = {'token_count': {'$lte': 250}}
```

**检索结果（Top-5，均满足 token_count ≤ 250）：**

```text
Rank 1: token_count = 198
Rank 2: token_count = 201
Rank 3: token_count = 243
Rank 4: token_count = 217
Rank 5: token_count = 232
```

---

