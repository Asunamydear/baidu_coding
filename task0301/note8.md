# Task 0301 – 生成答案评估、缓存策略与批量处理

---

### question ###

本周任务是在完成 RAG 生成模块（Task0224）基础上：

对生成答案进行质量评估，并搭建缓存与批量处理策略，包括：

1. 创建多维度评估指标
2. 缓存策略
3. 批量处理（optional）

目标：

- 生成答案评估，缓存策略与批量处理
- 将评估、缓存与批量处理模块接入上周的 `MedicalGenerationPipeline`
- 使用上周的测试 query 再次运行，验证各模块运行情况

---

## 本周文件结构说明

```
task0301/
│
├── answer_evaluator.py   （答案评估模块）
├── cache_manager.py      （缓存策略模块）
├── batch_processor.py    （批量处理模块）
├── run_eval.py           （集成验证脚本，主入口）
├── eval_single.json      （Part1 单次评估结果，自动生成）
├── eval_batch.json       （Part3 批量评估结果，自动生成）
└── run_eval.log          （运行日志，自动生成）
```

---

## 1. 评估部分（`answer_evaluator.py`）

创建多维度评估指标，集成四个评估维度：

### a. 文本相似性

使用 `rouge_score` 库进行评估：

```python
from rouge_score import rouge_scorer
scorer = RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
```

计算 ROUGE-1 / ROUGE-2 / ROUGE-L 的 Precision / Recall / F1。

### b. 关键信息提取评估

使用正则表达式提取医学关键信息，相关字段（单词或短语）：

| 类别     | 示例                                   |
| -------- | -------------------------------------- |
| 百分比   | `33%`、`百分之三十`                    |
| 剂量信息 | `500mg`、`500毫克`                     |
| 时间范围 | `12 weeks`、`3个月`                    |
| 安全信息 | `risk`、`副作用`、`不良反应`           |
| 治疗建议 | `recommend`、`建议`、`治疗方案`        |
| 作用机制 | `mechanism`、`pathway`、`机制`、`激活` |

计算召回率：生成的答案覆盖了多少标准答案中的关键信息

```
recall = overlap / gt_matches
```

> **实现说明**：采用存在性匹配（而非精确字符串匹配），只要预测答案中存在该类别的任意匹配，即视为覆盖（recall=1.0）。改进原因：精确匹配会因语言不同（中英文）或具体数值差异导致 overlap 永远为 0。

### c. 幻觉检测评估

无依据的绝对化表述信号，如（单词或短语）：

- `"研究表明"` → 缺少具体引用
- `"已被证明"` → 缺乏限定条件
- `"100%"` → 医学中极少 100%
- `"完全（安全｜有效｜无害）"` → 过度绝对化

生成评估分数，上述信号越多风险越高，归一化到 `[0, 1]`，分为 none / low / medium / high 四级。

### d. 可读性评估

```
平均句子长度（词/字数）
长句比例（超过 40 词/字的句子占比）
英文单词平均字符数
可读性等级：good / fair / poor
```

---

统一评估入口：

```python
evaluator = AnswerEvaluator()
result = evaluator.evaluate(prediction=answer, reference=ref)
evaluator.print_report(result, query=query)
```

输出示例：

```
[ROUGE]         rouge1-F=0.4211  rougeL-F=0.3860
[Key Info]      recall=0.6667
[Hallucination] score=0.4000  risk=MEDIUM
[Readability]   avg_sent_len=11.0  long_ratio=0.00%  level=GOOD
  - time_range: ref=1, pred=1, covered=yes, recall=1.0
  - safety:     ref=1, pred=1, covered=yes, recall=1.0
  - mechanism:  ref=1, pred=2, covered=yes, recall=1.0
```

---

## 2. 缓存策略（`cache_manager.py`）

生成缓存键：查询 + 上下文的哈希。

```python
GenerationCache(maxsize=200, ttl_seconds=3600, temp_threshold=0.4)
```

| 机制            | 实现                                                                  |
| --------------- | --------------------------------------------------------------------- |
| 缓存键          | `sha256(query + \x00 + context)`                                      |
| 大小限制（LRU） | `OrderedDict`，满时淘汰最久未使用条目，避免内存溢出                   |
| TTL 设置        | 医学知识有时效性，写入时记录时间戳，读取时检查是否过期                |
| 温度限制        | 只缓存低温度（确定性）生成结果，`temperature ≤ temp_threshold` 才写入 |
| 线程安全        | 所有读写操作加 `threading.Lock()`                                     |

接口：

```python
key = cache.make_key(query, context)   # 生成缓存键
val = cache.get(key)                   # 读缓存（过期自动返回 None）
ok  = cache.set(key, result, temperature=0.3)  # 写缓存
stats = cache.stats()                  # 命中率统计
```

---

## 3. 批量处理（`batch_processor.py`）

尝试使用 `ThreadPoolExecutor` 等线程工具并行批量生成并收集结果：

```python
BatchProcessor(generate_fn, max_workers=2, cache=cache)
results = processor.run(queries)
```

- 根据 CPU 核心数调整并发数（默认 `min(cpu_count, 8)`），避免过度竞争
- 单个任务失败不影响其他任务（内部 `try/except`，返回占位错误结果）
- 确保输入输出顺序一致（`{future: idx}` 映射，`as_completed` 后按 idx 还原）
- 集成缓存：先查缓存，命中则跳过 LLM 调用

---

## 4. 集成验证脚本（`run_eval.py`）

交付内容：答案评估部分代码与相关指标、缓存以及批量处理策略代码。

完成搭建后使用上周的测试 query 再次运行，验证功能。

运行方式：

```bash
conda activate med_rag
python task0301/run_eval.py
```

三个演示部分：

| Part   | 内容                  | 验证点                            |
| ------ | --------------------- | --------------------------------- |
| Part 1 | 单次生成 + 四维度评估 | 评估器能正确打分                  |
| Part 2 | 同一 query 跑两次     | 第二次命中缓存，耗时接近 0        |
| Part 3 | 3 条 query 并行批量   | 并行耗时 < 串行，汇总展示评估指标 |

双语参考答案（`REFERENCE_ANSWERS`）已内置于脚本中，支持模型用中英文作答时均能产生有效召回。

---

## Example Run

运行结果（Part 3 批量评估）：

```
Query: What is the effect of metformin on cardiovascular outcomes...
  rouge1=0.0714  rougeL=0.0714  recall=0.0  halluc=0.0(none)  cache=True

Query: Does aspirin reduce the risk of myocardial infarction...
  rouge1=0.1212  rougeL=0.0909  recall=1.0  halluc=0.0(none)  cache=False  elapsed=186.86s

Query: What are the benefits of statins...
  rouge1=0.0759  rougeL=0.0506  recall=0.0  halluc=0.15(low)  cache=False  elapsed=214.53s
```

幻觉信号触发 `vague_citation_zh`（研究表明），评估器正确捕获中文幻觉特征。

---

## 本周完成内容总结

- 实现 `AnswerEvaluator`，包含 ROUGE、关键信息召回（中英文正则）、幻觉检测、可读性评估四个维度
- 实现 `GenerationCache`，支持 LRU + TTL + 温度过滤 + 线程安全
- 实现 `BatchProcessor`，支持并行生成、失败隔离、顺序保证、缓存集成
- 编写 `run_eval.py` 集成验证脚本，三部分验证全部通过
- 关键信息召回采用存在性匹配，解决中英文混合场景下精确匹配失效问题

---

## 当前系统状态

```
Task0118: 向量索引构建                完成
Task0125: 查询理解增强                完成
Task0202: 多路检索 + 重排序           完成
Task0216: 上下文组装 + 提示词         完成
Task0224: LLM 集成 + 生成流水线       完成
Task0301: 评估 + 缓存 + 批量处理      完成
```

系统已具备完整的 RAG 端对端能力，并附带评估与缓存层：

```
用户问题 → 检索 → 上下文组装 → 多阶段 LLM 生成 → 答案
                                                  ↓
                          缓存层（命中则跳过生成）← ←
                                                  ↓
                          批量处理（并行多路 query）
                                                  ↓
                          四维度评估（ROUGE / 关键信息 / 幻觉 / 可读性）
```
