## 百度实习资料汇总

- 每周task单独放一个文件夹，ignore忽视数据集，数据集太大没法传送

---

## Task 0301 – 生成答案评估、缓存策略与批量处理

### 文件说明

| 文件                  | 作用                     |
| --------------------- | ------------------------ |
| `answer_evaluator.py` | 四维度答案质量评估器     |
| `cache_manager.py`    | LRU + TTL + 温度过滤缓存 |
| `batch_processor.py`  | 并行批量生成处理器       |
| `run_eval.py`         | 集成验证脚本（主入口）   |
| `eval_single.json`    | Part1 单次生成评估结果   |
| `eval_batch.json`     | Part3 批量生成评估结果   |
| `run_eval.log`        | 完整运行日志             |

---

### 代码说明

#### 1. `AnswerEvaluator`（`answer_evaluator.py`）

四个评估维度，统一入口：

```python
evaluator = AnswerEvaluator()
result = evaluator.evaluate(prediction=answer, reference=ref)
evaluator.print_report(result, query=query)
```

| 维度         | 方法                    | 说明                                                |
| ------------ | ----------------------- | --------------------------------------------------- |
| ROUGE 相似性 | `rouge_similarity()`    | rouge1 / rouge2 / rougeL 的 F1/P/R                  |
| 关键信息召回 | `key_info_recall()`     | 6类正则（百分比/剂量/时间/安全/机制等），存在性匹配 |
| 幻觉检测     | `hallucination_score()` | 8条中英文信号规则，归一化为 [0,1] 风险分            |
| 可读性评估   | `readability_score()`   | 平均句长、长句比例、可读性等级（good/fair/poor）    |

#### 2. `GenerationCache`（`cache_manager.py`）

```python
cache = GenerationCache(maxsize=200, ttl_seconds=3600, temp_threshold=0.4)
key  = cache.make_key(query)          # sha256 哈希键
val  = cache.get(key)                 # 读缓存，过期自动返回 None
ok   = cache.set(key, result, temperature=0.3)  # 只缓存低温度结果
```

- LRU：`OrderedDict`，`move_to_end` + `popitem(last=False)` 实现淘汰
- TTL：写入时记录时间戳，`get` 时检查是否超期
- 温度过滤：`temperature > temp_threshold` 时直接跳过写入
- 线程安全：`threading.Lock()` 保护所有读写

#### 3. `BatchProcessor`（`batch_processor.py`）

```python
processor = BatchProcessor(generate_fn, max_workers=2, cache=cache)
results   = processor.run(queries)     # 并行，结果顺序与输入一致
processor.print_summary(results)
```

- 并发数默认 `min(cpu_count, 8)`
- 单任务失败返回 `{"error": "...", "answer": ""}` 占位，不影响其他任务
- 先查缓存，命中则跳过 LLM 调用

---

### 评估报告（`eval_batch.json` 摘要）

三条 TEST_QUERIES 的批量评估结果（第二次运行，更新参考答案前）：

| Query                | rouge1-F | rougeL-F | recall | halluc     | elapsed | cache  |
| -------------------- | -------- | -------- | ------ | ---------- | ------- | ------ |
| metformin 心血管结局 | 0.0714   | 0.0714   | 0.0    | 0.0 (none) | 0.0s    | ✅ HIT  |
| aspirin 心梗风险     | 0.1212   | 0.0909   | 1.0    | 0.0 (none) | 186.86s | ❌ MISS |
| statin 冠心病收益    | 0.0759   | 0.0506   | 0.0    | 0.15 (low) | 214.53s | ❌ MISS |

- **metformin** 在 Part 2 缓存测试阶段已写入缓存，Part 3 直接命中，耗时 0 秒
- **statin** 答案触发幻觉信号 `vague_citation_zh`（研究表明），评分 0.15（low）
- **aspirin** 关键信息召回率 1.0，AI 答案成功覆盖安全类关键词

---

### 运行日志（`run_eval.log` 关键节点）

```
[INFO] 初始化 MedicalGenerationPipeline ...
[INFO] Load pretrained SentenceTransformer: BAAI/bge-small-en-v1.5
[INFO] Ollama 连接成功，可用模型: ['deepseek-r1:8b']
[INFO] Pipeline 初始化完成。

# Part 1 – 单次生成
[INFO] Query: What is the effect of metformin on cardiovascular outcomes...
[INFO] 上下文组装完成: 5 块, 1287 tokens
[INFO] 草稿生成完成 (耗时 47.22s)
[INFO] 批判性审查完成 (耗时 29.7s)
[INFO] === 生成完成 | 总耗时: 117.92s | 答案长度: 303 words ===
[INFO] 评估结果已保存: eval_single.json

# Part 2 – 缓存测试
[INFO] [CACHE SET] ok=True, query='What is the effect of metformin...'
[INFO] [CACHE HIT] 'What is the effect of metformin...'

# Part 3 – 批量并行
[INFO] BatchProcessor 初始化: max_workers=2
[INFO] BatchProcessor.run: 3 queries, max_workers=2
[INFO] [CACHE HIT] query='What is the effect of metformin...'   ← 0秒返回
[INFO] === 开始生成流水线 | query: Does aspirin reduce the risk... ===
[INFO] === 开始生成流水线 | query: What are the benefits of statins... ===
[INFO] 上下文组装完成: 3 块, 1025 tokens
[INFO] 上下文组装完成: 6 块, 1073 tokens
[INFO] 草稿生成完成 (耗时 72.02s)            ← 两条并行跑
[INFO] 草稿生成完成 (耗时 90.62s)
[INFO] === 生成完成 | 总耗时: 186.86s ===
[INFO] === 生成完成 | 总耗时: 214.53s ===
[INFO] 批量评估结果已保存: eval_batch.json
```

---

### 运行方式

```bash
conda activate med_rag
cd d:\learn\steamcode\coding
python task0301/run_eval.py
```

> 需要 Ollama 服务在线：`ollama serve`（另开终端）
