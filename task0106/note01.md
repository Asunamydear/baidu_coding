## Question 1.1

（观察 title, journal, pub_date, pmid 等字段。思考：journal 和 pub_date 未来是否可作为元数据过滤器，实现“检索近5年《Nature》上的文献”？pmid 能否作为追溯原文的链接？）：  
- 元数据是用于描述文本内容的结构化信息（如期刊、发表时间、唯一标识等），元数据过滤是指在向量检索前基于这些结构化字段对文档集合进行条件筛选，以缩小检索范围并提高检索相关性。
- 本数据没有title, journal,pub_date，不能检索5年《nature》的文献，存在的字段也没办法当过滤器，有abstract_id（唯一），这个可以当作追溯原文的链接。
- feature：{'abstract_id': Value('string'), 'label': Value('string'), 'text': Value('string'), 'sentence_id': Value('int64')}

## Question 1.2 

（字段完整性检查（字段缺失率？清洗策略：丢弃或填充） 
 如果 abstract 缺失率 > 1%，你需要制定清洗策略：是丢弃，还是用 title 填充？）

- text 缺失数量: 0
- text 缺失率: 0.0
- text的缺失数量为零所以不需要清洗，如果需要清洗，直接丢弃就好了

## Qusetion 1.3
（基础质量分析）

- 极短文本数量: 328 （小于20）单句小于20词，所以几乎不存在，基础质量非常高，可以忽略。
- 极短文本比例: 0.001856863033706593

## Qusetion 2.1
（分层抽样，按照tokenizer使用的如下
- "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",和deepseek-r1:8b是一样的
### token 最短的 5 篇摘要
| abstract_id | token_len |
|------------|-----------|
| 25752109   | 109       |
| 24473376   | 119       |
| 25407377   | 121       |
| 24387919   | 121       |
| 24636143   | 126       |

### token 最长的 5 篇摘要

| abstract_id | token_len |
|------------|-----------|
| 25130995   | 1247      |
| 26144908   | 1175      |
| 25481791   | 1124      |
| 25795409   | 1107      |
| 24717919   | 1099      |

### token 接近中位数的 5 篇摘要

| abstract_id | token_len |
|------------|-----------|
| 25679343   | 365       |
| 24844551   | 365       |
| 24655865   | 365       |
| 26016823   | 365       |
| 25965710   | 365       |

## Question2.2

(结构：是否遵循“背景-方法-结果-结论”（IMRaD）结构？)

- 完全遵循，分类标签就是这么分的

## Question2.3
术语：是否充满缩写（如“EGFR”、“PCI”）？

## Question2.4
同一概念是否有不同表述（如“heart attack”和“myocardial infarction”）？


