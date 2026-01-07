## Question 1.1

（观察 title, journal, pub_date, pmid 等字段。思考：journal 和 pub_date 未来是否可作为元数据过滤器，实现“检索近5年《Nature》上的文献”？pmid 能否作为追溯原文的链接？）：  
- 元数据是用于描述文本内容的结构化信息（如期刊、发表时间、唯一标识等），元数据过滤是指在向量检索前基于这些结构化字段对文档集合进行条件筛选，以缩小检索范围并提高检索相关性。
- 本数据没有title, journal,pub_date，不能检索5年《nature》的文献，存在的字段也没办法当过滤器，有abstract_id（唯一），这个可以当作追溯原文的链接。
- feature：{'abstract_id': Value('string'), 'label': Value('string'), 'text': Value('string'), 'sentence_id': Value('int64')}

Question 1.2 
-