## question1
环境准备与数据加载
	dataframe转换
	基础清洗
	为每篇文献生成唯一标识id（如果原始数据没有合适id）（pmid）
- 数据原本就有id，dataframe已经转换

## question2

2实施制定的文本分割策略
	a 根据长度进行智能分割（如源数据）：
		加载tokenizer
		初始化分割器
		length_function=self._count_tokens
		为每个块生成唯一ID

            chunk_data = {
                "chunk_id": chunk_id,
                "text": text,
                "doc_id": document['doc_id'],  	# 归属的原文ID
                "chunk_index": i,              			# 块在原文中的序号
                "total_chunks": len(texts),    		# 原文被分成的总块数
                "source_title": document['title'],      # 保留原文标题，便于追溯
            }
	
	b 整体不分割（如200k-rct类型数据）：
		data = {
            	"chunk_id": document['doc_id'],  	# 直接使用文献ID作为块ID
           	"text": full_text,
            	"doc_id": document['doc_id'],
            	"chunk_index": 0,
           	"total_chunks": 1,
            	"source_title": document['title'],
            	"token_count": self._count_tokens(full_text),
        	}
		
- 