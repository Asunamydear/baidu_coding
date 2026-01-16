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
		
- 如代码所示，生成chunks.jsonl文件

## question3
- 如代码所示，生成stats.json文件

## question4

- {
  "processed_date": "2026-01-17T03:07:41.349256",
  "data_split": "train",
  "original_documents": 15000,
  "docs_split_over_512": 1292,
  "total_chunks": 16679,
  "chunks_per_doc": 1.1119333333333334,
  "chunk_size": 400,
  "chunk_overlap": 80,
  "max_tokens_no_split": 512,
  "output_file": "chunks.jsonl"
}