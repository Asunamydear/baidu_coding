# task0125/rag_retrieval.py
# 作用：查询理解与增强（实体/同义词/缩写）+ filters -> where + Chroma 向量检索（无验证/无边界测试）

import os
import re
import json
import textwrap
from typing import Dict, List, Any, Optional, Tuple

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


# ====== Path / Index Config ======
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "task0118", "chroma_store"))

COLLECTION_NAME = "pubmed_rct20k"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
TOP_K = 5

# 边界保护（只用于增强模块，不做“验证脚本”）
MIN_QUERY_CHARS = 2
MAX_QUERY_CHARS = 2000
MAX_QUERY_WORDS = 300


# ====== Static medical resources (示例，可扩展) ======
MEDICAL_SYNONYMS: Dict[str, List[str]] = {
    "mi": ["myocardial infarction", "heart attack"],
    "t2dm": ["type 2 diabetes mellitus", "type 2 diabetes"],
    "htn": ["hypertension", "high blood pressure"],
    "cvd": ["cardiovascular disease", "cardiovascular diseases"],
}

MEDICAL_PATTERNS: Dict[str, str] = {
    "drug": r"\b(aspirin|metformin|atorvastatin|warfarin|insulin)\b",
}


# ====== Query understanding & augmentation ======
def _normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _truncate_query(q: str) -> str:
    q = q.strip()
    if len(q) > MAX_QUERY_CHARS:
        q = q[:MAX_QUERY_CHARS].rstrip()
    words = q.split()
    if len(words) > MAX_QUERY_WORDS:
        q = " ".join(words[:MAX_QUERY_WORDS]).strip()
    return q


def _find_entities(clean_lower: str) -> Dict[str, List[str]]:
    entities: Dict[str, List[str]] = {}
    for ent_type, pattern in MEDICAL_PATTERNS.items():
        matches = re.findall(pattern, clean_lower, flags=re.IGNORECASE)
        seen = set()
        uniq = []
        for m in matches:
            m2 = m.lower()
            if m2 not in seen:
                seen.add(m2)
                uniq.append(m2)
        entities[ent_type] = uniq
    return entities


def _expand_synonyms(clean_lower: str) -> Dict[str, List[str]]:
    expansions: Dict[str, List[str]] = {}
    for abbr, syns in MEDICAL_SYNONYMS.items():
        if re.search(rf"\b{re.escape(abbr)}\b", clean_lower):
            expansions[abbr] = syns
    return expansions


def _build_augmented_text(clean_query: str, entities: Dict[str, List[str]], synonyms: Dict[str, List[str]]) -> str:
    extra_terms: List[str] = []

    for _, items in entities.items():
        extra_terms.extend(items)

    for _, syns in synonyms.items():
        extra_terms.extend(syns)

    seen = set()
    uniq_terms = []
    for t in extra_terms:
        t2 = t.strip().lower()
        if t2 and t2 not in seen:
            seen.add(t2)
            uniq_terms.append(t.strip())

    if not uniq_terms:
        return clean_query

    # 用分隔符降低“把一堆词当自然句子”的干扰
    return f"{clean_query} | " + "; ".join(uniq_terms)


def _extract_filters(clean_lower: str) -> Dict[str, Any]:

    filters: Dict[str, Any] = {}

    # 1) 显式 doc_id：支持写法 doc_id=xxx / doc:xxx / doc_id:xxx
    m = re.search(r"\bdoc(?:_id)?\s*[:=]\s*([a-zA-Z0-9_.-]+)\b", clean_lower)
    if m:
        filters["doc_id"] = m.group(1)

    # 2) 显式 token_count 条件（用户直接写 token_count<=250 这种）
    # 支持: token_count <= 250 / token<=250 / tokens<400
    m = re.search(r"\b(token_count|tokens?|token)\s*(<=|>=|<|>)\s*(\d+)\b", clean_lower)
    if m:
        op = m.group(2)
        val = int(m.group(3))
        filters["token_count"] = (op, val)

    # 3) “短/长/多块”的语义词触发（你上周验证里用的阈值）
    # 短：<=250；长：>=450；多块：total_chunks>1
    if re.search(r"\b(short|brief|concise|compact)\b", clean_lower):
        filters["token_count"] = ("<=", 250)
    if re.search(r"\b(long|detailed|in-depth|indepth)\b", clean_lower):
        filters["token_count"] = (">=", 450)
    if re.search(r"\b(multi[- ]?chunk|full\s*paper|whole\s*document)\b", clean_lower):
        filters["total_chunks"] = (">", 1)

    return filters


def filters_to_where(filters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    把 filters 翻译成 Chroma 的 where 格式
    支持字段：doc_id / token_count / total_chunks
    """
    if not filters:
        return None

    where: Dict[str, Any] = {}

    # doc_id 精确匹配
    if "doc_id" in filters:
        where["doc_id"] = filters["doc_id"]

    # token_count 范围
    if "token_count" in filters:
        op, val = filters["token_count"]
        op_map = {"<": "$lt", "<=": "$lte", ">": "$gt", ">=": "$gte"}
        if op in op_map:
            where["token_count"] = {op_map[op]: int(val)}

    # total_chunks 范围
    if "total_chunks" in filters:
        op, val = filters["total_chunks"]
        op_map = {"<": "$lt", "<=": "$lte", ">": "$gt", ">=": "$gte"}
        if op in op_map:
            where["total_chunks"] = {op_map[op]: int(val)}

    return where if where else None


def process_medical_query(query: str) -> Dict[str, Any]:
    original_query = "" if query is None else str(query)

    clean_query = _normalize_whitespace(original_query)
    clean_query = _truncate_query(clean_query)

    if len(clean_query) < MIN_QUERY_CHARS:
        return {
            "ok": False,
            "reason": "empty_or_too_short",
            "original_query": original_query,
            "clean_query": clean_query,
            "entities": {},
            "synonyms": {},
            "filters": {},
            "vector_query": "",
            "keyword_query": "",
        }

    clean_lower = clean_query.lower()

    entities = _find_entities(clean_lower)
    synonyms = _expand_synonyms(clean_lower)
    filters = _extract_filters(clean_lower)

    augmented_text = _build_augmented_text(clean_query, entities, synonyms)

    # BGE 检索最佳实践前缀
    vector_query = f"Represent this question for searching relevant passages: {augmented_text}"
    keyword_query = augmented_text

    return {
        "ok": True,
        "original_query": original_query,
        "clean_query": clean_query,
        "entities": entities,
        "synonyms": synonyms,
        "filters": filters,
        "vector_query": vector_query,
        "keyword_query": keyword_query,
    }


# ====== Retrieval ======
def pick_one_sentence(text: str) -> str:
    parts = text.split(".")
    if len(parts) > 1:
        return parts[0].strip() + "."
    return " ".join(text.split()[:20])


def load_collection():
    if not os.path.exists(CHROMA_DIR):
        raise FileNotFoundError(f"Chroma dir not found: {CHROMA_DIR}")

    client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_collection(name=COLLECTION_NAME)


def run_query(title: str, collection, model: SentenceTransformer, query_text: str, where_filter: Optional[Dict[str, Any]] = None):
    print("\n" + title)

    query_vec = model.encode([query_text])[0]

    if where_filter is None:
        results = collection.query(query_embeddings=[query_vec], n_results=TOP_K)
    else:
        results = collection.query(query_embeddings=[query_vec], n_results=TOP_K, where=where_filter)

    for i in range(len(results["ids"][0])):
        print("\nRank", i + 1)
        print("Distance =", results["distances"][0][i])
        print("Chunk ID =", results["ids"][0][i])
        print("Metadata =", results["metadatas"][0][i])

        # 可选：打印一点文本头，方便人工看相关性
        doc = results["documents"][0][i]
        snippet = pick_one_sentence(doc) if isinstance(doc, str) else str(doc)
        print("Text head =", textwrap.shorten(snippet, width=220, placeholder="..."))

    return results


def main():
    print("Using Chroma dir:", CHROMA_DIR)

    model = SentenceTransformer(EMBED_MODEL_NAME)
    collection = load_collection()
    print("Total vectors in index =", collection.count())

    user_query = input("\nEnter your medical question: ").strip()
    query_info = process_medical_query(user_query)

    if not query_info.get("ok"):
        print("\nQuery rejected:", query_info.get("reason"))
        return

    # filters -> where
    where_filter = filters_to_where(query_info["filters"])

    print("\n=== Query Info ===")
    print("clean_query   =", query_info["clean_query"])
    print("entities      =", query_info["entities"])
    print("synonyms      =", query_info["synonyms"])
    print("filters       =", query_info["filters"])
    print("where_filter  =", where_filter)
    print("vector_query  =", query_info["vector_query"])
    print("keyword_query =", query_info["keyword_query"])

    run_query(
        title="RAG Vector Search (augmented) with where-filter",
        collection=collection,
        model=model,
        query_text=query_info["vector_query"],
        where_filter=where_filter,
    )


if __name__ == "__main__":
    main()
