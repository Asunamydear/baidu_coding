# task0301/answer_evaluator.py
# 作用：多维度评估 RAG 生成答案质量
#   a. ROUGE 文本相似性
#   b. 关键信息提取召回率
#   c. 幻觉风险检测

import re
from typing import Dict, Any, List, Optional


# ==========================================
# a. ROUGE 文本相似性
# ==========================================

def rouge_similarity(prediction: str, reference: str) -> Dict[str, float]:
    """
    使用 rouge_score 库计算 ROUGE-1 / ROUGE-2 / ROUGE-L。
    返回每个指标的 precision / recall / f1。
    若无参考答案，返回全 0。
    """
    if not prediction or not reference:
        return {
            "rouge1_f": 0.0, "rouge1_p": 0.0, "rouge1_r": 0.0,
            "rouge2_f": 0.0, "rouge2_p": 0.0, "rouge2_r": 0.0,
            "rougeL_f": 0.0, "rougeL_p": 0.0, "rougeL_r": 0.0,
        }

    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = scorer.score(reference, prediction)
        return {
            "rouge1_f": round(scores["rouge1"].fmeasure, 4),
            "rouge1_p": round(scores["rouge1"].precision, 4),
            "rouge1_r": round(scores["rouge1"].recall, 4),
            "rouge2_f": round(scores["rouge2"].fmeasure, 4),
            "rouge2_p": round(scores["rouge2"].precision, 4),
            "rouge2_r": round(scores["rouge2"].recall, 4),
            "rougeL_f": round(scores["rougeL"].fmeasure, 4),
            "rougeL_p": round(scores["rougeL"].precision, 4),
            "rougeL_r": round(scores["rougeL"].recall, 4),
        }
    except ImportError:
        print("[WARNING] rouge_score 未安装，请执行: pip install rouge-score")
        return {}


# ==========================================
# b. 关键信息提取（正则 + 召回率）
# ==========================================

# 关键医学信息的正则匹配规则（中英文混合）
KEY_INFO_PATTERNS: Dict[str, str] = {
    # 百分比
    "percentage": r"\b\d+(\.\d+)?\s*%",

    # 剂量信息（如 500mg, 2.5 mg/kg, 100 IU）
    "dosage": r"\b\d+(\.\d+)?\s*(mg|g|mcg|ug|ml|IU|mmol|μg)(\s*/\s*(kg|day|dose|d))?\b",

    # 时间范围（如 12 weeks, 6 months, 2 years, 24-hour）
    "time_range": r"\b\d+[\-–]?\d*\s*(week|month|year|day|hour|hr|wk|mo|yr)s?\b",

    # 安全性词（副作用/不良反应）—— 中英文
    "safety": (
        r"\b(adverse\s+event|side\s+effect|toxicity|risk|harm|contraindication"
        r"|adverse\s+reaction|safety|tolerance|tolerability)\b"
        r"|风险|副作用|不良反应|安全性|禁忌"
    ),

    # 治疗建议 —— 中英文
    "treatment": (
        r"\b(recommend|treatment|therapy|intervention|regimen|guideline|protocol"
        r"|suggest|prescribe)\b"
        r"|建议|治疗|方案|疗法|干预"
    ),

    # 作用机制 —— 中英文
    "mechanism": (
        r"\b(mechanism|pathway|inhibit|activate|receptor|signaling|pharmacology"
        r"|mode\s+of\s+action|MOA)\b"
        r"|机制|原理|作用|通路|抑制|激活"
    ),
}


def _extract_key_info(text: str) -> Dict[str, List[str]]:
    """从文本中提取各类关键医学信息，返回 {类型: [匹配列表]}"""
    results: Dict[str, List[str]] = {}
    for info_type, pattern in KEY_INFO_PATTERNS.items():
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        # findall 可能返回 tuple（有 group），提取第一个元素或原字符串
        cleaned = []
        for m in matches:
            if isinstance(m, tuple):
                cleaned.append(m[0] if m[0] else "")
            else:
                cleaned.append(m)
        results[info_type] = [x.strip() for x in cleaned if x.strip()]
    return results


def key_info_recall(prediction: str, reference: str) -> Dict[str, Any]:
    """
    计算生成答案对参考答案中关键信息的召回率。
    recall = |pred_matches ∩ ref_matches| / |ref_matches|  (按类型分别计算然后取均值)
    若 reference 为空，返回 recall=0。
    """
    if not reference:
        return {"recall": 0.0, "details": {}, "pred_counts": {}, "ref_counts": {}}

    pred_info = _extract_key_info(prediction or "")
    ref_info = _extract_key_info(reference)

    details: Dict[str, Dict[str, Any]] = {}
    recall_scores = []

    for info_type in KEY_INFO_PATTERNS:
        pred_set = set(x.lower() for x in pred_info.get(info_type, []))
        ref_set = set(x.lower() for x in ref_info.get(info_type, []))

        gt_count = len(ref_set)
        if gt_count == 0:
            # 参考中没有该类型，跳过（不计入均值）
            details[info_type] = {
                "ref_count": 0,
                "pred_count": len(pred_set),
                "overlap": 0,
                "recall": None   # N/A
            }
            continue

        overlap = len(pred_set & ref_set)
        recall = overlap / gt_count
        recall_scores.append(recall)

        details[info_type] = {
            "ref_count": gt_count,
            "pred_count": len(pred_set),
            "overlap": overlap,
            "recall": round(recall, 4)
        }

    avg_recall = round(sum(recall_scores) / len(recall_scores), 4) if recall_scores else 0.0

    return {
        "recall": avg_recall,
        "details": details,
        "pred_counts": {k: len(v) for k, v in pred_info.items()},
        "ref_counts": {k: len(v) for k, v in ref_info.items()},
    }


# ==========================================
# c. 幻觉检测
# ==========================================

# 无依据的绝对化表述信号（中英文）
HALLUCINATION_SIGNALS: List[Dict[str, Any]] = [
    # ---- 模糊来源引用（高风险）----
    {"pattern": r"\b(studies show|research shows|research indicates|it has been shown)\b",
     "weight": 1.5, "label": "vague_citation_en"},
    {"pattern": r"研究表明|研究显示|已有研究",
     "weight": 1.5, "label": "vague_citation_zh"},

    # ---- 缺乏限定条件（高风险）----
    {"pattern": r"\b(has been proven|is proven|it is proven|proven to)\b",
     "weight": 1.5, "label": "overconfident_claim_en"},
    {"pattern": r"已被证明|已证实|已被验证",
     "weight": 1.5, "label": "overconfident_claim_zh"},

    # ---- 100% 绝对化（高风险）----
    {"pattern": r"\b100\s*%\b",
     "weight": 2.0, "label": "absolute_100_percent"},

    # ---- 过度绝对化形容词 ----
    {"pattern": r"\b(completely\s+(safe|effective|harmless|cured)|always\s+(works|effective))\b",
     "weight": 1.5, "label": "absolute_adj_en"},
    {"pattern": r"完全(安全|有效|无害|治愈)|绝对(安全|有效|无副作用)",
     "weight": 1.5, "label": "absolute_adj_zh"},

    # ---- 无限定的最高级 ----
    {"pattern": r"\b(the (best|only|most effective) (treatment|drug|therapy))\b",
     "weight": 1.0, "label": "unsupported_superlative_en"},
    {"pattern": r"最(佳|好|有效|安全|常用)的(治疗|药物|方案)",
     "weight": 1.0, "label": "unsupported_superlative_zh"},

    # ---- 通用性过强表述 ----
    {"pattern": r"\b(all patients|every patient|no patients|none of the patients)\b",
     "weight": 1.0, "label": "overgeneralization_en"},
    {"pattern": r"所有患者|所有病人|每位患者|无一例外",
     "weight": 1.0, "label": "overgeneralization_zh"},
]


def hallucination_score(prediction: str) -> Dict[str, Any]:
    """
    基于信号词检测生成文本中的幻觉风险。
    返回归一化评分 [0, 1]（0=无风险，1=高风险）及触发的信号列表。
    """
    if not prediction:
        return {"score": 0.0, "risk_level": "none", "signals": [], "raw_weight": 0.0}

    triggered = []
    total_weight = 0.0

    for sig in HALLUCINATION_SIGNALS:
        matches = re.findall(sig["pattern"], prediction, flags=re.IGNORECASE)
        if matches:
            w = sig["weight"] * len(matches)
            total_weight += w
            triggered.append({
                "label": sig["label"],
                "count": len(matches),
                "weight": round(w, 2),
            })

    # 归一化：满分阈值设为 10（超过此值均视为极高风险）
    NORM_CEILING = 10.0
    score = round(min(total_weight / NORM_CEILING, 1.0), 4)

    if score == 0.0:
        risk_level = "none"
    elif score < 0.2:
        risk_level = "low"
    elif score < 0.5:
        risk_level = "medium"
    else:
        risk_level = "high"

    return {
        "score": score,
        "risk_level": risk_level,
        "signals": triggered,
        "raw_weight": round(total_weight, 2),
    }


# ==========================================
# 统一评估入口
# ==========================================

class AnswerEvaluator:
    """
    统一答案评估器，集成三个维度：
      1. ROUGE 文本相似性
      2. 关键信息召回率
      3. 幻觉风险分
    """

    def evaluate(self, prediction: str, reference: Optional[str] = None) -> Dict[str, Any]:
        """
        Args:
            prediction: 待评估的生成答案
            reference:  参考答案（Ground Truth）。若为 None，则跳过 ROUGE/召回率。
        Returns:
            {
              "rouge": {...},
              "key_info": {...},
              "hallucination": {...},
              "summary": {...}   # 核心指标汇总
            }
        """
        rouge = rouge_similarity(prediction, reference or "")
        key_info = key_info_recall(prediction, reference or "")
        halluc = hallucination_score(prediction)

        summary = {
            "rouge1_f":        rouge.get("rouge1_f", 0.0),
            "rougeL_f":        rouge.get("rougeL_f", 0.0),
            "key_info_recall": key_info.get("recall", 0.0),
            "halluc_score":    halluc.get("score", 0.0),
            "halluc_risk":     halluc.get("risk_level", "none"),
        }

        return {
            "rouge": rouge,
            "key_info": key_info,
            "hallucination": halluc,
            "summary": summary,
        }

    def print_report(self, result: Dict[str, Any], query: str = ""):
        """友好打印评估报告"""
        print("\n" + "=" * 60)
        if query:
            print(f"Query: {query[:80]}")
        print("=" * 60)

        s = result["summary"]
        print(f"[ROUGE]         rouge1-F={s['rouge1_f']:.4f}  rougeL-F={s['rougeL_f']:.4f}")
        print(f"[Key Info]      recall={s['key_info_recall']:.4f}")
        print(f"[Hallucination] score={s['halluc_score']:.4f}  risk={s['halluc_risk'].upper()}")

        # 触发的幻觉信号
        signals = result["hallucination"].get("signals", [])
        if signals:
            print("  Signals triggered:")
            for sig in signals:
                print(f"    - {sig['label']} (×{sig['count']}, weight={sig['weight']})")

        # 关键信息细节
        details = result["key_info"].get("details", {})
        active = {k: v for k, v in details.items() if v.get("recall") is not None}
        if active:
            print("  Key info recall by type:")
            for k, v in active.items():
                print(f"    - {k}: ref={v['ref_count']}, pred={v['pred_count']}, "
                      f"overlap={v['overlap']}, recall={v['recall']}")
        print("=" * 60)


# ==========================================
# 简单自测
# ==========================================

if __name__ == "__main__":
    evaluator = AnswerEvaluator()

    pred = (
        "Studies show that metformin has been proven to completely reduce cardiovascular risk by 100% "
        "in all patients. The recommended dose is 500mg twice daily for 12 weeks. "
        "The mechanism involves AMPK pathway activation."
    )
    ref = (
        "Metformin reduces cardiovascular risk in type 2 diabetes by 33% (95% CI). "
        "Typical dose: 500-2000mg/day for 6-12 months. Mechanism: AMPK activation."
    )

    result = evaluator.evaluate(pred, ref)
    evaluator.print_report(result, query="Effect of metformin on cardiovascular outcomes")
