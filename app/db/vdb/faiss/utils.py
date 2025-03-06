import numpy as np
from typing import List, Optional

def normalize_vectors(vectors: List[List[float]]) -> np.ndarray:
    """归一化向量"""
    vectors_np = np.array(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors_np, axis=1, keepdims=True)
    return vectors_np / norms

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """计算余弦相似度"""
    a_np = np.array(a)
    b_np = np.array(b)
    return np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np))

def filter_results_by_score(
    results: List[dict],
    threshold: float = 0.5,
    score_key: str = "distance"
) -> List[dict]:
    """根据分数过滤结果"""
    return [r for r in results if r[score_key] >= threshold] 