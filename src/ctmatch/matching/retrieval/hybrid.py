import logging
from typing import Dict, List, Tuple

import numpy as np
from numpy.linalg import norm

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    *ranked_lists: List[Tuple[int, float]],
    k: int = 60,
) -> List[Tuple[int, float]]:
    """
    Combine multiple ranked result lists using Reciprocal Rank Fusion.

    Each input is a list of (doc_index, score) tuples, sorted by descending score.
    RRF score for a document = sum(1 / (k + rank_i)) across all lists where it appears.
    """
    rrf_scores: Dict[int, float] = {}
    for ranked_list in ranked_lists:
        for rank, (doc_idx, _score) in enumerate(ranked_list):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)

    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return fused


def hybrid_filter(
    bm25_index,
    query: str,
    topic_embedding: np.ndarray,
    doc_embeddings_df,
    doc_set: List[int],
    top_n: int,
    bm25_top_n: int = 0,
    emb_top_n: int = 0,
    rrf_k: int = 60,
) -> List[int]:
    """
    Hybrid retrieval combining BM25 and embedding similarity via RRF.

    Retrieves from both methods independently, then fuses with RRF.
    bm25_top_n and emb_top_n control how many candidates each method contributes
    (defaults to 2x top_n if not specified).
    """
    logger.info(f"running hybrid filter on {len(doc_set)} docs")

    per_method = bm25_top_n or emb_top_n or min(len(doc_set), top_n * 2)

    # BM25 ranking
    bm25_results = bm25_index.retrieve(query, top_n=per_method, doc_set=doc_set)

    # Embedding ranking
    norm_topic = norm(topic_embedding)
    emb_scored = []
    for doc_idx in doc_set:
        doc_emb = doc_embeddings_df.iloc[doc_idx].values
        sim = np.dot(topic_embedding, doc_emb) / (norm_topic * norm(doc_emb))
        emb_scored.append((doc_idx, float(sim)))
    emb_scored.sort(key=lambda x: x[1], reverse=True)
    emb_results = emb_scored[:per_method]

    # Fuse
    fused = reciprocal_rank_fusion(bm25_results, emb_results, k=rrf_k)
    return [doc_idx for doc_idx, _score in fused[:top_n]]
