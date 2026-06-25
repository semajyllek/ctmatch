import logging
from typing import List

import numpy as np
from numpy.linalg import norm

logger = logging.getLogger(__name__)


def sim_filter(
    topic_embedding: np.ndarray,
    topic_category: np.ndarray,
    doc_embeddings_df,
    doc_categories_df,
    doc_set: List[int],
    top_n: int,
) -> List[int]:
    """
    Filter documents by combined embedding cosine similarity + category match.
    """
    logger.info(f"running sim filter on {len(doc_set)} docs")

    topic_cat_vec = _exclusive_argmax(topic_category)
    norm_topic = norm(topic_embedding)
    topic_argmax = np.argmax(topic_cat_vec)

    cosine_dists = []
    for doc_idx in doc_set:
        doc_cat_vec = _redist_other_category(doc_categories_df.iloc[doc_idx].values)
        doc_cat_vec = _exclusive_argmax(doc_cat_vec)
        doc_emb_vec = doc_embeddings_df.iloc[doc_idx].values

        cat_dist = 0.0 if np.argmax(doc_cat_vec) == topic_argmax else 1.0
        emb_dist = np.dot(topic_embedding, doc_emb_vec) / (norm_topic * norm(doc_emb_vec))
        cosine_dists.append(cat_dist + emb_dist)

    sorted_indices = list(np.argsort(cosine_dists))[:min(len(doc_set), top_n)]
    return [doc_set[i] for i in sorted_indices]


def embedding_only_filter(
    topic_embedding: np.ndarray,
    doc_embeddings_df,
    doc_set: List[int],
    top_n: int,
) -> List[int]:
    """Filter documents by embedding cosine similarity only (no category model)."""
    logger.info(f"running embedding-only filter on {len(doc_set)} docs")

    norm_topic = norm(topic_embedding)
    sims = []
    for doc_idx in doc_set:
        doc_emb = doc_embeddings_df.iloc[doc_idx].values
        sims.append(np.dot(topic_embedding, doc_emb) / (norm_topic * norm(doc_emb)))

    sorted_indices = list(np.argsort(sims)[::-1])[:min(len(doc_set), top_n)]
    return [doc_set[i] for i in sorted_indices]


def _exclusive_argmax(vector: np.ndarray) -> np.ndarray:
    result = np.zeros(len(vector))
    result[np.argmax(vector)] = 1
    return result


def _redist_other_category(category_vec: np.ndarray, other_dim: int = 8) -> np.ndarray:
    other_wt = category_vec[other_dim]
    other_wt_dist = other_wt / (len(category_vec) - 1)
    redist = category_vec + other_wt_dist
    redist[other_dim] = 0
    return redist
