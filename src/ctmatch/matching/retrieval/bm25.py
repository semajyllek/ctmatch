import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None


class BM25Index:
    """BM25 index over clinical trial document texts."""

    def __init__(self, doc_texts: List[str]) -> None:
        if BM25Okapi is None:
            raise ImportError("rank_bm25 is required: pip install rank-bm25")
        tokenized = [self._tokenize(text) for text in doc_texts]
        self.index = BM25Okapi(tokenized)
        self._size = len(doc_texts)

    def retrieve(self, query: str, top_n: int, doc_set: Optional[List[int]] = None) -> List[Tuple[int, float]]:
        """
        Retrieve top_n documents by BM25 score.

        If doc_set is provided, only score/rank those document indices.
        Returns list of (doc_index, score) sorted by descending score.
        """
        tokenized_query = self._tokenize(query)
        scores = self.index.get_scores(tokenized_query)

        if doc_set is not None:
            scored = [(idx, scores[idx]) for idx in doc_set]
        else:
            scored = [(idx, scores[idx]) for idx in range(self._size)]

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_n]

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return text.lower().split()


def bm25_filter(
    bm25_index: BM25Index,
    query: str,
    doc_set: List[int],
    top_n: int,
) -> List[int]:
    """Filter documents using BM25 ranking. Returns top_n doc indices."""
    logger.info(f"running bm25 filter on {len(doc_set)} docs")
    results = bm25_index.retrieve(query, top_n=top_n, doc_set=doc_set)
    return [idx for idx, _score in results]


def build_bm25_index(doc_texts_df) -> BM25Index:
    """Build a BM25 index from the doc_texts DataFrame (single 'text' column)."""
    texts = [row[0] if isinstance(row, (list, tuple)) else str(row)
             for row in doc_texts_df.values]
    logger.info(f"building BM25 index over {len(texts)} documents")
    return BM25Index(texts)
