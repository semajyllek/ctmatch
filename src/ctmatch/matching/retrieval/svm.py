import logging
from typing import List

import numpy as np
from sklearn import svm

logger = logging.getLogger(__name__)


def svm_filter(
    topic_embedding: np.ndarray,
    doc_embeddings_df,
    doc_set: List[int],
    top_n: int,
) -> List[int]:
    """
    Filter documents by training a linear SVM with the topic as the single
    positive example and all candidate docs as negatives. Documents closest
    to the decision boundary on the positive side are most relevant.
    """
    logger.info(f"running svm filter on {len(doc_set)} documents")

    topic_vec = topic_embedding[np.newaxis, :]
    x = np.concatenate([topic_vec, doc_embeddings_df.iloc[doc_set].values], axis=0)
    y = np.zeros(len(doc_set) + 1)
    y[0] = 1

    clf = svm.LinearSVC(class_weight='balanced', verbose=False, max_iter=10000, tol=1e-6, C=0.1)
    clf.fit(x, y)

    similarities = clf.decision_function(x)
    result = list(np.argsort(-similarities)[:min(len(doc_set) + 1, top_n + 1)])
    result.remove(0)

    return [doc_set[(r - 1)] for r in result]
