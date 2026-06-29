
import logging
from typing import Any, Dict, List, Optional, Tuple


# external imports
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from numpy.linalg import norm
from sklearn import svm
import numpy as np
import torch


# package tools
from .reranking.classifier import ClassifierModel
from ..utils.ctmatch_utils import get_processed_data
from ..config import PipeConfig
from .topic import PipeTopic
from ..data.dataprep import DataPrep
from ..device import resolve_device, get_pipeline_device


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


CT_CATEGORIES = [
    "pulmonary", "cardiac", "gastrointestinal", "renal", "psychological", "genetic", "pediatric",
    "neurological", "cancer", "reproductive", "endocrine", "infection", "healthy", "other"
]


GEN_INIT_PROMPT = (
    "I will give you a patient description and a set of clinical trial documents. "
    "Each document will have an ID. Return ONLY a comma-separated list of IDs ranked "
    "from most to least relevant for the patient. Example: 3, 1, 7, 2\n"
)


class CTMatch:

    def __init__(self, pipe_config: Optional[PipeConfig] = None) -> None:
        self.pipe_config = pipe_config if pipe_config is not None else PipeConfig(ir_setup=True)
        self.device = resolve_device()
        self.data = DataPrep(self.pipe_config)
        self.classifier_model = ClassifierModel(self.pipe_config, self.data, self.device)
        self.embedding_model = SentenceTransformer(self.pipe_config.embedding_model_checkpoint)
        self._gen_model = None
        self.category_model = None
        self.filters: Optional[List[str]] = pipe_config.filters

        # filter params
        self.sim_top_n = 10000
        self.svm_top_n = 100
        self.classifier_top_n = 50
        self.gen_top_n = 10


    # main api method
    def match_pipeline(self, topic: str, top_k: int = 10, doc_set: Optional[List[int]] = None) -> List[str]:
        print(f"{top_k=}")

        if doc_set is None:
            doc_set = [i for i in range(len(self.data.index2docid))]
        else:
            self.reset_filter_params(len(doc_set))

        pipe_topic = self.get_pipe_topic(topic)

        if self.filters is None or ('sim' in self.filters):
            doc_set = self.sim_filter(pipe_topic, doc_set, top_n=self.sim_top_n)

        if self.filters is None or ('svm' in self.filters):
            doc_set = self.svm_filter(pipe_topic, doc_set, top_n=self.svm_top_n)

        if self.filters is None or ('classifier' in self.filters):
            doc_set = self.classifier_filter(pipe_topic, doc_set, top_n=self.classifier_top_n)

        if self.filters is None or ('gen' in self.filters):
            doc_set = self.gen_filter(pipe_topic, doc_set, top_n=top_k)

        return self.get_return_data(doc_set[:min(top_k, len(doc_set))])


    def reset_filter_params(self, val: int) -> None:
        self.sim_top_n = self.svm_top_n = self.classifier_top_n = self.gen_top_n = val


    # ------------------------------------------------------------------------------------------ #
    # filtering methods
    # ------------------------------------------------------------------------------------------ #

    def sim_filter(self, pipe_topic: PipeTopic, doc_set: List[int], top_n: int) -> List[int]:
        logger.info(f"running sim filter on {len(doc_set)} docs")

        norm_topic_emb = norm(pipe_topic.embedding_vec)
        scores = []
        for doc_idx in doc_set:
            doc_cat_vec = self.redist_other_category(self.data.doc_categories_df.iloc[doc_idx].values)
            doc_emb_vec = self.data.doc_embeddings_df.iloc[doc_idx].values

            cat_sim = np.dot(pipe_topic.category_vec, doc_cat_vec)
            emb_sim = np.dot(pipe_topic.embedding_vec, doc_emb_vec) / (norm_topic_emb * norm(doc_emb_vec))
            scores.append(cat_sim + emb_sim)

        sorted_indices = list(np.argsort(-np.array(scores)))[:min(len(doc_set), top_n)]
        return [doc_set[i] for i in sorted_indices]


    def svm_filter(self, topic: PipeTopic, doc_set: List[int], top_n: int) -> List[int]:
        logger.info(f"running svm filter on {len(doc_set)} documents")

        topic_embedding_vec = topic.embedding_vec[np.newaxis, :]
        x = np.concatenate([topic_embedding_vec, self.data.doc_embeddings_df.iloc[doc_set].values], axis=0)
        y = np.zeros(len(doc_set) + 1)
        y[0] = 1

        clf = svm.LinearSVC(class_weight='balanced', verbose=False, max_iter=10000, tol=1e-6, C=0.1)
        clf.fit(x, y)

        similarities = clf.decision_function(x)
        result = list(np.argsort(-similarities)[:min(len(doc_set) + 1, top_n + 1)])
        result.remove(0)
        return [doc_set[(r - 1)] for r in result]


    def classifier_filter(self, pipe_topic: PipeTopic, doc_set: List[int], top_n: int) -> List[int]:
        logger.info(f"running classifier filter on {len(doc_set)} documents")

        doc_texts = [v[0] for v in self.data.doc_texts_df.iloc[doc_set].values]
        probs = self.classifier_model.batch_inference(pipe_topic.topic_text, doc_texts, return_preds=True)

        rel_col = self.classifier_model.relevant_col_index()
        rel_scores = probs[:, rel_col].cpu().numpy()
        sorted_indices = list(np.argsort(-rel_scores)[:min(len(doc_set), top_n)])

        logger.info(
            f"classifier_filter: ranking by col {rel_col} (P(relevant)) — "
            f"min={rel_scores.min():.3f}, max={rel_scores.max():.3f}, mean={rel_scores.mean():.3f}"
        )

        return [doc_set[i] for i in sorted_indices]


    def gen_filter(self, topic: PipeTopic, doc_set: List[int], top_n: int = 10) -> List[int]:
        """
        LLM-based final reranking stage. Breaks large doc sets into token-budget-sized
        subqueries, iteratively halving until top_n candidates remain.
        """
        logger.info(f"running gen filter on {len(doc_set)} documents")
        assert top_n > 0, "top_n must be greater than 0"

        if self._gen_model is None:
            from .generation.gen_model import GenModel
            self._gen_model = GenModel(self.pipe_config)

        ranked_docs = doc_set
        iters = 0
        while (len(ranked_docs) > top_n) and (iters < 10) and (len(ranked_docs) // 2 > top_n):
            query_prompts = self.get_subqueries(topic, ranked_docs)
            logger.info(f"calling gen model on {len(query_prompts)} subqueries")

            subrankings = []
            for prompt in query_prompts:
                subrank = self._gen_model.gen_response(prompt)
                subrankings.extend(subrank[:len(subrank) // 2])

            ranked_docs = subrankings
            iters += 1

        return ranked_docs[:min(len(ranked_docs), top_n)]


    # ------------------------------------------------------------------------------------------ #
    # filter helper methods
    # ------------------------------------------------------------------------------------------ #

    def get_pipe_topic(self, topic):
        pipe_topic = PipeTopic(
            topic_text=topic,
            embedding_vec=self.get_embeddings([topic])[0],
            category_vec=self.get_categories(topic)
        )
        return pipe_topic


    def get_embeddings(self, texts: List[str]) -> List[float]:
        return self.embedding_model.encode(texts)

    def get_categories(self, text: str) -> str:
        if self.category_model is None:
            self.category_model = pipeline(
                'zero-shot-classification',
                model=self.pipe_config.category_model_checkpoint,
                device=get_pipeline_device()
            )
        output = self.category_model(text, candidate_labels=CT_CATEGORIES)
        score_dict = {output['labels'][i]: output['scores'][i] for i in range(len(output['labels']))}
        sorted_keys = sorted(score_dict.keys())
        return self.redist_other_category(np.array([score_dict[k] for k in sorted_keys]))

    def redist_other_category(self, category_vec: np.ndarray, other_dim: int = 8) -> np.ndarray:
        other_wt = category_vec[other_dim]
        other_wt_dist = other_wt / (len(category_vec) - 1)
        redist_cat_vec = category_vec + other_wt_dist
        redist_cat_vec[other_dim] = 0
        return redist_cat_vec

    def get_gen_query_prompt(self, topic: PipeTopic, doc_set: List[int]) -> str:
        query_prompt = f"{GEN_INIT_PROMPT}Patient description: {topic.topic_text}\n"

        for i, doc_text in enumerate(self.data.doc_texts_df.iloc[doc_set].values):
            query_prompt += f"ID: {doc_set[i]}, Eligibility Criteria: {doc_text[0]}\n"
            if len(query_prompt.split()) > self.pipe_config.max_query_length:
                break

        return query_prompt, i

    def get_subqueries(self, topic: PipeTopic, doc_set: List[int]) -> List[str]:
        query_prompts = []
        i = 0
        while i < len(doc_set) - 1:
            query_prompt, used_i = self.get_gen_query_prompt(topic, doc_set[i:])
            query_prompts.append(query_prompt)
            i += used_i
        return query_prompts

    def get_return_data(self, doc_set: List[int]) -> List[Tuple[str, str]]:
        return_data = []
        for idx in doc_set:
            nctid = self.data.index2docid.iloc[idx].values[0]
            return_data.append((nctid, self.data.doc_texts_df.iloc[idx].values[0]))
        return return_data
