
import logging
from typing import List, NamedTuple, Optional, Union

from .utils.eval_utils import (
    calc_first_positive_rank, calc_f1, get_kz_topic2text, get_trec_topic2text
)
from .modelconfig import ModelConfig
from .match import CTMatch
from pathlib import Path
from tqdm import tqdm
import numpy as np

logger = logging.getLogger(__name__)


class EvaluatorConfig(NamedTuple):
    rel_paths: List[str]
    trec_topic_path: Union[Path, str]  = None
    kz_topic_path: Union[Path, str] = None
    max_topics: int = 200
    openai_api_key: Optional[str] = None


class Evaluator:
    def __init__(self, eval_config: EvaluatorConfig) -> None:
        self.rel_paths: List[str] = eval_config.rel_paths
        self.trec_topic_path: Union[Path, str]  = eval_config.trec_topic_path
        self.kz_topic_path: Union[Path, str] = eval_config.kz_topic_path
        self.max_topics: int = eval_config.max_topics
        self.rel_dict: dict = None
        self.topic2text: dict = None
        self.ctm = None
        self.openai_api_key = eval_config.open_ai_api_key
   
        assert self.rel_paths is not None, "paths to relevancy judgments must be set in model_config if model_config.evaluate=True"
        assert ((self.trec_topic_path is not None) or (self.kz_topic_path is not None)), "at least one of trec_topic_path or kz_topic_path) must be set in model_config if model_config.evaluate=True"
    
        self.setup()

    def get_combined_rel_dict(self, rel_paths: List[str]) -> dict:
        combined_rel_dict = dict()
        for rel_path in rel_paths:
            with open(rel_path, 'r') as f:
                for line in f.readlines():
                    topic_id, _, doc_id, rel = line.split()
                    if topic_id not in combined_rel_dict:
                        combined_rel_dict[topic_id] = dict()
                    combined_rel_dict[topic_id][doc_id] = int(rel)
        return combined_rel_dict
    
    def setup(self):
        self.rel_dict = self.get_combined_rel_dict(self.rel_paths)
        self.topicid2text = dict()
        if self.kz_topic_path is not None:
            self.topicid2text = get_kz_topic2text(self.kz_topic_path)

        if self.trec_topic_path is not None:
            self.topicid2text.update(get_trec_topic2text(self.trec_topic_path))

        # loads all remaining needed datasets into memory
        model_config = ModelConfig(
            openai_api_key=self.openai_api_key,
        )
        self.ctm = CTMatch(model_config=model_config)



    def evaluate(self):
        """
        desc: run the pipeline over every topic and associated labelled set of documents,
                and compute the mean mrr over all topics (how far down to the first relevant document)
        """
        frrs, f1s, fprs = [], [], []
        for i, (topic_id, topic_text) in enumerate(tqdm(self.topicid2text.items())):
            if i > self.max_topics:
                logger.info(f"max topics reached: {self.max_topics}, returning early")
                break
            
            if topic_id not in self.rel_dict:
                # can't evaluate with no judgments
                continue
                
            doc_ids = list(self.rel_dict[topic_id].keys())
            logger.info(f"number of ranked docs: {len(doc_ids)}")
            doc_set = self.get_indexes_from_ids(doc_ids)
            pipe_topic = self.ctm.get_pipe_topic(topic_text)
            
            # ranking = self.ctm.sim_filter(pipe_topic, doc_set)
            # ranking = self.ctm.svm_filter(pipe_topic, doc_set)
            # ranking = self.ctm.classifier_filter(pipe_topic, doc_set)
            ranking = self.ctm.gen_filter(pipe_topic, doc_set)
            # ranking = self.ctm.match_pipeline(topic_text, doc_set=doc_set)
    
            ranked_ids = [self.ctm.data.index2docid.iloc[r].values[0] for r in ranking]
            fpr, frr = calc_first_positive_rank(ranked_ids, self.rel_dict[topic_id])
            f1 = calc_f1(ranked_ids, self.rel_dict[topic_id])

            fprs.append(fpr)
            frrs.append(frr)
            f1s.append(f1)
        
        mean_fpr = sum(fprs)/len(fprs)
        mean_frr = sum(frrs)/len(frrs)
        mean_f1 = sum(f1s)/len(f1s)

        return {"mean_fpr":mean_fpr, "mean_frr":mean_frr, "mean_f1":mean_f1}


    def get_indexes_from_ids(self, doc_id_set: List[str]) -> List[int]:
        """
        desc:       get the indexes of the documents in doc_id_set in the order they appear in the ranking
        returns:    list of indexes
        """
        doc_indices = []
        for doc_id in doc_id_set:
            index_row = np.where(self.ctm.data.index2docid['text'] == doc_id)
            if len(index_row[0]) == 0:
                continue
            doc_indices.append(index_row[0][0])
        return doc_indices








            
        