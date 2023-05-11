
import logging
from typing import List, NamedTuple, Union

from .utils.ctmatch_utils import get_kz_topic2text, get_trec_topic2text
from .utils.eval_utils import calc_mrr
from .match import CTMatch
from pathlib import Path

logger = logging.getLogger(__name__)


class EvaluatorConfig(NamedTuple):
    rel_paths: List[str]
    trec_topic_path: Union[Path, str]  = None
    kz_topic_path: Union[Path, str] = None


class Evaluator:
    def __init__(self, eval_config: EvaluatorConfig) -> None:
        self.rel_paths: List[str] = eval_config.rel_paths
        self.trec_topic_path: Union[Path, str]  = eval_config.trec_topic_path
        self.kz_topic_path: Union[Path, str] = eval_config.kz_topic_path
        self.rel_dict: dict = None
        self.topic2text: dict = None
   
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
            self.topicid2text = get_kz_topic2text(self.trec_topic_path, self.kz_topic_path)

        if self.trec_topic_path is not None:
            self.topicid2text.update(get_trec_topic2text(self.trec_topic_path))


    def evaluate(self):
        """
        desc: run the pipeline over every topic and associated labelled set of documents,
              and compute the mean mrr over all topics (how far down to the first relevant document)
        """
        mrrs = []
        for topicid, topic_text in self.topicid2text.items():
            doc_set = list(self.rel_dict[topicid].keys())
            ranking = CTMatch().match_pipeline(topic_text, doc_set)
            mrr = calc_mrr(ranking, self.rel_dict[topicid])
            mrrs.append(mrr)
        
        mean_mrr = sum(mrrs)/len(mrrs)
        logger.info(f"mean mrr: {mean_mrr}")








            
        