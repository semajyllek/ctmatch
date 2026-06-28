
from typing import Dict, List, Tuple

from sklearn.metrics import f1_score
from collections import defaultdict
from lxml import etree
import numpy as np
import os

  
def get_trec_topic2text(topic_path) -> Dict[str, str]:
    """
    desc:       main method for processing a single XML file of TREC21 patient descriptions called "topics" in this sense
    returns:    dict of topicid: topic text
    """

    topic2text = {}
    topic_root = etree.parse(topic_path).getroot()
    for topic in topic_root:
        topic2text[topic.attrib['number']] = topic.text
    
    return topic2text
    


def get_kz_topic2text(topic_path) -> Dict[str, str]:
    """
    desc:       main method for processing a single XML file of TREC21 patient descriptions called "topics" in this sense
    returns:    dict of topicid: topic text
    """

    topic2text = {}
    with open(topic_path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            
            if line.startswith('<TOP>'):
                topic_id, text = None, None
                continue

            if line.startswith('<NUM>'):
                topic_id = line[5:-6]

            elif line.startswith('<TITLE>'):
                text = line[7:].strip()
                topic2text[topic_id] = text

    return topic2text
          


def calc_ndcg(ranked_ids: List[str], doc2rel: Dict[str, int], k: int = 10) -> float:
    """
    NDCG@k with graded relevance (0=not relevant, 1=partial, 2=relevant).
    Unjudged docs are treated as 0. IDCG computed from the full judged set.
    """
    dcg = sum(
        (2 ** doc2rel.get(doc_id, 0) - 1) / np.log2(i + 2)
        for i, doc_id in enumerate(ranked_ids[:k])
    )
    ideal_rels = sorted(doc2rel.values(), reverse=True)[:k]
    idcg = sum((2 ** rel - 1) / np.log2(i + 2) for i, rel in enumerate(ideal_rels))
    return dcg / idcg if idcg > 0 else 0.0


def calc_first_positive_rank(ranked_ids: List[str], doc2rel: Dict[str, int], pos_val: int = 2) -> Tuple[int, float]:
    """
    desc:       compute the mean reciprocal rank of a ranking
    returns:    mrr 
    """
    for i, doc_id in enumerate(ranked_ids):
        if doc2rel[doc_id] == pos_val:
            return i + 1, 1./float(i+1)
    return len(ranked_ids) + 1, 0.0


def calc_f1(ranked_ids: List[str], doc2rel: Dict[str, int]) -> Dict[str, Dict[str, float]]:
  label_counts = get_label_counts(doc2rel)
  predicted, ground_truth = [], []
  for doc_id in ranked_ids:
    # 2, 1, 0
    ground_truth.append(doc2rel[doc_id])
    pred_label = get_predicted_label(label_counts)
    predicted.append(pred_label)
    label_counts[pred_label] -= 1

  return f1_score(ground_truth, predicted, average='micro')

  

def get_label_counts(doc2rel: Dict[str, int]) -> Dict[int, int]:
  """
  return an ordered list of [(2, <count_2s>), (1, <count_1s>), (0, count_0s)]
  """
  label_counts = defaultdict(int)
  for scored_doc in doc2rel:
    label = doc2rel[scored_doc]
    label_counts[label] += 1
  return label_counts

def get_predicted_label(label_counts: Dict[int, int]) -> int:
  if label_counts[2] > 0:
    return 2
  if label_counts[1] > 0:
    return 1
  return 0


def load_eval_datasets(trec_root: str, kz_root: str) -> Dict[str, dict]:
    """
    Load topics and qrels for TREC 2021, TREC 2022, and KZ eval sets.

    Returns a dict keyed by dataset name, each value is:
        {'topic2text': {topic_id: str}, 'rel_dict': {topic_id: {doc_id: int}}}
    """
    from ..utils.ctmatch_utils import get_test_rels

    specs = {
        'trec21': (
            os.path.join(trec_root, 'trec_21_topics.xml'),
            os.path.join(trec_root, 'trec_21_judgments.txt'),
            get_trec_topic2text,
        ),
        'trec22': (
            os.path.join(trec_root, 'topics2022.xml'),
            os.path.join(trec_root, 'qrels2022.txt'),
            get_trec_topic2text,
        ),
        'kz': (
            os.path.join(kz_root, 'topics-2014_2015-description.topics'),
            os.path.join(kz_root, 'qrels-clinical_trials.txt'),
            get_kz_topic2text,
        ),
    }
    datasets = {}
    for name, (topic_path, rel_path, parser) in specs.items():
        if os.path.exists(topic_path) and os.path.exists(rel_path):
            topic2text = parser(topic_path)
            rel_dict, _ = get_test_rels(rel_path)
            datasets[name] = {'topic2text': topic2text, 'rel_dict': rel_dict}
            n_judged = sum(len(v) for v in rel_dict.values())
            print(f'{name}: {len(topic2text)} topics, {n_judged:,} judged pairs')
        else:
            print(f'{name}: skipped (files not found at {topic_path})')
    return datasets


def load_doc_texts() -> Dict[str, str]:
    """
    Build a doc_id → text dict from the HF IR dataset (semaj83/ctmatch_ir).
    Downloads ~100K trial texts; takes roughly a minute on first call.
    """
    from ..data.dataprep import DataPrep
    from ..config import PipeConfig

    print('Loading doc texts from semaj83/ctmatch_ir...')
    dp = DataPrep(PipeConfig(ir_setup=True))
    ids = dp.index2docid['text'].tolist()
    texts = dp.doc_texts_df['text'].tolist()
    print(f'Loaded {len(ids):,} docs')
    return dict(zip(ids, texts))