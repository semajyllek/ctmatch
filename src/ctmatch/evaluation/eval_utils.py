
from typing import Dict, List, Tuple

from sklearn.metrics import f1_score
from collections import defaultdict
from lxml import etree
import numpy as np

  
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