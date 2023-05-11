
from typing import Dict, List
from lxml import etree

  
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

                  

def calc_mrr(ranking: List[str], doc2rel: Dict[str, int], pos_val: int = 2) -> float:
    """
    desc:       compute the mean reciprocal rank of a ranking
    returns:    mrr 
    """
    for i, doc_id in enumerate(ranking):
        if doc2rel[doc_id] == pos_val:
            return 1/(i+1)
    return 0.0
