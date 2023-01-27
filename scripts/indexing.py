from drive.MyDrive.trec21_ct_full.trec_ct.scripts.trec21_vis import get_relled
from sklearn.metrics import confusion_matrix
from pyserini.search import SimpleSearcher
from sklearn.metrics import ndcg_score
from collections import defaultdict
import numpy as np
import copy
import json
import os

"""
functions for indexing documents, ranking

"""



def get_top_N(N, index_path, topic_text, val_ids, score_dict, topic_id, inc_or_exc, rel_dict, id2topic, search_max=1500):
  searcher = SimpleSearcher(index_path)
  searcher.set_bm25(0.9, 0.4)
  #searcher.set_rm3(10, 10, 0.5)
  hits = searcher.search(topic_text, search_max)
  i = 0
  num_found = 0
  all_found = 0
  ones_found, twos_found = 0,0
  new_hits = []
  total_qrelled = len(rel_dict[topic_id])
  #print(f"Total QRelled Docs for this topic: {total_qrelled}, total valid: {len(val_ids)}")
  totals = get_relled(topic_id, rel_dict)
  total_2s = totals['twos']
  total_1s = totals['ones']
  
  #print(f"Total 2s: {total_2s}, total 1s: {total_1s}")
  

  RR1, RR2 = 0, 0
  recall_100_1, recall_1000_1 = 0, 0
  recall_100_2, recall_1000_2 = 0, 0
  RR10_1, RR10_2, RR100_1, RR100_2 = 0, 0, 0, 0
  while (i < len(hits)) and (num_found < N):
    rel = 0
    if (topic_id in rel_dict) and (hits[i].docid in rel_dict[topic_id]):
      rel = rel_dict[topic_id][hits[i].docid]
      if rel == 1:
        ones_found += 1
        if RR1 == 0:
          RR1 = i + 1

        if (i <= 10) and (RR10_1 == 0):
          RR10_1 = i + 1

        if (i <= 100):
          recall_100_1 += 1


          if RR100_1 == 0:
            RR100_1 = i + 1
        
        if (i < 1000):
          recall_1000_1 += 1



      elif rel == 2:
        twos_found += 1
        if RR2 == 0:
          RR2 = i + 1

        if (i <= 10) and (RR10_2 == 0):
          RR10_2 = i + 1

        if i <= 100:
          recall_100_2 += 1
          if RR100_2 == 0:
            RR100_2 = i + 1

        if i <= 1000:
          recall_1000_2 += 1

        


    if (hits[i].docid in rel_dict[topic_id]) and  (rel > 0):
      #if i < 10:
        #print(f"hits rank: {i}, topic_id: {topic_id}, doc_id: {hits[i].docid}, actual rel: {rel_dict[topic_id][hits[i].docid]}, score={hits[i].score}, relevance? {hits[i].docid in val_ids}")
      all_found += 1

    
    if hits[i].docid in val_ids:
      score_dict[topic_id][hits[i].docid][inc_or_exc] = hits[i].score
      num_found += 1
      new_hits.append(hits[i].docid)
    
    i += 1
 
  
  recall_any = 1.0 if (total_qrelled == 0) else (num_found / total_qrelled)
  twos_recall = 1.0 if (len(total_2s) == 0) else (twos_found / len(total_2s))
  ones_recall = 1.0 if (len(total_1s) == 0) else (ones_found / len(total_1s))
  RR1 =  RR1 if (RR1 == 0) else (1. / RR1)
  RR2 =  RR2 if (RR2 == 0) else (1. / RR2)
  RR10_1 = RR10_1 if (RR10_1 == 0) else (1. / RR10_1)
  RR10_2 = RR10_2 if (RR10_2 == 0) else (1. / RR10_2)
  RR100_1 = RR100_1 if (RR100_1 == 0) else (1. / RR100_1)
  RR100_2 = RR100_2 if (RR100_2 == 0) else (1. / RR100_2)
  recall_100_1 = 1.0 if (len(total_1s) == 0) else  (recall_100_1 / len(total_1s))
  recall_100_2 = 1.0 if (len(total_2s) == 0) else  (recall_100_2 / len(total_2s))
  recall_1000_1 = 1.0 if (len(total_1s) == 0) else  (recall_1000_1 / len(total_1s))
  recall_1000_2 = 1.0 if (len(total_2s) == 0) else  (recall_1000_2 / len(total_2s))
  #print(f"Number found: {num_found}, out of all {len(hits)} hits, with N = {N}, out of total qrelled: {recall_any}, all found: {all_found}")
  recalls = {
    "twos_recall":twos_recall, 
    "ones_recall":ones_recall, 
    "rr1": RR1, 
    "rr2": RR2, 
    "rr10_1":RR10_1, 
    "rr10_2":RR10_2,
    "rr100_1":RR100_1,
    "rr100_2":RR100_2,
    "recall_100_1": recall_100_1,
    "recall_1000_1": recall_1000_1,
    "recall_100_2": recall_100_2,
    "recall_1000_2":recall_1000_2
  }
  #print(recalls.items())
  return new_hits, recalls
    



def no_rel_filter_by_topic(test_rel_dict, filtered_docs_by_topic):
  new_filtered_docs_by_topic = {}
  for topic_id in filtered_docs_by_topic.keys():
    rel_set = set(test_rel_dict[topic_id].keys())
    new_filtered_docs_by_topic[topic_id] = rel_set and filtered_docs_by_topic[topic_id]
  return new_filtered_docs_by_topic
    




def create_indexes(input_folder, index_folder):
  os.system(f"sudo python -m pyserini.index -collection JsonCollection -generator DefaultLuceneDocumentGenerator -threads 1 -input {input_folder} -index {index_folder} -storePositions -storeDocvectors -storeRaw")











#--------------------------------------------------------------------------------#
# Evaluation
#--------------------------------------------------------------------------------#



def get_total_rel_counts(topic, test_rel_dict):
  total_counts = {0:0, 1:0, 2:0}
  for rtopic in test_rel_dict:
    for rdoc in test_rel_dict[rtopic]:
      total_counts[test_rel_dict[rtopic][rdoc]] += 1
  return total_counts
  

def get_misses_counts(misses, test_rel_dict):
  all_rel_misses_counts = {1:0, 2:0}
  all_total_rel_counts = {1:0, 2:0}
  for topic, rel_dict in misses.items():
    rel_misses_counts = {1:0, 2:0}
    for rel in rel_dict.keys():
      rel_misses_counts[rel] += len(misses[topic][rel])
      
      
    total_counts = get_total_rel_counts(topic, test_rel_dict)
    all_rel_misses_counts[1] += rel_misses_counts[1]
    all_rel_misses_counts[2] += rel_misses_counts[2]
    all_total_rel_counts[1] += total_counts[1]
    all_total_rel_counts[2] += total_counts[2]
    print(topic, rel_misses_counts[1]/total_counts[1], rel_misses_counts[2]/total_counts[2])
  return all_rel_misses_counts, all_total_rel_counts






def create_score_dict(filtered_docs_by_topic):
  score_dict = defaultdict(lambda: defaultdict( lambda: defaultdict(float)))
  for topic_id, doc_set in filtered_docs_by_topic.items():
    for doc in doc_set:
      score_dict[topic_id][doc]['include_score'] = 0.0
      score_dict[topic_id][doc]['exclude_score'] = 0.0
  return score_dict



def evaluate(topics, filtered_docs_by_topic, test_rel_dict, index_path, inc_or_exc, id2doc, id2topic, score_dict=None):
  ndcg_scores = []
  conf_mat = np.zeros((3, 3), dtype=int)
  RR_any_list, RR_1_list, RR_2_list = [], [], []      # for reciprocal ranks
  RR10_any_list, RR100_any_list = [], []
  recall_100_1_list, recall_1000_1_list = [], []
  recall_100_2_list, recall_1000_2_list = [], []
  recall_100_any_list, recall_1000_any_list = [], [] 
  score_dict = create_score_dict(filtered_docs_by_topic) if (score_dict is None) else score_dict  # for every topic, doc pair, score
  search_rel_recalls = {"twos_recalls":[], "ones_recalls":[]}
  for topic in topics:
    n_try = len(filtered_docs_by_topic[topic['id']]) + 2000
    hits, rel_recall = get_top_N(
                              N=n_try, 
                              index_path=index_path, 
                              topic_text=topic['contents'], 
                              val_ids=filtered_docs_by_topic[topic['id']], 
                              topic_id=topic["id"],
                              inc_or_exc=inc_or_exc,
                              rel_dict=test_rel_dict,
                              score_dict=score_dict,
                              id2topic=id2topic
                       )
    search_rel_recalls["twos_recalls"].append(rel_recall["twos_recall"])
    search_rel_recalls["ones_recalls"].append(rel_recall["ones_recall"])
    cuts = get_cuts(test_rel_dict, topic['id'])
    actual_relevances = []
    predict_dict = {}
    
    RR_1_list.append(rel_recall["rr1"])
    RR_2_list.append(rel_recall["rr2"])
    RR_any_list.append(max(rel_recall['rr1'], rel_recall['rr2']))
    RR10_any_list.append(max(rel_recall['rr10_1'], rel_recall['rr10_2']))
    RR100_any_list.append(max(rel_recall['rr100_1'], rel_recall['rr100_2']))
    recall_100_1_list.append(rel_recall["recall_100_1"])
    recall_1000_1_list.append(rel_recall["recall_1000_1"])
    recall_100_2_list.append(rel_recall["recall_100_2"])
    recall_1000_2_list.append(rel_recall["recall_1000_2"])
    recall_100_any_list.append(max(rel_recall["recall_100_1"], rel_recall["recall_100_2"]))
    recall_1000_any_list.append(max(rel_recall["recall_1000_1"], rel_recall["recall_1000_2"]))


    for i, doc_id in enumerate(hits):
      if doc_id in test_rel_dict:
        actual_rel = test_rel_dict[doc_id]
      else:
        actual_rel = 0                       # assumption that pooling eliminated irrelevant docs... strong assumption!!
      actual_relevances.append(actual_rel)
    
    predicted_relevances = [2 for _ in range(cuts[2])] + [1 for _ in range(cuts[1])] + [0 for _ in range(len(hits) - cuts[2] - cuts[1])]

    new_conf_mat = np.asarray(confusion_matrix(actual_relevances, predicted_relevances))
    if new_conf_mat.shape[0] == 3:
      conf_mat = np.add(conf_mat, new_conf_mat)

    

    num_not_zero = sum([r for r in actual_relevances if r != 0])
    
    if len(actual_relevances) > 0:
      ndcg = ndcg_score(np.asarray([actual_relevances]), np.asarray([predicted_relevances]))
      #print(f"topic_id: {topic['id']}, ndcg score: {ndcg}")
      ndcg_scores.append(ndcg)

  MRR_1 = sum(RR_1_list) / len(RR_1_list)
  MRR_2 = sum(RR_2_list) / len(RR_2_list)
  MRR = sum(RR_any_list) / len(RR_any_list)
  MRR10 = sum(RR10_any_list) / len(RR10_any_list)
  MRR100 = sum(RR100_any_list) / len(RR100_any_list)
  twos_recall_search_mean = sum(search_rel_recalls['twos_recalls']) / len(search_rel_recalls['twos_recalls'])
  ones_recall_search_mean = sum(search_rel_recalls['ones_recalls']) / len(search_rel_recalls['ones_recalls'])
  recall_at_100_1 = sum(recall_100_1_list) / len(recall_100_1_list)
  recall_at_100_2 = sum(recall_100_2_list) / len(recall_100_2_list)
  recall_100 = sum(recall_100_any_list) / len(recall_100_any_list)
  recall_at_1000_1 = sum(recall_1000_1_list) / len(recall_1000_1_list)
  recall_at_1000_2 = sum(recall_1000_2_list) / len(recall_1000_2_list)
  recall_1000 = sum(recall_1000_any_list) / len(recall_1000_any_list)

  print(f"Mean Reciprocal Rank (MRR) across all topic:doc relevance-scored pairs: MRR_1 = {MRR_1}, MRR_2 = {MRR_2}, MRR = {MRR}")
  print(f"Mean Reciprocal Rank (MRR) across all topic:doc relevance-scored pairs: MRR@10 = {MRR10}, MRR@100 = {MRR100}")
  print(f"Mean Recall For Total Search Max of 1000, 2s Recall: {twos_recall_search_mean}, 1s Recall: {ones_recall_search_mean}")
  print(f"Mean recall@100_1: {recall_at_100_1}, mean recall@1000_1: {recall_at_1000_1}")
  print(f"Mean recall@100_2: {recall_at_100_2}, mean recall@1000_2: {recall_at_1000_2}")
  
  mean_ndcg = sum(ndcg_scores) / len(ndcg_scores)
  print(f"meand ndcg score: {mean_ndcg}")
  print("confusion matrix:")
  print(conf_mat)
  
  return score_dict


def get_cuts(test_rel_dict, topic_id):
  rel_counts = defaultdict(int)
  for doc_id, rel in test_rel_dict[topic_id].items():
    rel_counts[rel] += 1
  return rel_counts



def examine_coverage(scores, filtered_docs_by_topic, test_rel_dict):
  missing_pairs = defaultdict(lambda: defaultdict(int))
  misses = defaultdict(lambda: defaultdict(list))
  total_2s, total_1s = 0, 0
  missed_2s, missed_1s = 0, 0
  for topic_id in scores.keys():
    missing_inc, missing_exc = 0, 0
    for doc_id in scores[topic_id].keys():
      rel = test_rel_dict[topic_id][doc_id]
      if rel > 1:
        #print(scores[topic_id][doc_id]["include_score"])
        total_2s += 1
      elif rel > 0:
        total_1s += 1

      if scores[topic_id][doc_id]["include_score"] == 0:
        missing_inc += 1
        if rel > 1:
          missed_2s += 1
          misses[topic_id][rel].append(doc_id)
        if rel > 0:
          missed_1s +=1
          misses[topic_id][rel].append(doc_id)
        

      """
      if scores[topic_id][doc_id]["exclude_score"] == 0:
        missing_exc += 1
        rel = test_rel_dict[topic_id][doc_id]
        if rel > 0:
          misses[topic_id][rel].append(doc_id)
      """

    missing_pairs[topic_id]["missing_inc"] = missing_inc
    missing_pairs[topic_id]["missing_exc"] = missing_exc

  recall_2s = (total_2s - missed_2s) / total_2s
  recall_1s = (total_1s - missed_1s) / total_1s
  print(f"recall 2s: {recall_2s}, recall_1s: {recall_1s}")
  return missing_pairs, misses

  

def display_errors(misses, id2topic, id2doc):
  for topic_id, rel_dict in misses.items():
    print(f"Topic ID: {topic_id}")
    print(id2topic[topic_id]['raw_text'])
    for doc_id in rel_dict[2]:
      print(f"Doc ID: {doc_id}")
      print(id2doc[doc_id]['eligibility/criteria/textblock']["raw_text"])


