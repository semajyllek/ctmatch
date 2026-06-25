	
from typing import Dict, NamedTuple, Tuple
from utils.ctmatch_utils import get_processed_data
from collections import defaultdict
import ct_data_paths
import random

from ctproc.scripts.vis_scripts import (
  analyze_test_rels
)


class ExplorePaths(NamedTuple):
	doc_path: str
	topic_path: str
	rel_path: str



# --------------------------------------------------------------------------------------------------------------- #
# EDA functions
# --------------------------------------------------------------------------------------------------------------- #

def explore_kz_data(rand_print: float = 0.001) -> None:
	kz_data_paths = ExplorePaths(
		rel_path = ct_data_paths.KZ_REL_PATH,
		doc_path = ct_data_paths.KZ_PROCESSED_DOC_PATH,
		topic_path = ct_data_paths.KZ_RELLED_TOPIC_PATH
	)
	
	explore_data(kz_data_paths, rand_print=rand_print)


def explore_trec_data(part: int = 1, rand_print: float = 0.001) -> None:
	# post processing analysis
	trec_data_paths = ExplorePaths(
		rel_path = ct_data_paths.TREC_REL_PATH,
		doc_path = f'/Users/jameskelly/Documents/cp/ctmatch/data/trec_data/processed_trec_data/processed_trec22_docs_part{part}.jsonl',
		topic_path = ct_data_paths.TREC_RELLED_TOPIC_PATH
	)
	
	explore_data(trec_data_paths, rand_print=rand_print)



def explore_data(data_paths: ct_data_paths.ExplorePaths, rand_print: float) -> None:

	# process relevancy judgements
	type_dict, rel_dict, all_qrelled_docs = analyze_test_rels(data_paths.rel_path)
	
	# get processed topics
	id2topic = {t['id']:t for t in get_processed_data(data_paths.topic_path)}
	print(f"number of processed topics: {len(id2topic)}")

	# get relevant processed docs
	id2docs = {doc['id']:doc for doc in get_processed_data(data_paths.doc_path, get_only=all_qrelled_docs)}
	print(f"number of relevant processed docs: {len(id2docs)}")

	explore_pairs(id2topic, id2docs, rel_dict, max_print=1000, rand_print=rand_print)





def explore_pairs(id2topic: Dict[str, Dict[str, str]], id2docs: Dict[str, Dict[str, str]], rel_dict: Dict[str, Dict[str, str]], rand_print: float, max_print:int = 100000) -> None:
	rel_scores = defaultdict(int)
	age_mismatches, gender_mismatches = 0, 0
	for pt_id, topic in id2topic.items():
		for doc_id in rel_dict[pt_id]:
			if doc_id in id2docs:
				rel_score = rel_dict[pt_id][doc_id]
				rel_scores[rel_score] += 1
				if rel_score == 2:
					age_mismatches, gender_mismatches = check_match(
						topic = topic, 
						doc = id2docs[doc_id], 
						rel_score = rel_score,
						age_mismatches = age_mismatches, 
						gender_mismatches = gender_mismatches
					)

				if random.random() < rand_print:
					print_pair(topic, id2docs[doc_id], rel_score, marker='%')

	print(rel_scores.items())
	print(f"{age_mismatches=}, {gender_mismatches=}")




def check_match(topic: Dict[str, str], doc: Dict[str, str], rel_score: int, age_mismatches: int, gender_mismatches: int) -> Tuple[int, int]:
	age_matches = age_match(doc['elig_min_age'], doc['elig_max_age'], topic['age'])
	if not age_matches:
		#print_pair(topic, doc, rel_score)
		age_mismatches += 1

	gender_matches = gender_match(doc['elig_gender'], topic['gender'])
	if not gender_matches:
		#print_pair(topic, doc, rel_score)
		gender_mismatches += 1
	
	return age_mismatches, gender_mismatches


					
def print_pair(topic: Dict[str, str], doc: Dict[str, str], rel_score: int, marker: str = '*') -> None:
	print(marker*200)
	print(f"topic id: {topic['id']}, nct_id: {doc['id']}, rel score: {rel_score}")
	print(f"topic info: \nage: {topic['age']}, gender: {topic['gender']}")
	print(topic['raw_text'])
	print(f"doc info: gender: {doc['elig_gender']}, min age: {doc['elig_min_age']}, max age: {doc['elig_max_age']}")
	print(doc['elig_crit']['raw_text'])
	print(marker*200)
	print()
