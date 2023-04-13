
from typing import Dict, List, NamedTuple, Optional, Set, Tuple, Union
from collections import defaultdict
from pprint import pprint
import random
import json

from ctmatch_utils import get_processed_data, truncate
from ctproc import CTConfig, CTProc, CTDocument, CTTopic
from scripts.vis_scripts import (
  analyze_test_rels
)


TREC_REL_PATH =  "/Users/jameskelly/Documents/cp/ctmatch/data/trec_data/trec_21_judgments.txt"
KZ_REL_PATH =  "/Users/jameskelly/Documents/cp/ctmatch/data/kz_data/qrels-clinical_trials.txt"

TREC_RELLED_TOPIC_PATH = "/Users/jameskelly/Documents/cp/ctmatch/data/trec_data/processed_trec_data/processed_trec21_topics.jsonl" 
KZ_RELLED_TOPIC_PATH = '/Users/jameskelly/Documents/cp/ctmatch/data/kz_data/processed_kz_data/processed_kz_topics.jsonl' 

KZ_DOC_PATH = '/Users/jameskelly/Documents/cp/ctmatch/data/kz_data/clinicaltrials.gov-16_dec_2015.zip'
KZ_PROCESSED_DOC_PATH = '/Users/jameskelly/Documents/cp/ctmatch/data/kz_data/processed_kz_data/processed_kz_docs.jsonl'

KZ_PREP_PATH = '/Users/jameskelly/Documents/cp/ctmatch/data/kz_data/kz_labelled_multifield.jsonl'
TREC_PREP_PATH = '/Users/jameskelly/Documents/cp/ctmatch/data/trec_data/trec21_labelled_multifield.jsonl'

TREC_ML_PATH = '/Users/jameskelly/Documents/cp/ctmatch/data/trec_data/trec_data.jsonl'
KZ_ML_PATH = '/Users/jameskelly/Documents/cp/ctmatch/data/kz_data/kz_data.jsonl'


class ExplorePaths(NamedTuple):
	doc_path: str
	topic_path: str
	rel_path: str


class DataConfig(NamedTuple):
	save_path: str
	trec_or_kz: str = 'trec'
	filtered_topic_keys: Set[str] = {'id', 'text_sents', 'age', 'gender'}
	filtered_doc_keys: Set[str] = {'id', 'elig_min_age', 'elig_max_age', 'elig_gender', 'condition', 'elig_crit'}
	max_topic_len: Optional[int] = None
	max_inc_len: Optional[int] = None
	max_exc_len: Optional[int] = None
	prepend_elig_age: bool = True
	prepend_elig_gender: bool = True
	include_only: bool = False
	sep: str = '[SEP]'


def proc_docs_and_topics(trec_or_kz: str = 'trec') -> Tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, str]]]:

	doc_tuples, topic_tuples = get_data_tuples(trec_or_kz)

	id2topic = dict()
	for topic_source, topic_target in topic_tuples:
		id2topic.update(proc_topics(topic_source, topic_target, trec_or_kz=trec_or_kz))
		print(f"processed {trec_or_kz} topic source: {topic_source}, and wrote to {topic_target}")
	
	id2doc = dict()
	for doc_source, doc_target in doc_tuples:
		id2doc.update(proc_docs(doc_source, doc_target))
		print(f"processed {trec_or_kz} doc source: {doc_source}, and wrote to {doc_target}")


	return id2topic, id2doc





def proc_docs(doc_path: str, output_path: str) -> Dict[str, CTDocument]:

	ct_config = CTConfig(
		data_path=doc_path, 
		write_file=output_path,
    	nlp=True
	)

	cp = CTProc(ct_config)
	id2doc = {res.id : res for res in cp.process_data()}
	return id2doc



def proc_topics(topic_path: str, output_path: str, trec_or_kz: str = 'trec') -> Dict[str, CTTopic]:
	
	ct_config = CTConfig(
		data_path=topic_path, 
		write_file=output_path,
    	nlp=True,
    	is_topic=True,
		trec_or_kz=trec_or_kz
	)

	cp = CTProc(ct_config)
	id2topic = {res.id : res for res in cp.process_data()}
	return id2topic




def get_data_tuples(trec_or_kz: str = 'trec') -> Tuple[Tuple[str, str], Tuple[str, str]]:
	if trec_or_kz == 'trec':
		return get_trec_doc_data_tuples(), get_trec_topic_data_tuples()
	return get_kz_doc_data_tuples(), get_kz_topic_data_tuples()


# --------------------------------------------------------------------------------------------------------------- #
# data from TREC clinical track 2021 & 2022
# --------------------------------------------------------------------------------------------------------------- #


def get_trec_doc_data_tuples() -> List[Tuple[str]]:
	trec22_pt1_docs = '/Users/jameskelly/Documents/cp/ctmatch/data/trec_data/trec_docs_21/ClinicalTrials.2021-04-27.part1.zip'
	trec_pt1_target = '/Users/jameskelly/Documents/cp/ctmatch/data/trec_data/processed_trec_data/processed_trec22_docs_part1.jsonl'

	trec22_pt2_docs = '/Users/jameskelly/Documents/cp/ctmatch/data/trec_data/trec_docs_21/ClinicalTrials.2021-04-27.part2.zip'
	trec_pt2_target = '/Users/jameskelly/Documents/cp/ctmatch/data/trec_data/processed_trec_data/processed_trec22_docs_part2.jsonl'

	trec22_pt3_docs = '/Users/jameskelly/Documents/cp/ctmatch/data/trec_data/trec_docs_21/ClinicalTrials.2021-04-27.part3.zip'
	trec_pt3_target = '/Users/jameskelly/Documents/cp/ctmatch/data/trec_data/processed_trec_data/processed_trec22_docs_part3.jsonl'

	trec22_pt4_docs = '/Users/jameskelly/Documents/cp/ctmatch/data/trec_data/trec_docs_21/ClinicalTrials.2021-04-27.part4.zip'
	trec_pt4_target = '/Users/jameskelly/Documents/cp/ctmatch/data/trec_data/processed_trec_data/processed_trec22_docs_part4.jsonl'

	trec22_pt5_docs = '/Users/jameskelly/Documents/cp/ctmatch/data/trec_data/trec_docs_21/ClinicalTrials.2021-04-27.part5.zip'
	trec_pt5_target = '/Users/jameskelly/Documents/cp/ctmatch/data/trec_data/processed_trec_data/processed_trec22_docs_part5.jsonl'

	trec_doc_data_tuples = [
		(trec22_pt1_docs, trec_pt1_target), 
		(trec22_pt2_docs, trec_pt2_target), 
		(trec22_pt3_docs, trec_pt3_target),
		(trec22_pt4_docs, trec_pt4_target),
		(trec22_pt5_docs, trec_pt5_target)
	]

	return trec_doc_data_tuples


def get_trec_topic_data_tuples() -> List[Tuple[str]]:
	trec21_topic_path = '/Users/jameskelly/Documents/cp/ctmatch/data/trec_data/trec_21_topics.xml'
	trec21_topic_target = '/Users/jameskelly/Documents/cp/ctmatch/data/trec_data/processed_trec_data/processed_trec21_topics.jsonl'
	trec22_topic_path = '/Users/jameskelly/Documents/cp/ctmatch/data/trec_data/trec_22_topics.xml'
	trec22_topic_target = '/Users/jameskelly/Documents/cp/ctmatch/data/trec_data/processed_trec_data/processed_trec22_topics.jsonl'

	trec_topic_data_tuples = [
		(trec21_topic_path, trec21_topic_target), 
		(trec22_topic_path, trec22_topic_target)
	]
	return trec_topic_data_tuples




# --------------------------------------------------------------------------------------------------------------- #
# data from Koontz, et al. (2016)
# --------------------------------------------------------------------------------------------------------------- #
def get_kz_doc_data_tuples() -> List[Tuple[str]]:
	# kz_doc_data_tuples = []
	# for i in range(1, 18):
	# 	kz_doc_path = f'/Users/jameskelly/Documents/cp/ctmatch/data/kz_data/kz_doc_splits/kz_doc_split{i}.zip'
	# 	kz_doc_target = f'/Users/jameskelly/Documents/cp/ctmatch/data/kz_data/processed_kz_data/processed_kz_doc_split{i}.jsonl'
	# 	kz_doc_data_tuples.append((kz_doc_path, kz_doc_target))
	kz_docs = KZ_DOC_PATH
	kz_docs_target = KZ_PROCESSED_DOC_PATH
	return [(kz_docs, kz_docs_target)]
	
	#return kz_doc_data_tuples

def get_kz_topic_data_tuples() -> List[Tuple[str]]:
	kz_topic_desc_path = '/Users/jameskelly/Documents/cp/ctmatch/data/kz_data/topics-2014_2015-description.topics'
	kz_topic_target = '/Users/jameskelly/Documents/cp/ctmatch/data/kz_data/processed_kz_data/processed_kz_topics.jsonl'
	kz_topic_data_tuples = [
		(kz_topic_desc_path, kz_topic_target)
	]
	return kz_topic_data_tuples





	
# --------------------------------------------------------------------------------------------------------------- #
# EDA functions
# --------------------------------------------------------------------------------------------------------------- #

def explore_kz_data(rand_print: float = 0.001) -> None:
	kz_data_paths = ExplorePaths(
		rel_path = KZ_REL_PATH,
		doc_path = KZ_PROCESSED_DOC_PATH,
		topic_path = KZ_RELLED_TOPIC_PATH
	)
	
	explore_data(kz_data_paths, rand_print=rand_print)


def explore_trec_data(part: int = 1, rand_print: float = 0.001) -> None:
	# post processing analysis
	trec_data_paths = ExplorePaths(
		rel_path = TREC_REL_PATH,
		doc_path = f'/Users/jameskelly/Documents/cp/ctmatch/data/trec_data/processed_trec_data/processed_trec22_docs_part{part}.jsonl',
		topic_path = TREC_RELLED_TOPIC_PATH
	)
	
	explore_data(trec_data_paths, rand_print=rand_print)



def explore_data(data_paths: ExplorePaths, rand_print: float) -> None:

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


# --------------------------------------------------------------------------------------------------------------- #
# utility functions
# --------------------------------------------------------------------------------------------------------------- #

def age_match(min_doc_age: float, max_doc_age: float, topic_age: float) -> bool:
	if topic_age < min_doc_age:
		return False
	if topic_age > max_doc_age:
		return False
	return True

def gender_match(doc_gender: str, topic_gender: str) -> bool:
	if doc_gender == 'All':
		return True
	if doc_gender == topic_gender:
		return True
	return False


def get_topic_and_rel_path(trec_or_kz: str = 'trec') -> Tuple[str, str]:
	if trec_or_kz == 'trec':
		rel_path = TREC_REL_PATH
		topic_path = TREC_RELLED_TOPIC_PATH
	else:
		rel_path = KZ_REL_PATH
		topic_path = KZ_RELLED_TOPIC_PATH
	return topic_path, rel_path


def get_doc_and_topic_mappings(all_qrelled_docs: Set[str], doc_tuples: List[Tuple[str, str]], topic_path: str) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, str]]]:
	"""
	desc: get mappings of doc ids to doc dicts and topic ids to topic dicts
	"""
	
	# get all processed topics
	id2topic = {t['id']:t for t in get_processed_data(topic_path)}

	# get all processed docs
	id2doc = dict()
	for _, processed_doc_path in doc_tuples:
		print(f"gettting docs from {processed_doc_path}")
		for doc in get_processed_data(processed_doc_path):
			if doc['id'] in all_qrelled_docs:
				id2doc[doc['id']] = doc
	
	return id2doc, id2topic



def prep_topic_text(topic: Dict[str, Union[List[str], str, float]], dconfig: DataConfig) -> str:
	topic_text = ' '.join(topic['text_sents'])
	topic_text = truncate(topic_text, dconfig.max_topic_len)
	return topic_text

def prep_doc_text(doc: Dict[str, Union[List[str], str, float]], dconfig: DataConfig) -> str:

	# combine lists of strings into single string
	doc_inc = ' '.join(doc['elig_crit']['include_criteria'])
	doc_exc = ' '.join(doc['elig_crit']['exclude_criteria'])


	if 'condition' in dconfig.filtered_doc_keys:
		doc_inc = f"{' '.join(doc['condition'])} {doc_inc}"

	#truncate criteria separately if in config
	doc_inc = truncate(doc_inc, dconfig.max_inc_len)
	doc_exc = truncate(doc_exc, dconfig.max_exc_len)


	if dconfig.prepend_elig_gender:
		doc_inc = f"{doc['elig_gender']} {dconfig.sep} {doc_inc}"

	if dconfig.prepend_elig_age:
		doc_inc = f"{doc['elig_min_age']}-{doc['elig_max_age']} {dconfig.sep} {doc_inc}"

	# combine criteria into single string
	if dconfig.include_only:
		return doc_inc
	return f"{doc_inc} {dconfig.sep} {doc_exc}"

	
def create_combined_doc(
	doc, topic, 
	rel_score, 
	dconfig: DataConfig
):
	combined = dict()

	# get filtered and truncated and SEP tokenized topic text
	combined['topic'] = prep_topic_text(topic, dconfig)

	# get filtered and truncated and SEP tokenized doc text
	combined['doc'] = prep_doc_text(doc, dconfig)

	# get relevancy score as string 
	combined['relevancy_score'] = str(rel_score)

	return combined


def save_relled_dataset(
		dconfig: DataConfig
) -> None:
	"""
	trec_or_kz: 'trec' or 'kz'
	desc: create dict of triplets of topic, doc, relevancy scores,
	      save into a single jsonl file
	"""
	print(f"trec_or_kz: {dconfig.trec_or_kz}")
	topic_path, rel_path = get_topic_and_rel_path(dconfig.trec_or_kz)
	

	# get set of all relevant doc ids
	_, rel_dict, all_qrelled_docs = analyze_test_rels(rel_path)
	
	# get path to processed docs (already got topic path)
	doc_tuples, _ = get_data_tuples(dconfig.trec_or_kz)

	# get mappings of doc ids to doc dicts and topic ids to topic dicts
	id2doc, id2topic = get_doc_and_topic_mappings(all_qrelled_docs, doc_tuples, topic_path) 
	print(len(id2doc), len(all_qrelled_docs))
	
	missing_docs = set()

	# save combined triples of doc, topic, relevancy score
	with open(dconfig.save_path, 'w') as f:
		for topic_id in rel_dict:
			for doc_id in rel_dict[topic_id]:
					if doc_id in id2doc:
						combined = create_combined_doc(
							id2doc[doc_id], id2topic[topic_id], 
							rel_dict[topic_id][doc_id], 
						    dconfig=dconfig
						)

						# save to file as jsonl
						f.write(json.dumps(combined))
						f.write('\n')
					else:
						missing_docs.add(doc_id)


	print(f"number of docs missing: {len(missing_docs)}")
	for md in missing_docs:
		print(md)


def explore_prepped(triples_path: str) -> None:
	with open(triples_path, 'r') as f:
		for i, line in enumerate(f.readlines()):
			combined = json.loads(line)
			if combined['relevancy_score'] == 2:
				if random.random() < 0.0001:
					pprint(combined)
					break
			




if __name__ == '__main__':
	# proc_docs_and_topics('trec')
	# explore_trec_data(part=2, rand_print=0.001) # select part 1-5 (~70k docs per part)
	# explore_kz_data(rand_print=0.00001) # all in one file (~200k docs)

	save_relled_dataset(DataConfig(save_path=KZ_ML_PATH, trec_or_kz='kz'))
	# save_relled_dataset(DataConfig(save_path=TREC_ML_PATH, trec_or_kz='trec'))
	# explore_prepped(TREC_PREP_PATH)


					






