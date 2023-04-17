
from typing import List, Tuple

TREC_REL_PATH =  "/Users/jameskelly/Documents/cp/ctmatch/data/trec_data/trec_21_judgments.txt"
KZ_REL_PATH =  "/Users/jameskelly/Documents/cp/ctmatch/data/kz_data/qrels-clinical_trials.txt"

TREC_RELLED_TOPIC_PATH = "/Users/jameskelly/Documents/cp/ctmatch/data/trec_data/processed_trec_data/processed_trec21_topics.jsonl" 
KZ_RELLED_TOPIC_PATH = '/Users/jameskelly/Documents/cp/ctmatch/data/kz_data/processed_kz_data/processed_kz_topics.jsonl' 

KZ_DOC_PATH = '/Users/jameskelly/Documents/cp/ctmatch/data/kz_data/clinicaltrials.gov-16_dec_2015.zip'
KZ_PROCESSED_DOC_PATH = '/Users/jameskelly/Documents/cp/ctmatch/data/kz_data/processed_kz_data/processed_kz_docs.jsonl'

TREC_ML_PATH = '/Users/jameskelly/Documents/cp/ctmatch/data/trec_data/trec_data.jsonl'
KZ_ML_PATH = '/Users/jameskelly/Documents/cp/ctmatch/data/kz_data/kz_data.jsonl'


def get_data_tuples(trec_or_kz: str = 'trec') -> List[Tuple[str, str]]:
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
