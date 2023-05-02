
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple, Union
import ctproc_ct_data_paths as ctpaths
import numpy as np
import random
import json

from proc import CTConfig, CTProc, CTDocument, CTTopic
from scripts.vis_scripts import analyze_test_rels
from ctproc_ctmatch_utils import get_processed_data, truncate
import ctproc_eda as eda

LLM_END_PROMPT: str = "Revelance score (0, 1, or 2) : [CLS] "

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
	downsample_zeros_n: Optional[int] = None
	sep: str = '[SEP]'
	llm_prep: bool = False
	first_n_only: Optional[int] = None
	convert_snli: bool = False
	infer_category_model: Optional[str] = None




def proc_docs_and_topics(trec_or_kz: str = 'trec') -> Tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, str]]]:

	doc_tuples, topic_tuples = ctpaths.get_data_tuples(trec_or_kz)

	id2topic = dict()
	for topic_source, topic_target in topic_tuples:
		id2topic.update(proc_topics(topic_source, topic_target, trec_or_kz=trec_or_kz))
		print(f"processed {trec_or_kz} topic source: {topic_source}, and wrote to {topic_target}")
	
	id2doc = dict()
	# for doc_source, doc_target in doc_tuples:
	# 	id2doc.update(proc_docs(doc_source, doc_target))
	# 	print(f"processed {trec_or_kz} doc source: {doc_source}, and wrote to {doc_target}")


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



def filter_doc_for_ir(doc, dconfig) -> Dict[str, List[str]]:
	new_doc = dict()
	new_doc['id'] = doc['id']
	new_doc['text'] = prep_doc_text(doc, dconfig)
	return new_doc


def prep_ir_dataset(dconfig: DataConfig):
	# need a file of all docs with their 
	# 1. ids, 
	# 2. combined text... 
	# 3. 
	
	# get path to processed docs
	doc_tuples, _ = ctpaths.get_data_tuples(dconfig.trec_or_kz)
		
	# get all processed docs
	id2doc = dict()
	for _, processed_doc_path in doc_tuples:
		print(f"getting docs from {processed_doc_path}")
		for doc in get_processed_data(processed_doc_path):
			doc = filter_doc_for_ir(doc, dconfig)
			doc['category'] = np.asarray(sorted(doc['category']).values())  # makes a consistently ordered category vector
			id2doc[doc.id] = doc
	return id2doc


# --------------------------------------------------------------------------------------------------------------- #
# pre-processing functions to save a form of triples for a particular model spec
# --------------------------------------------------------------------------------------------------------------- #

def prep_fine_tuning_dataset(
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
	rel_type_dict, rel_dict, all_qrelled_docs = analyze_test_rels(rel_path)
	
	# get path to processed docs (already got topic path)
	doc_tuples, _ = ctpaths.get_data_tuples(dconfig.trec_or_kz)

	# get mappings of doc ids to doc dicts and topic ids to topic dicts
	id2doc, id2topic = get_doc_and_topic_mappings(all_qrelled_docs, doc_tuples, topic_path) 
	print(len(id2doc), len(all_qrelled_docs))
	
	missing_docs = set()
	skipped = 0

	# save combined triples of doc, topic, relevancy score
	with open(dconfig.save_path, 'w') as f:
		print(f"saving to: {dconfig.save_path}")

		for topic_id in rel_dict:
			for doc_id in rel_dict[topic_id]:
				label = rel_dict[topic_id][doc_id]	
				if downsample_zero(label, rel_type_dict['0'], dconfig):
					skipped += 1
					continue
					
				if doc_id in id2doc:
					combined = create_combined_doc(
						id2doc[doc_id], 
						id2topic[topic_id], 
						label, 
						dconfig=dconfig, 
					)

					# save to file as jsonl
					f.write(json.dumps(combined))
					f.write('\n')
				else:
					missing_docs.add(doc_id)


	print(f"number of docs missing: {len(missing_docs)}, number of zeros skipped: {skipped}")
	for md in missing_docs:
		print(md)



def create_combined_doc(
	doc, topic, 
	rel_score, 
	dconfig: DataConfig,
):
	combined = dict()

	# get filtered and truncated and SEP tokenized topic text
	combined['topic'] = prep_topic_text(topic, dconfig)

	# get filtered and truncated and SEP tokenized doc text
	combined['doc'] = prep_doc_text(doc, dconfig)

	# get relevancy score as string 
	if dconfig.convert_snli:
		rel_score = convert_label_snli(rel_score)

	combined['label'] = str(rel_score)

	return combined


def convert_label_snli(label: int) -> int:
	if label == 2:
		return 1
	elif label == 1:
		return 2
	return label



def downsample_zero(label: str, zero_ct: int, dconfig: DataConfig) -> bool:
	if dconfig.downsample_zeros_n is not None:
		if (label == 0) and (random.random()  >  (dconfig.downsample_zeros_n / zero_ct)):
			return True
	return False


def prep_topic_text(topic: Dict[str, Union[List[str], str, float]], dconfig: DataConfig) -> str:
	topic_text = ' '.join(topic['text_sents'])
	topic_text = truncate(topic_text, dconfig.max_topic_len)
	return topic_text


def get_n_crit(crit_list: List[str], dconfig: DataConfig) -> List[str]:
	if dconfig.first_n_only is not None:
		crit_list = crit_list[:min(len(crit_list), dconfig.first_n_only)]
	return crit_list


def prep_doc_text(doc: Dict[str, Union[List[str], str, float]], dconfig: DataConfig) -> str:
	

	# combine lists of strings into single string
	doc_inc = ' '.join(get_n_crit(doc['elig_crit']['include_criteria'], dconfig))
	doc_exc = ' '.join(get_n_crit(doc['elig_crit']['exclude_criteria'], dconfig))


	if 'condition' in dconfig.filtered_doc_keys:
		doc_inc = f"{' '.join(doc['condition'])} {doc_inc}"
		if dconfig.llm_prep:
			doc_inc = "Condition: " + doc_inc + ", "

	#truncate criteria separately if in config
	doc_inc = truncate(doc_inc, dconfig.max_inc_len)
	doc_exc = truncate(doc_exc, dconfig.max_exc_len)


	if dconfig.prepend_elig_gender:
		doc_inc = f"{doc['elig_gender']} {dconfig.sep} {doc_inc}"
		if dconfig.llm_prep:
			doc_inc = "Gender: " + doc_inc + ", "

	if dconfig.prepend_elig_age:
		if dconfig.llm_prep:
			doc_inc = f"Trial Doc: A person who is between {doc['elig_min_age']}-{doc['elig_max_age']} years old who meets the following Inclusion Criteria: {doc_inc}"
		else:
			doc_inc = f"eligible ages (years): {doc['elig_min_age']}-{doc['elig_max_age']}, {dconfig.sep} {doc_inc}"

	# combine criteria into single string
	if dconfig.include_only:
		if dconfig.llm_prep:
			doc_inc += LLM_END_PROMPT
		return doc_inc
	
	if dconfig.llm_prep:
		return f"{doc_inc} and does not meet these Exclusion Criteria: {doc_exc} {LLM_END_PROMPT}"

	return f"{doc_inc} {dconfig.sep} {doc_exc}"

	


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
		rel_path = ctpaths.TREC_REL_PATH
		topic_path = ctpaths.TREC_RELLED_TOPIC_PATH
	else:
		rel_path = ctpaths.KZ_REL_PATH
		topic_path = ctpaths.KZ_RELLED_TOPIC_PATH
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
		print(f"getting docs from {processed_doc_path}")
		for doc in get_processed_data(processed_doc_path):
			if doc['id'] in all_qrelled_docs:
				id2doc[doc['id']] = doc
	
	return id2doc, id2topic


if __name__ == '__main__':
	# proc_docs_and_topics('kz')
	# eda.explore_trec_data(part=2, rand_print=0.001) # select part 1-5 (~70k docs per part)
	# eda.explore_kz_data(rand_print=0.00001) # all in one file (~200k docs)


# class DataConfig(NamedTuple):
# 	save_path: str
# 	trec_or_kz: str = 'trec'
# 	filtered_topic_keys: Set[str] = {'id', 'text_sents', 'age', 'gender'}
# 	filtered_doc_keys: Set[str] = {'id', 'elig_min_age', 'elig_max_age', 'elig_gender', 'condition', 'elig_crit'}
# 	max_topic_len: Optional[int] = None
# 	max_inc_len: Optional[int] = None
# 	max_exc_len: Optional[int] = None
# 	prepend_elig_age: bool = True
# 	prepend_elig_gender: bool = True
# 	include_only: bool = False
# 	downsample_zeros_n: Optional[int] = None
# 	sep: str = '[SEP]'
# 	llm_prep: bool = False
# 	first_n_only: Optional[int] = None
# 	convert_snli: bool = False
# 	infer_category_model: Optional[str] = None

	dconfig = DataConfig(
		trec_or_kz='trec',
		save_path=ctpaths.TREC_ML_PATH, # make sure to change this!
		sep='',
		first_n_only=10,
		max_topic_len=200,
		llm_prep=False,
		prepend_elig_age=True,
		prepend_elig_gender=False
	)
	prep_fine_tuning_dataset(dconfig)
	#eda.explore_prepped(ctpaths.TREC_KZ_PATH)

