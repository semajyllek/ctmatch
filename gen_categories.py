
from typing import List
from ctmatch_utils import get_processed_data
from ct_data_paths import get_data_tuples
from transformers import pipeline
import json

CAT_GEN_MODEL = "facebook/bart-large-mnli"

CT_CATEGORIES = [
    "pulmonary", "cardiac", "gastrointestinal", "renal", "psychological", "neurological", "cancer", "reproductive", "endocrine", "other"
]


# --------------------------------------------------------------------------------------------------------------- #
# this script is for applying zero-shot classification labels from 'facebook/bart-large-mnli' to the 
# documents of the dataset, including test, because we can assume this is something that is realistic to pre-compute
# since you have the documents apriori
# --------------------------------------------------------------------------------------------------------------- #
GET_ONLY = {'NCT00391586'}
GET_ONLY = None

def add_condition_category_labels(trec_or_kz: str = 'trec', model_checkpoint=CAT_GEN_MODEL, start: int = 0) -> None:
	pipe = pipeline(model=model_checkpoint)
	chunk_size = 1000

	# open the processed documents and add the category labels
	doc_tuples, _ = get_data_tuples(trec_or_kz=trec_or_kz)
	for _, target in doc_tuples:
		print(f"reading and writing to: {target}")
		data = get_processed_data(target, get_only=GET_ONLY)
		print(f"got {len(data)} records from {target}...")

		# overwrite with new records having inferred category feature
		with open('test_category_data', 'w') as f:
			i = start
			print(f'starting at: {i}')
			while i < len(data):
				next_chunk_end = min(len(data), i+chunk_size)
				conditions = [' '.join(doc['condition']).lower() for doc in data[i:next_chunk_end]]
				categories = gen_categories(pipe, conditions)
				print(f"generated {len(categories)} categories for {len(conditions)} conditions...")
				for j in range(i, next_chunk_end):
					data[j]['category'] = categories[j - i]
					f.write(json.dumps(data[j]))
					f.write('\n')

				
				print(f"{i=}, doc condition: {data[i]['condition']}, generated category: {data[i]['category'].items()}")
				i += chunk_size
			

		

	

def gen_categories(pipe, texts: List[str]) -> str:
	categories = []
	for output in pipe(texts, candidate_labels=CT_CATEGORIES):
		score_dict = {output['labels'][i]:output['scores'][i] for i in range(len(output['labels']))}
		#category = max(score_dict, key=score_dict.get)
		categories.append(score_dict)
	return categories




if __name__ == '__main__':
	add_condition_category_labels(model_checkpoint=CAT_GEN_MODEL, trec_or_kz='kz')