from typing import Generator, List, Optional, Tuple
from ctmatch_utils import get_processed_data
from ct_data_paths import get_data_tuples
from transformers import pipeline
import json

CAT_GEN_MODEL = "facebook/bart-large-mnli"
CAT_GEN_MODEL = "microsoft/biogpt"

CT_CATEGORIES = [
    "pulmonary", "cardiac", "gastrointestinal", "renal", "psychological", "genetic", "pediatric",
	"neurological", "cancer", "reproductive", "endocrine", "infection", "other"
]


# --------------------------------------------------------------------------------------------------------------- #
# this script is for applying zero-shot classification labels from 'facebook/bart-large-mnli' to the 
# documents of the dataset, including test, because we can assume this is something that is realistic to pre-compute
# since you have the documents apriori
# --------------------------------------------------------------------------------------------------------------- #
GET_ONLY = {'NCT00391586'}
GET_ONLY = None


def stream_condition_data(data_chunk) -> Generator[str, None, None]:
    for doc in data_chunk:
	    yield ' '.join(doc['condition']).lower()

def add_condition_category_labels(
	trec_or_kz: str = 'trec', 
	model_checkpoint=CAT_GEN_MODEL, 
	start: int = 0,
	doc_tuples: Optional[List[Tuple[str, str]]] = None,
  category_label='category'
) -> None:
	pipe = pipeline(model=model_checkpoint, device=0)
	chunk_size = 1000
	new_categories = []

	# open the processed documents and add the category labels
	if doc_tuples is None:
		doc_tuples, _ = get_data_tuples(trec_or_kz=trec_or_kz)

	for _, target in doc_tuples:
		print(f"reading and writing to: {target}")
		data = [d for d in get_processed_data(target, get_only=GET_ONLY)]
		print(f"got {len(data)} records from {target}...")
		

		# overwrite with new records having inferred category feature
		with open('test_category_data', 'w') as f:
			i = start
			print(f'starting at: {i}')
			while i < len(data):
				next_chunk_end = min(len(data), i+chunk_size)
				conditions = stream_condition_data(data[i:next_chunk_end])
				categories = gen_categories(pipe, conditions)
				print(f"generated {len(categories)} categories for {chunk_size} conditions...")
				for j in range(i, next_chunk_end):
					data[j][category_label] = categories[j - i]
					f.write(json.dumps(data[j]))
					f.write('\n')

				print(f"{i=}, doc condition: {data[i]['condition']}, generated category: {data[i]['category'].items()}")
				i += chunk_size
		

def gen_categories(pipe, text_dataset: Generator[str, None, None]) -> str:
	categories = []
	for output in pipe(text_dataset, candidate_labels=CT_CATEGORIES, batch_size=64):
		score_dict = {output['labels'][i]:output['scores'][i] for i in range(len(output['labels']))}
		#category = max(score_dict, key=score_dict.get)
		categories.append(score_dict)
	return categories



if __name__ == '__main__':
	add_condition_category_labels(model_checkpoint=CAT_GEN_MODEL, trec_or_kz='kz', start=3000)