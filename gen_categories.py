

from ctmatch_prep import get_data_tuples
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


def add_condition_category_labels(model_checkpoint=CAT_GEN_MODEL, trec_or_kz: str = 'trec') -> None:
	pipe = pipeline(model_checkpoint)

	# open the processed documents and add the category labels
	for _, target in get_data_tuples(trec_or_kz=trec_or_kz):
		print(f"reading and writing to: {target}")
		with open(target, 'r') as f:
			data = json.load(f)

		for record in data:
			record['category'] = gen_category(pipe, line['doc']['condition'])
			print(record['doc'])

		

		# overwrite with new records having inferred category feature
		with open('test_category_data', 'w') as f:
			for line in data:
				f.write(json.dumps(line))
				f.write('\n')


def gen_category(pipe, text: str) -> str:
	output = pipe(text, candidate_labels=CT_CATEGORIES)
	score_dict = {output['labels'][i]:output['scores'][i] for i in range(len(output['labels']))}
	return sorted(score_dict.items, lambda x: x[1], reverse=True)[0][0]



if __name__ == '__main__':
	add_condition_category_labels('kz')