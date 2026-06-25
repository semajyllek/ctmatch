
from typing import Generator, List, Optional, Tuple
from ctmatch.utils.ctmatch_utils import get_processed_data
from ctmatch.ct_data_paths import get_data_tuples
from transformers import pipeline
import numpy as np
import json

CAT_GEN_MODEL = "facebook/bart-large-mnli"
#CAT_GEN_MODEL = "microsoft/biogpt"

CT_CATEGORIES = [
    "pulmonary", "cardiac", "gastrointestinal", "renal", "psychological", "genetic", "pediatric",
	"neurological", "cancer", "reproductive", "endocrine", "infection", "healthy", "other"
]


# --------------------------------------------------------------------------------------------------------------- #
# this script is for applying zero-shot classification labels from 'facebook/bart-large-mnli' to the 
# documents of the dataset, including test, because we can assume this is something that is realistic to pre-compute
# since you have the documents apriori
# --------------------------------------------------------------------------------------------------------------- #
GET_ONLY = None


def stream_condition_data(data_chunk, doc_or_topic: str = 'doc') -> Generator[str, None, None]:
    for d in data_chunk:
      if doc_or_topic == 'topic':
        yield d['raw_text']
      else:
        condition = d['condition']
        if len(condition) == 0:
          yield 'no information'
        else:     
          yield ' '.join(condition).lower()
     

def add_condition_category_labels(
	trec_or_kz: str = 'trec', 
	model_checkpoint=CAT_GEN_MODEL, 
	start: int = 0,
	doc_tuples: Optional[List[Tuple[str, str]]] = None,
    category_label='category',
    doc_or_topic: str = 'doc'
) -> None:
    pipe = pipeline(model=model_checkpoint, device=0)
    chunk_size = 1000
    
    # open the processed documents and add the category labels
    if doc_tuples is None:
        doc_tuples, _ = get_data_tuples(trec_or_kz=trec_or_kz)
        
    for _, target in doc_tuples:
        print(f"reading and writing to: {target}")
        data = [d for d in get_processed_data(target, get_only=GET_ONLY)]
        print(f"got {len(data)} records from {target}...")
        
        # overwrite with new records having inferred category feature
        with open('/content/drive/MyDrive/ct_data23/processed_trec_topic_X.jsonl', 'w') as f:
            i = start
            print(f'starting at: {i}')
            while i < len(data):
                next_chunk_end = min(len(data), i+chunk_size)
                conditions = stream_condition_data(data[i:next_chunk_end], doc_or_topic=doc_or_topic)
                categories = gen_categories(pipe, conditions)
                print(f"generated {len(categories)} categories for {chunk_size} conditions...")
                for j in range(i, next_chunk_end):
                    data[j][category_label] = categories[j - i]
                    f.write(json.dumps(data[j]))
                    f.write('\n')
                    
                if doc_or_topic == 'doc':
                    print(f"{i=}, doc condition: {data[i]['condition']}, generated category: {data[i]['category'].items()}")
                else:
                    print(f"{i=}, topic raw text condition: {data[i]['raw_text']}, generated category: {data[i]['category'].items()}")
                    
                i += chunk_size
		

def gen_categories(pipe, text_dataset: Generator[str, None, None]) -> str:
	categories = []
	for output in pipe(text_dataset, candidate_labels=CT_CATEGORIES, batch_size=64):
		score_dict = {output['labels'][i]:output['scores'][i] for i in range(len(output['labels']))}
		#category = max(score_dict, key=score_dict.get)
		categories.append(score_dict)
	return categories


def gen_single_category_vector(pipe, text: str) -> str:
    output = pipe(text, candidate_labels=CT_CATEGORIES)
    score_dict = {output['labels'][i]:output['scores'][i] for i in range(len(output['labels']))}
    return np.array(sorted(score_dict, key=score_dict.get, reverse=True))