
from typing import Dict, List

def calc_mrr(ranking: List[str], doc2rel: Dict[str, int], pos_val: int = 2) -> float:
	"""
	desc:       compute the mean reciprocal rank of a ranking
	returns:    mrr 
	"""
	for i, doc_id in enumerate(ranking):
		if doc2rel[doc_id] == pos_val:
			return 1/(i+1)
	return 0.0
