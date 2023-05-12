
from typing import Any, Dict, List, Optional, Set
from sklearn.metrics.pairwise import linear_kernel
from collections import defaultdict
from numpy.linalg import norm
from datasets import Dataset
import numpy as np
import json
import re





#----------------------------------------------------------------#
# global regex patterns for use throughout the methods
#----------------------------------------------------------------#


EMPTY_PATTERN = re.compile('[\n\s]+')
"""
both_inc_and_exc_pattern = re.compile(r\"\"\"[\s\n]*[Ii]nclusion [Cc]riteria:?               # top line of both
                                      (?:[ ]+[Ee]ligibility[ \w]+\:[ ])?                  # could contain this unneeded bit next
                                      (?P<include_crit>[ \n\-\.\?\"\%\r\w\:\,\(\)]*)      # this should get all inclusion criteria as a string
                                      [Ee]xclusion[ ][Cc]riteria:?                        # delineator to exclusion criteria
                                      (?P<exclude_crit>[\w\W ]*)                          # exclusion criteria as string
                                      \"\"\", re.VERBOSE)
"""
INC_ONLY_PATTERN = re.compile('[\s\n]+[Ii]nclusion [Cc]riteria:?([\w\W ]*)')
EXC_ONLY_PATTERN = re.compile('[\n\r ]+[Ee]xclusion [Cc]riteria:?([\w\W ]*)')
AGE_PATTERN = re.compile('(?P<age>\d+) *(?P<units>\w+).*')
YEAR_PATTERN = re.compile('(?P<year>[yY]ears?.*)')
MONTH_PATTERN = re.compile('(?P<month>[mM]o(?:nth)?)')
WEEK_PATTERN = re.compile('(?P<week>[wW]eeks?)')

BOTH_INC_AND_EXC_PATTERN = re.compile("[\s\n]*[Ii]nclusion [Cc]riteria:?(?: +[Ee]ligibility[ \w]+\: )?(?P<include_crit>[ \n\-\.\?\"\%\r\w\:\,\(\)]*)[Ee]xclusion [Cc]riteria:?(?P<exclude_crit>[\w\W ]*)")



# -------------------------------------------------------------------------------------- #
# pretokenization utils (should be in a tokenizer...)
# -------------------------------------------------------------------------------------- #

def truncate(s: str, max_tokens: Optional[int] = None) -> str:
  if max_tokens is None:
    return s
  s_tokens = s.split()
  return ' '.join(s_tokens[:min(len(s_tokens), max_tokens)])



# -------------------------------------------------------------------------------------- #
# I/O utils
# -------------------------------------------------------------------------------------- #

def save_docs_jsonl(docs: List[Any], writefile: str) -> None:
  """
  desc:    iteratively writes contents of docs as jsonl to writefile 
  """
  with open(writefile, "w") as outfile:
    for doc in docs:
      json.dump(doc, outfile)
      outfile.write("\n")


def get_processed_data(proc_loc: str, get_only: Optional[Set[str]] = None):
  """
  proc_loc:    str or path to location of docs in jsonl form
  """
  with open(proc_loc, 'r') as json_file:
    json_list = list(json_file)

  if get_only is None:
    for json_str in json_list:
      yield json.loads(json_str)

  else:
    for s in json_list:
      s_data = json.loads(s)
      if s_data["id"] in get_only:
        yield s_data
        get_only.remove(s_data['id'])
        if len(get_only) == 0:
          return

  
  


def train_test_val_split(dataset, splits: Dict[str, float], seed: int = 37) -> Dataset:
  """
  splits a dataset having only "train" into one having train, test, val, with 
  split sizes determined by splits["train"] and splits["val"] (dict must have those keys)

  """
  dataset = dataset["train"].train_test_split(train_size=splits["train"], seed=seed)
  train = dataset["train"]
  sub = train.train_test_split(test_size=splits["val"],  seed=seed)
  new_train = sub["train"]
  new_val = sub["test"]
  dataset["train"] = new_train
  dataset["validation"] = new_val
  return dataset



#----------------------------------------------------------------#
# computation methods
#----------------------------------------------------------------#

def exclusive_argmax(vector: np.ndarray) -> np.ndarray:
    mask = np.zeros(len(vector))
    argmax = np.argmax(vector)
    vector = vector * mask
    vector[argmax]  = 1
    return vector    


#----------------------------------------------------------------#
# evaluation methods (duplicated from ctproc scripts)
#----------------------------------------------------------------#

def get_test_rels(rel_path):
    rel_dict = defaultdict(lambda:defaultdict(int))
    rel_type_dict = defaultdict(int)
    for line in open(rel_path, 'r').readlines():
        topic_id, _, doc_id, rel = re.split(r'\s+', line.strip())
        rel_dict[topic_id][doc_id] = int(rel)
        rel_type_dict[rel] += 1
    return rel_dict, rel_type_dict



