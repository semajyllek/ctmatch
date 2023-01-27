#from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
from typing import Dict, List, Tuple
from sklearn.metrics import f1_score
from torch import nn
import numpy as np
import sklearn
import random
import torch
import json




#------------------------------------------#
# huggingface transformers methods, objects
#------------------------------------------#
"""

class WeightedLossTrainer(Trainer):
  def compute_loss(self, model, inputs, return_outputs=False):
    outputs = model(**inputs)
    logits = outputs.get("logits")
    labels = inputs.get("labels")
    loss_func = nn.CrossEntropyLoss(weights=label_weights)
    loss = loss_func(logits, labels)
    return (loss, outputs) if return_outputs else loss




def compute_metrics(pred):
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  f1 = f1_score(labels, preds, average="weighted")
  return {"f1":f1}

"""




















#-----------------------------------------------------#
# methods to get data and build Dataset object from ct data
#-----------------------------------------------------#


def load_results(jsonl_readfile):
  results = []
  with open(jsonl_readfile, 'r') as json_file:
    json_list = list(json_file)
    for json_str in json_list:
      result = json.loads(json_str)
      results.append(result)
  return results





def train_test_val_split(dataset: Dataset, splits=Dict[str, float]) -> Dataset:
  """
  splits a dataset having only "train" into one having train, test, val, with 
  split sizes determined by splits["train"] and splits["val"] (dict must have those keys)

  """
  dataset = dataset["train"].train_test_split(train_size=splits["train"])
  train = dataset["train"]
  sub = train.train_test_split(test_size=splits["val"])
  new_train = sub["train"]
  new_val = sub["test"]
  dataset["train"] = new_train
  dataset["validation"] = new_val
  return dataset




def create_dataset(id2doc, id2topic, rel_dict, writefile) -> Tuple[List[Tuple[str, str]], List[int]]:
  
  with open(writefile, 'w') as wf:
    for topic_id in rel_dict:
      topic_text = id2topic[topic_id]["sents"]
      for doc_id in rel_dict[topic_id]:
        crit_text = id2doc[doc_id]["include_criteria"] + " [SEP] " + id2doc[doc_id]["exclude_criteria"]
        new_ex = {"topic_text":topic_text, "crit_text":crit_text, "label":rel_dict[topic_id][doc_id]}
        json.dump(new_ex, wf)
        wf.write("\n")




def get_crit_trunc_sent(
    result: Dict[str, str], 
    max_tokens: int, 
    label: str
    ) -> str:
    
    token_count = 0
    trunc_sent = ""
    if label in ["include_criteria", "exclude_criteria"]:
      filtered_result = result['eligibility/criteria/textblock'][label] 
    else:
      filtered_result = result[label]

    for crit_sent in filtered_result:
      crit_sent_split = crit_sent.split()
      avail_ct = max_tokens - (token_count + len(crit_sent_split))
      if avail_ct < 0:
        break 
      
      using = min(len(crit_sent_split), avail_ct)
      trunc_sent +=  ' ' + ' '.join(crit_sent_split[:using])
      token_count += using

    return trunc_sent.strip()



def truncate_sents(
    jsonl_readfile: str, 
    jsonl_writefile: str,
    labels: List[str],
    max_lens: List[int]):
  
  assert len(max_lens) == len(labels), "labels and lengths arrays must be the same length"
  with open(jsonl_readfile, 'r') as json_file:
    json_list = list(json_file)

  

  with open(jsonl_writefile, 'w') as wf:
    for json_str in json_list:
      result = json.loads(json_str)
    
      new_result = {'id': result['id']}
      for label, max_len in zip(labels, max_lens):
        new_result[label] = get_crit_trunc_sent(result, max_len, label)
    
      json.dump(new_result, wf)
      wf.write('\n')




def get_vocab(data_path, doc_or_topic="doc"):

  counter = Counter()
  lengths = set()

  with open(data_path, 'r') as json_file:
    json_list = list(json_file)


  for json_str in json_list:
    result = json.loads(json_str)

    new_inc_entry = []
    if doc_or_topic=="doc":
      include_result = result['eligibility/criteria/textblock']['include_criteria']
    else:
      include_result = result['sents']

    for sent in include_result:
      data_sent = [w.lower() for w in sent.split()]
      new_inc_entry += data_sent
      counter.update(data_sent)
    lengths.add(len(new_inc_entry))

  return counter, lengths

"""
def tokenize_text(examples):
  return scibert_tokenizer(examples["topic_text"], examples["crit_text"], truncation=True, padding=True, max_length=512)
"""