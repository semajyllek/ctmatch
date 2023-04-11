

from transformers import AutoTokenizer,  AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, ClassLabel, Dataset, Features, Value
from torch import nn
import pandas as pd
import numpy as np

from ctmatch_utils import compute_metrics
from typing import NamedTuple



TREC_DATA = '/Users/jameskelly/Documents/cp/ctmatch/data/trec_data/trec_data.jsonl'
KZ_DATA = '/Users/jameskelly/Documents/cp/ctmatch/data/kz_data/kz_data.jsonl'



class WeightedLossTrainer(Trainer):

  def __init__(self, label_weights, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.label_weights = label_weights

  def compute_loss(self, model, inputs, return_outputs=False):
    outputs = model(**inputs)
    logits = outputs.get("logits")
    labels = inputs.get("labels")
    loss_func = nn.CrossEntropyLoss(weight=self.label_weights)
    loss = loss_func(logits, labels)
    return (loss, outputs) if return_outputs else loss



class ModelConfig(NamedTuple):
  model_checkpoint: str
  max_length: int
  padding: str
  truncation: bool
  batch_size: int
  learning_rate: float
  num_train_epochs: int
  weight_decay: float
  warmup_steps: int

  

class CTMatch:
  def __init__(self, model_config: ModelConfig):
    self.model_config = model_config
    self.tokenizer = AutoTokenizer.from_pretrained(model_config.model_checkpoint)
    self.ct_dataset = self.load_data(model_config.dataset)
    self.ct_dataset_df = self.ct_dataset["train"].to_pandas()
    self.model = load_model(self.ct_dataset, model_config.model_checkpoint, model_config.max_length)
    return self.ct_dataset, self.model
  

   
  # ------------------ Data Loading ------------------ #
  def load_data(self, trec_or_kz: str) -> Dataset:
    data_path = f"data/{trec_or_kz}_data/{trec_or_kz}_data.json"
    self.ct_dataset = load_dataset('json', data_files=data_path)
    self.add_features()
    self.tokenize_dataset()
    self.ct_dataset.rename_column("label", "labels")
    self.ct_dataset.rename_column("topic_text", "sentence1")
    self.ct_dataset.rename_column("crit_text", "sentence2")
    return self.ct_dataset
 


  def add_features(self, ct_dataset) -> Dataset:
    features = Features({
      'crit_text': Value(dtype='string', id=None),
      'label': ClassLabel(names=["not_relevant", "partially_relevant", "relevant"]),
      'topic_text': Value(dtype='string', id=None)})
    ct_dataset["train"] = ct_dataset["train"].map(lambda x: x, batched=True, features=features)
    ct_dataset["test"] = ct_dataset["test"].map(lambda x: x, batched=True, features=features)
    ct_dataset["validation"] = ct_dataset["validation"].map(lambda x: x, batched=True, features=features)  
    return ct_dataset


  def tokenize_function(self):
    return self.tokenizer(
      self.ct_dataset["crit_text"], self.ct_dataset["topic_text"], 
      truncation=self.model_config.truncation, 
      padding=self.model_config.padding, 
      max_length=self.model_config.max_length
    )

  def tokenize_dataset(self):
    self.ct_dataset = self.ct_dataset.map(self.tokenize_function, batched=True)


# ------------------ Model Loading ------------------ #
def get_label_mapping(features):
  id2label = {idx:features["label"].int2str(idx) for idx in range(3)}
  label2id = {v:k for k, v in id2label.items()}
  return id2label, label2id

def get_label_weights(self):
  label_weights = (1 - (self.ct_dataset_df["label"].value_counts().sort_index() / len(self.ct_dataset_df))).values
  label_weights = torch.from_numpy(label_weights).float().to("cuda")


def get_trainer(self):
  return WeightedLossTrainer(
    model=self.model,
    args=self.get_training_args_obj(),
    compute_metrics=compute_metrics,
    train_dataset=self.ct_dataset["train"],
    eval_dataset=self.ct_dataset["validation"],
    tokenizer=self.tokenizer,
    label_weights=self.get_label_weights()
  )

def get_training_args_obj(self):
  return TrainingArguments(
    output_dir=self.model_config.output_dir,
    num_train_epochs=self.model_confi.num_train_epochs,
    learning_rate=self.model_config.learning_rate,
    per_device_train_batch_size=self.model_config.batch_size,
    per_device_eval_batch_size=self.model_config.batch_size,
    weight_decay=self.model_config.weight_decay,
    evaluation_strategy="epoch",
    logging_steps=self.model_config.logging_steps,
  )
  
def load_model(self):
  id2label, label2id = get_label_mapping(self.ct_dataset.features)
  self.model = AutoModelForSequenceClassification.from_pretrained(self.model_checkpoint, num_labels=3, id2label=id2label, label2id=label2id)
  self.trainer = self.get_trainer()
  return self.trainer



def get_embedding_similarity(self, topic, document):
    topic_input = self.tokenizer(topic, return_tensors='pt').to('cuda')
    doc_input = self.tokenizer(document, return_tensors='pt').to('cuda')
    topic_output = self.model(**topic_input, output_hidden_states=True)
    doc_output = self.model(**doc_input, output_hidden_states=True)
    topic_last_hidden = np.squeeze(topic_output.hidden_states[-1].detach().cpu().numpy(), axis=0)
    doc_last_hidden = np.squeeze(doc_output.hidden_states[-1].detach().cpu().numpy(), axis=0)
    topic_emb = np.mean(topic_last_hidden, axis=0)
    doc_emb = np.mean(doc_last_hidden, axis=0)
    return dot(topic_emb, doc_emb)/(norm(topic_emb) * norm(doc_emb))









def run_ctmatch_classifier(model_config: ModelConfig):
  ctmatch_dataset, ctmatch_model = CTMatch(model_config)
  ctmatch_model.train()
  predictions = ctmatch_model.predict(ctmatch_dataset["test"])
  print(predictions.metrics.items())





if __name__ == '__main__':
    # trec_data_path = '/Users/jameskelly/Documents/cp/ctmatch/data/trec_data/trec21_labelled_triples.jsonl'
    # kz_data_path = '/Users/jameskelly/Documents/cp/ctmatch/data/kz_data/kz_labelled_triples.jsonl'
    # new_trec_data_path = '/Users/jameskelly/Documents/cp/ctmatch/data/trec_data/trec_data.jsonl'
    # new_kz_data_path = '/Users/jameskelly/Documents/cp/ctmatch/data/kz_data/kz_data.jsonl'
    # create_dataset(trec_data_path, new_trec_data_path)
    # create_dataset(kz_data_path, new_kz_data_path)

    config = ModelConfig(
      dataset='kz',
      model_checkpoint='allenai/scibert_scivocab_uncased',
      max_length=512,
      batch_size=16,
      learning_rate=2e-5,
      num_train_epochs=3,
      weight_decay=0.01,
      warmup_steps=500,
    )

    run_ctmatch_classifier(config)

