

from transformers import AutoTokenizer,  AutoModelForSequenceClassification, Trainer, TrainingArguments, get_scheduler
from datasets import load_dataset, ClassLabel, Dataset, Features, Value
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import Dataset, TensorDataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from torch.optim import AdamW
from numpy.linalg import norm
from tqdm.auto import tqdm
from pathlib import Path
from numpy import dot
from torch import nn
import numpy as np
import evaluate
import spacy
import torch

from ctmatch.ctmatch_utils import compute_metrics, train_test_val_split
from typing import Dict, NamedTuple, Optional, Tuple



TREC_DATA = Path('/Users/jameskelly/Documents/cp/ctmatch/data/trec_data/trec_data.jsonl')
KZ_DATA = Path('/Users/jameskelly/Documents/cp/ctmatch/data/kz_data/kz_data.jsonl')



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
    data_path: Path
    model_checkpoint: str
    max_length: int
    padding: str
    truncation: bool
    batch_size: int
    learning_rate: float
    train_epochs: int
    weight_decay: float
    warmup_steps: int
    seed: int
    splits: Dict[str, float]
    output_dir: Optional[str] = None
    convert_snli: bool = False
    use_trainer: bool = True


  

class CTMatch:
    def __init__(self, model_config: ModelConfig) -> None:
        self.model_config = model_config
        self.tokenizer = AutoTokenizer.from_pretrained(model_config.model_checkpoint)
        self.ct_dataset = self.load_data()
        self.ct_dataset_df = self.ct_dataset["train"].to_pandas()
        self.optimizer = None
        self.lr_scheduler = None
        self.trainer = None
        self.num_training_steps = self.model_config.train_epochs * len(self.ct_dataset['train'])
        self.model = self.load_model()

        # embedding attrs
        self._spacy_model = None
        self._tfidf_model = None
        
        if not self.model_config.use_trainer:
            self.train_dataloader, self.val_dataloader = self.get_dataloaders()
            



    def train_and_predict(self):
        if self.trainer is not None:
            self.trainer.train()
            predictions = self.trainer.predict(self.ct_dataset["test"])
            print(predictions.metrics.items())
        else:
            self.torch_train()
            self.torch_eval()
            





     # ------------------ native torch training loop ------------------ #
    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        train_dataloader = DataLoader(self.ct_dataset['train'], shuffle=True, batch_size=self.model_config.batch_size)
        val_dataloader = DataLoader(self.ct_dataset['validation'], batch_size=self.model_config.batch_size)
        return train_dataloader, val_dataloader

    def torch_train(self):
        progress_bar = tqdm(range(self.num_training_steps))
        self.model.train()
        for epoch in range(self.model_config.train_epochs):
            for batch in self.train_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()

                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                progress_bar.update(1)
            

            
                


                
        
               


    def torch_eval(self):
        metric = evaluate.load("f1")
        self.model.eval()
        for batch in self.val_dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # don't learn during evaluation
            with torch.no_grad():
                outputs = self.model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])

        metric.compute()



  

    
    # ------------------ Data Loading ------------------ #
    def load_data(self) -> Dataset:
        self.ct_dataset = load_dataset('json', data_files=self.model_config.data_path.as_posix())
        self.ct_dataset = train_test_val_split(self.ct_dataset, self.model_config.splits, self.model_config.seed)
        self.add_features()
        self.tokenize_dataset()
        # self.ct_dataset = self.ct_dataset.rename_column("label", "labels")
        # self.ct_dataset = self.ct_dataset.rename_column("topic", "sentence1")
        # self.ct_dataset = self.ct_dataset.rename_column("doc", "sentence2")
        self.ct_dataset.set_format(type='torch', columns=['doc', 'label', 'topic', 'input_ids', 'attention_mask'])
        return self.ct_dataset

    
    def add_features(self) -> None:
        if self.model_config.convert_snli:
            names = ['contradiction', 'entailment', 'neutral']
        else:
            names = ["not_relevant", "partially_relevant", "relevant"]

        features = Features({
            'doc': Value(dtype='string', id=None),
            'label': ClassLabel(names=names),
            'topic': Value(dtype='string', id=None)
        })
        self.ct_dataset["train"] = self.ct_dataset["train"].map(lambda x: x, batched=True, features=features)
        self.ct_dataset["test"] = self.ct_dataset["test"].map(lambda x: x, batched=True, features=features)
        self.ct_dataset["validation"] = self.ct_dataset["validation"].map(lambda x: x, batched=True, features=features)  


    def tokenize_function(self, examples):
        return self.tokenizer(
            examples["doc"], examples["topic"], 
            truncation=self.model_config.truncation, 
            padding=self.model_config.padding, 
            max_length=self.model_config.max_length
        )

    def tokenize_dataset(self):
        self.ct_dataset = self.ct_dataset.map(self.tokenize_function, batched=True)





    # ------------------ Model Loading ------------------ #
    def load_model(self):
        id2label, label2id = self.get_label_mapping()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_config.model_checkpoint,
            num_labels=3,                                    # makes the last head be replaced with a linear layer with 3 outputs
            id2label=id2label, label2id=label2id
        )
        
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(f"Using device: {self.device}")
       
        self.optimizer = AdamW(self.model.parameters(), lr=self.model_config.learning_rate, weight_decay=self.model_config.weight_decay)
        self.num_training_steps = self.model_config.train_epochs * len(self.ct_dataset['train'])
        self.lr_scheduler = get_scheduler(
            name="linear", 
            optimizer=self.optimizer, 
            num_warmup_steps=self.model_config.warmup_steps, 
            num_training_steps=self.num_training_steps
        )

        if self.model_config.use_trainer:
            self.trainer = self.get_trainer()
        else:
            self.model = self.model.to(self.device)

        return self.model


    def get_label_mapping(self):
        id2label = {idx:self.ct_dataset['train'].features["label"].int2str(idx) for idx in range(3)}
        label2id = {v:k for k, v in id2label.items()}
        return id2label, label2id

    def get_label_weights(self):
        label_weights = (1 - (self.ct_dataset_df["label"].value_counts().sort_index() / len(self.ct_dataset_df))).values
        label_weights = torch.from_numpy(label_weights).float().to("cuda")


    def get_trainer(self):
        return WeightedLossTrainer(
            model=self.model,
            optimizers=(self.optimizer, self.lr_scheduler),
            args=self.get_training_args_obj(),
            compute_metrics=compute_metrics,
            train_dataset=self.ct_dataset["train"],
            eval_dataset=self.ct_dataset["validation"],
            tokenizer=self.tokenizer,
            label_weights=self.get_label_weights()
        )

    def get_training_args_obj(self):
        output_dir = self.model_config.output_dir if self.model_config.output_dir is not None else self.model_config.data_path.parent.parent.as_posix()
        
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.model_config.train_epochs,
            learning_rate=self.model_config.learning_rate,
            per_device_train_batch_size=self.model_config.batch_size,
            per_device_eval_batch_size=self.model_config.batch_size,
            weight_decay=self.model_config.weight_decay,
            evaluation_strategy="epoch",
            logging_steps=len(self.ct_dataset["train"]) // self.model_config.batch_size,
        )
        


    def multi_acc(self, y_pred, y_test):
        acc = (torch.log_softmax(y_pred, dim=1).argmax(dim=1) == y_test).sum().float() / float(y_test.size(0))
        return acc
    

    def get_sklearn_metrics(self):
        if self.model_config.use_trainer:
            y_preds = list(self.trainer.predict(self.ct_dataset["validation"]).predictions.argmax(axis=1))
        else:
            raise NotImplementedError("Only trainer is supported for sklearn metrics")
        y_trues = list(self.ct_dataset["validation"]["label"])
        return confusion_matrix(y_trues, y_preds), classification_report(y_trues, y_preds)


    # ------------------ Embedding Similarity ------------------ #
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


    @property
    def spacy_model(self):
        return self._spacy_model

    @spacy_model.getter
    def spacy_model(self):   
        if self._spacy_model is None:
            self._spacy_model = spacy.load("en_core_web_md")
        return self._spacy_model
    
    @property
    def tfidf_model(self):
        return self._tfidf_model

    @tfidf_model.getter
    def tfidf_model(self):
        if self._tfidf_model is None:
            self._tfidf_model = TfidfVectorizer()
        return self.tfidf_model

    def get_spacy_embedding_similarity(self, topic, document):
        topic_doc = self.spacy_model(topic)
        doc_doc = self.spacy_model(document)
        return topic_doc.similarity(doc_doc)
    
    def get_tfidf_similarity(self, topic, document):
        tfidf_matrix = self.tfidf_model.fit_transform([topic, document])
        return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)[0][1]











if __name__ == '__main__':
    # trec_data_path = '/Users/jameskelly/Documents/cp/ctmatch/data/trec_data/trec21_labelled_triples.jsonl'
    # kz_data_path = '/Users/jameskelly/Documents/cp/ctmatch/data/kz_data/kz_labelled_triples.jsonl'
    # new_trec_data_path = '/Users/jameskelly/Documents/cp/ctmatch/data/trec_data/trec_data.jsonl'
    # new_kz_data_path = '/Users/jameskelly/Documents/cp/ctmatch/data/kz_data/kz_data.jsonl'
    # create_dataset(trec_data_path, new_trec_data_path)
    # create_dataset(kz_data_path, new_kz_data_path)

    scibert_model = 'allenai/scibert_scivocab_uncased'

    config = ModelConfig(
        data_path=KZ_DATA,
        model_checkpoint=scibert_model,
        max_length=512,
        batch_size=16,
        learning_rate=2e-5,
        train_epochs=3,
        weight_decay=0.01,
        warmup_steps=500,
        splits={"train":0.8, "val":0.1}
    )

    ctm = CTMatch(config)
    ctm.train_and_predict()



