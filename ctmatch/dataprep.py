
from typing import Any, Tuple

# external imports
from datasets import Dataset, load_dataset, ClassLabel, Features, Value
from transformers import AutoTokenizer
from numpy.linalg import norm
import numpy as np

# package tools
from .utils.ctmatch_utils import train_test_val_split, get_processed_data
from .modelconfig import ModelConfig

SUPPORTED_LMS = [
    'roberta-large', 'cross-encoder/nli-roberta-base',
    'microsoft/biogpt', 'allenai/scibert_scivocab_uncased', 
    'facebook/bart-large', 'gpt2', 'semaj83/scibert_finetuned_ctmatch'
]


class DataPrep:
    # multiple 'datasets' need to be prepared for the pipeline
    # 1. the dataset for the classifier model triplets and a dataframe, ~ 25k rows
    # 2. the dataset for the category model, every doc ~200k rows
    # 3. the dataset for the embedding model, every doc < 200k rows



    def __init__(self, model_config: ModelConfig) -> None:
        self.model_config = model_config
        self.tokenizer = self.get_tokenizer()
        self.ct_dataset = None
        self.ct_train_dataset_df = None
        self.load_data()
        
	
    def get_tokenizer(self):
        model_checkpoint = self.model_config.model_checkpoint
        if model_checkpoint not in SUPPORTED_LMS:
            raise ValueError(f"Model checkpoint {model_checkpoint} not supported. Please use one of {SUPPORTED_LMS}")
        tokenizer = AutoTokenizer.from_pretrained(self.model_config.model_checkpoint)
        if self.model_config.model_checkpoint == 'gpt2':
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer


    # ------------------ Data Loading ------------------ #
    def load_data(self) -> Dataset:
        self.ct_dataset = load_dataset('semaj83/ctmatch', data_files=self.model_config.classified_data_path)
        self.ct_dataset = train_test_val_split(self.ct_dataset, self.model_config.splits, self.model_config.seed)
        self.add_features()
        self.tokenize_dataset()
        self.ct_dataset = self.ct_dataset.rename_column("label", "labels")
        # self.ct_dataset = self.ct_dataset.rename_column("topic", "sentence1")
        # self.ct_dataset = self.ct_dataset.rename_column("doc", "sentence2")
        self.ct_dataset.set_format(type='torch', columns=['doc', 'labels', 'topic', 'input_ids', 'attention_mask'])
        if not self.model_config.use_trainer:
            self.ct_dataset = self.ct_dataset.remove_columns(['doc', 'topic'])  # removing labels for next-token prediction...

        self.ct_train_dataset_df = self.ct_dataset['train'].remove_columns(['input_ids', 'attention_mask', 'token_type_ids']).to_pandas()

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
            examples["topic"], examples["doc"], 
            truncation=self.model_config.truncation, 
            padding=self.model_config.padding, 
            max_length=self.model_config.max_length
        )

    def tokenize_dataset(self):
        self.ct_dataset = self.ct_dataset.map(self.tokenize_function, batched=True)


    def get_category_data(self, vectorize=True):
        category_data = dict()
        sorted_cat_keys = None
        for cdata in get_processed_data(self.model_config.category_path):

            # cdata = {<nct_id>: {cat1: float1, cat2: float2...}}
            cdata_id, cdata_dict = list(cdata.items())[0]
            if sorted_cat_keys is None:
                sorted_cat_keys = sorted(cdata_dict.keys())

            if vectorize:
                cat_vec = np.asarray([cdata_dict[k] for k in sorted_cat_keys])
            else:
                cat_vec = cdata_dict

            category_data[cdata_id] = cat_vec
        return category_data
    

     # ------------------ Embedding Similarity ------------------ #
    def create_doc_embeddings(self, split='train'):
        doc_embeddings = []
        for example in self.ct_dataset[split]:
            doc_encoding = self.tokenize_function(example)
            doc_embeddings.append(self.lm_model(**doc_encoding.last_hidden_state.numpy()))

        self.doc_embeddings = norm(np.array(doc_embeddings))



    def get_embedding_similarity(self, lm_model, topic, document):
        topic_input = self.tokenizer(topic, return_tensors='pt').to('cuda')
        doc_input = self.tokenizer(document, return_tensors='pt').to('cuda')
        topic_output = lm_model(**topic_input, output_hidden_states=True)
        doc_output =lm_model(**doc_input, output_hidden_states=True)
        topic_last_hidden = np.squeeze(topic_output.hidden_states[-1].detach().cpu().numpy(), axis=0)
        doc_last_hidden = np.squeeze(doc_output.hidden_states[-1].detach().cpu().numpy(), axis=0)
        topic_emb = np.mean(topic_last_hidden, axis=0)
        doc_emb = np.mean(doc_last_hidden, axis=0)
        return np.dot(topic_emb, doc_emb)/(norm(topic_emb) * norm(doc_emb))

  

