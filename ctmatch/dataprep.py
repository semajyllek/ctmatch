


# external imports
from datasets import Dataset, load_dataset, ClassLabel, Features, Value
from transformers import AutoTokenizer
import pandas as pd
import numpy as np

# package tools
from .utils.ctmatch_utils import train_test_val_split, get_processed_data, get_test_rels
from .pipeconfig import PipeConfig


# path to ctmatch dataset on HF hub
CTMATCH_CLASSIFICATION_DATASET_ROOT = "semaj83/ctmatch_classification"
CTMATCH_IR_DATASET_ROOT = "semaj83/ctmatch_ir"
CLASSIFIER_DATA_PATH = "combined_classifier_data.jsonl"
DOC_TEXTS_PATH = "doc_texts.txt"
DOC_CATEGORIES_VEC_PATH = "doc_categories.txt"
DOC_EMBEDDINGS_VEC_PATH = "doc_embeddings.txt"
INDEX2DOCID_PATH = "index2docid.txt"


SUPPORTED_LMS = [
    'roberta-large', 'cross-encoder/nli-roberta-base',
    'microsoft/biogpt', 'allenai/scibert_scivocab_uncased', 
    'facebook/bart-large', 'gpt2', 
    'semaj83/scibert_finetuned_ctmatch', 'semaj83/scibert_finetuned_pruned_ctmatch'

]


class DataPrep:
    # multiple 'datasets' need to be prepared for the pipeline
    # 1. the dataset for the classifier model triplets and a dataframe, ~ 25k rows
    # 2. the dataset for the category model, every doc ~200k rows
    # 3. the dataset for the embedding model, every doc < 200k rows



    def __init__(self, pipe_config: PipeConfig) -> None:
        self.pipe_config = pipe_config
        self.classifier_tokenizer = self.get_classifier_tokenizer()
        self.ct_dataset = None
        self.ct_train_dataset_df = None
        self.index2docid = None
        self.doc_embeddings_df = None
        self.doc_categories_df = None

        if pipe_config.ir_setup:
            self.load_ir_data()
        else:
            self.load_classifier_data()


        
	
    def get_classifier_tokenizer(self):
        model_checkpoint = self.pipe_config.classifier_model_checkpoint
        if model_checkpoint not in SUPPORTED_LMS:
            raise ValueError(f"Model checkpoint {model_checkpoint} not supported. Please use one of {SUPPORTED_LMS}")
        if 'scibert' in model_checkpoint:
            tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', use_fast=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.pipe_config.classifier_model_checkpoint)
        if self.pipe_config.classifier_model_checkpoint == 'gpt2':
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer


    # ------------------ Classifier Data Loading ------------------ #
    def load_classifier_data(self) -> Dataset:
        self.ct_dataset = load_dataset(CTMATCH_CLASSIFICATION_DATASET_ROOT, data_files=CLASSIFIER_DATA_PATH)
        self.ct_dataset = train_test_val_split(self.ct_dataset, self.pipe_config.splits, self.pipe_config.seed)
        self.add_features()
        self.tokenize_dataset()
        self.ct_dataset = self.ct_dataset.rename_column("label", "labels")
        # self.ct_dataset = self.ct_dataset.rename_column("topic", "sentence1")
        # self.ct_dataset = self.ct_dataset.rename_column("doc", "sentence2")
        self.ct_dataset.set_format(type='torch', columns=['doc', 'labels', 'topic', 'input_ids', 'attention_mask'])
        if not self.pipe_config.use_trainer:
            self.ct_dataset = self.ct_dataset.remove_columns(['doc', 'topic'])  # removing labels for next-token prediction...

        self.ct_train_dataset_df = self.ct_dataset['train'].remove_columns(['input_ids', 'attention_mask', 'token_type_ids']).to_pandas()

        return self.ct_dataset

    
    def add_features(self) -> None:
        if self.pipe_config.convert_snli:
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
        return self.classifier_tokenizer(
            examples["topic"], examples["doc"], 
            truncation=self.pipe_config.truncation, 
            padding=self.pipe_config.padding, 
            max_length=self.pipe_config.max_length
        )

    def tokenize_dataset(self):
        self.ct_dataset = self.ct_dataset.map(self.tokenize_function, batched=True)


    def get_category_data(self, vectorize=True):
        category_data = dict()
        sorted_cat_keys = None
        for cdata in get_processed_data(self.pipe_config.category_path):

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
    

    
    # ------------------ IR Data Loading ------------------ #
    def process_ir_data_from_hf(self, ds_path, is_text: bool = False):
        ds = load_dataset(CTMATCH_IR_DATASET_ROOT, data_files=ds_path)
        if is_text:
            return pd.DataFrame(ds['train'])
    
        arrays = [np.asarray(a['text'].split(','), dtype=float) for a in ds['train']]
        return pd.DataFrame(arrays)

    def load_ir_data(self) -> None:
        self.index2docid = self.process_ir_data_from_hf(INDEX2DOCID_PATH, is_text=True)
        self.doc_embeddings_df = self.process_ir_data_from_hf(DOC_EMBEDDINGS_VEC_PATH)
        self.doc_categories_df = self.process_ir_data_from_hf(DOC_CATEGORIES_VEC_PATH)
        self.doc_texts_df = self.process_ir_data_from_hf(DOC_TEXTS_PATH, is_text=True)

