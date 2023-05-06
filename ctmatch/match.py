
from typing import Dict, List, Optional, Set

# external imports
from sklearn.metrics.pairwise import cosine_similarity
from transformers.pipelines.pt_utils import KeyDataset
from pathlib import Path
from sklearn import svm
from tqdm import tqdm
import pandas as pd
import numpy as np
import spacy
import json

# package tools
from .models.classifier_model import ClassifierModel
from .utils.ctmatch_utils import get_processed_data
from .models.gen_model import GenModel
from .modelconfig import ModelConfig
from .pipetopic import PipeTopic
from .dataprep import DataPrep

# TODO: replace w huggingface dataset...
TREC_DATA = Path('/Users/jameskelly/Documents/cp/ctmatch/data/trec_data/trec_data.jsonl')
KZ_DATA = Path('/Users/jameskelly/Documents/cp/ctmatch/data/kz_data/kz_data.jsonl')


class CTMatch:
    def __init__(self, model_config: ModelConfig) -> None:
        self.model_config = model_config
        self.data = DataPrep(self.model_config)
        self.classifier_model = ClassifierModel(self.model_config, self.data)
        self.gen_model = GenModel(self.model_config)

        # embedding attrs
        self._spacy_model = None
        self._tfidf_model = None
        self.doc_embeddings = None

        # comparison matrices
        self.doc_tfidf_matrix = None
        self.doc_category_matrix = None
            


    def match_pipeline(self, topic: str, top_n: int, mode: str ='normal') -> List[str]:

        # get doc set
        doc_set = KeyDataset(self.data.ct_dataset, 'doc').to_pandas()

        # build comparison matrices
        self.build_comparison_matrices()
        
        # get topic representations for pipeline filters
        pipe_topic = PipeTopic(
            topic=topic, 
            classifier_model=self.classifier_model, 
            tfidf_model=self.tfidf_model,
            model_config=self.model_config
        )

        # first filter, category + tfidf similiarity
        doc_set = self.sim_filter(pipe_topic, doc_set, top_n=1000)

        # second filter, SVM
        doc_set = self.svm_filter(pipe_topic, doc_set, top_n=100)

        # third filter, LM
        doc_set = self.gen_filter(pipe_topic, doc_set, top_n=min(top_n, 100))

        return doc_set



    def sim_filter(self, pipe_topic: PipeTopic, doc_df: pd.DataFrame, top_n: int = 1000) -> List[str]:

        # get category similiarity
        category_sim = cosine_similarity(pipe_topic.topic_category_vector, self.doc_category_matrix).flatten()

        # first filter, top n docs by alpha similiarity
        top_n_indexes = np.argsort(category_sim)[::-1][:top_n]

        doc_df = self.get_filtered_docs(doc_df, get_only=top_n_indexes)
        return doc_df
    

    def svm_filter(self, topic_embedding, doc_df: pd.DataFrame, top_n: int = 100) -> List[int]:
        if self.doc_embeddings is None:
            self.data.create_doc_embeddings()
            
        x = np.concatenate([topic_embedding, doc_df['doc_embedding'].values.tolist()])
        y = np.zeros(len(doc_df) + 1)
        y[0] = 1
        clf = svm.LinearSVC(class_weight='balanced', verbose=False, max_iter=10000, tol=1e-6, C=0.1)
        clf.fit(x, y) 
        
        # infer for similarities
        similarities = clf.decision_function(x)
        sorted_neighbors = np.argsort(-similarities)
        return self.get_filtered_docs(doc_df, get_only=sorted_neighbors[:top_n])
        

    def gen_filter(self, pipe_topic: PipeTopic, doc_df: pd.DataFrame, top_n: int) -> List[str]:
        dist_dict_results = []
        for doc_text in doc_df['doc']:
            pos_ex, neg_ex = self.data.get_pos_neg_examples()
            dist_dict = self.gen_model.gen_relevance(pipe_topic.topic, doc_text, pos_ex, neg_ex)
            dist_dict['text'] = doc_text
            dist_dict_results.append(dist_dict)

        return sorted(dist_dict_results, key=lambda x: x['dist'], reverse=True)[:top_n]




    def get_filtered_docs(self, doc_df: pd.DataFrame, get_only: Optional[Set[int]] = None) -> pd.DataFrame:
        if get_only is None:
            return doc_df
        else:
            return doc_df.iloc[get_only]


    def prep_ir_text(self, doc: Dict[str, List[str]], max_len: int = 256) -> str:
        inc_text = ' '.join(doc['elig_crit']['include_criteria'])
        exc_text = ' '.join(doc['elig_crit']['exclude_criteria'])
        all_text = inc_text + exc_text
        split_text = all_text.split()
        return ' '.join(split_text[:min(max_len, len(split_text))])




    def prep_and_save_ir_dataset(self):
        category_data = self.data.get_category_data()
        with open(self.model_config.ir_save_path, 'w') as wf:
            for ir_data in self.prep_ir_data():
                ir_data['categories'] = str(category_data[ir_data['id']])
                wf.write(json.dumps(ir_data))
                wf.write('\n')


    def prep_ir_data(self):
        for data_path in self.model_config.processed_data_paths:
            for i, doc in enumerate(get_processed_data(data_path)):
                if i % 10000 == 0:
                    print(f"Prepping doc {i}")

                ir_data_entry = dict()
                ir_data_entry['id'] = doc['id']
                doc_text = self.prep_ir_text(doc)
                ir_data_entry['doc_text'] = doc_text
                yield ir_data_entry


    def save_texts(self):
        with open(Path(self.model_config.ir_save_path).parent / 'texts', 'w') as wf:
            for i, doc in enumerate(get_processed_data(self.model_config.ir_save_path)):
                if i % 10000 == 0:
                    print(f"Prepping doc {i}")

                #ir_data_entry['doc_embedding'] = self.gen_model.get_embedding(doc_text)
                #print(doc['doc_text'])
                wf.write(doc['doc_text'], 'utf-8')
                wf.write('\n')
       




if __name__ == '__main__':
    # trec_data_path = '/Users/jameskelly/Documents/cp/ctmatch/data/trec_data/trec21_labelled_triples.jsonl'
    # kz_data_path = '/Users/jameskelly/Documents/cp/ctmatch/data/kz_data/kz_labelled_triples.jsonl'
    # new_trec_data_path = '/Users/jameskelly/Documents/cp/ctmatch/data/trec_data/trec_data.jsonl'
    # new_kz_data_path = '/Users/jameskelly/Documents/cp/ctmatch/data/kz_data/kz_data.jsonl'
    # create_dataset(trec_data_path, new_trec_data_path)
    # create_dataset(kz_data_path, new_kz_data_path)

    scibert_model = 'allenai/scibert_scivocab_uncased'

    config = ModelConfig(
        name='{scibert_model}_ctmatch_finetuned',
        classified_data_path=KZ_DATA,
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
    ctm.classifier_model.train_and_predict()



