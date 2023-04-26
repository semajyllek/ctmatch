
from typing import List, Optional, Set

# external imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers.pipelines.pt_utils import KeyDataset
from pathlib import Path
from sklearn import svm
import pandas as pd
import numpy as np
import spacy

# package tools
from .models.gen_model import GenModel
from .modelconfig import ModelConfig
from .models.classifier_model import ClassifierModel
from .pipetopic import PipeTopic
from .dataprep import DataPrep


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
            


    def match_pipeline(self, topic: str, top_k: int) -> List[str]:

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
        doc_set = self.gen_filter(pipe_topic, doc_set, top_k=min(top_k, 100))

        return doc_set



    def sim_filter(self, pipe_topic: PipeTopic, doc_df: pd.DataFrame, top_n: int = 1000) -> List[str]:

        # get tfidf similiarity, dims: len(doc_df) x 1
        tfidf_sim = cosine_similarity(pipe_topic.topic_tfidf_vector, self.doc_tfidf_matrix).flatten()

        # get category similiarity
        category_sim = cosine_similarity(pipe_topic.topic_category_vector, self.doc_category_matrix).flatten()

        alpha_sim = tfidf_sim + category_sim

        # first filter, top n docs by alpha similiarity
        top_n_indexes = np.argsort(alpha_sim)[::-1][:top_n]

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
        

    def gen_filter(self, pipe_topic: PipeTopic, doc_df: pd.DataFrame, top_k: int) -> List[str]:
        dist_dict_results = []
        for doc_text in doc_df['doc']:
            pos_ex, neg_ex = self.data.get_pos_neg_examples()
            dist_dict = self.gen_model.gen_relevance(pipe_topic.topic, doc_text, pos_ex, neg_ex)
            dist_dict['text'] = doc_text
            dist_dict_results.append(dist_dict)

        return sorted(dist_dict_results, key=lambda x: x['dist'], reverse=True)[:top_k]




    def get_filtered_docs(self, doc_df: pd.DataFrame, get_only: Optional[Set[int]] = None) -> pd.DataFrame:
        if get_only is None:
            return doc_df
        else:
            return doc_df.iloc[get_only]
        

    def build_comparison_matrices(self) -> None:
        if self.doc_tfidf_matrix is None:
            self.build_doc_tfidf_matrix()

        if self.doc_category_matrix is None:
            self.build_doc_category_matrix()


    def build_doc_tfidf_matrix(self) -> None:
        self.doc_tfidf_matrix = self.tfidf_model.fit_transform(self.data.ct_dataset_df['doc'])

    
    def build_doc_category_matrix(self) -> None:
        self.doc_category_matrix = self.data.ct_dataset_df['category']



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
            self._tfidf_model = TfidfVectorizer(stop_words=['[PAD]'])
        return self.tfidf_model
    
    def get_sim_model(self, model):
        if model == 'spacy':
            return self.get_spacy_embedding_similarity
        elif model == 'tfidf':
            return self.get_tfidf_similarity
        else:
            raise ValueError(f"model {model} not supported")

    def get_top_n_similarities(self, topic, model: str, n=10):
        sim_model = self.get_sim_model(model)
        similarities = {}
        for i, doc in enumerate(self.doc_embeddings):
            similarities[i] = sim_model(topic, doc)

        sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        return sorted_similarities[:n]

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
        name='{scibert_model}_ctmatch_finetuned',
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



