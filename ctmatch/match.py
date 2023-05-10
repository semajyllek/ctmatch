
import logging
from typing import Dict, List, Optional, Tuple

# external imports
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from numpy.linalg import norm
from pathlib import Path
from sklearn import svm
import numpy as np
import torch
import json

# package tools
from .models.classifier_model import ClassifierModel
from .utils.ctmatch_utils import get_processed_data
from .models.gen_model import GenModel
from .modelconfig import ModelConfig
from .pipetopic import PipeTopic
from .dataprep import DataPrep


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


CT_CATEGORIES = [
    "pulmonary", "cardiac", "gastrointestinal", "renal", "psychological", "genetic", "pediatric",
	"neurological", "cancer", "reproductive", "endocrine", "infection", "healthy", "other"
]


GEN_INIT_PROMPT =  "I will give you a patient description and a set of clinical trial documents. Each document will have a NCTID. I would like you to return the set of NCTIDs ranked from most to least relevant for patient in the description.\n"


class CTMatch:
    def __init__(self, model_config: Optional[ModelConfig] = None) -> None:
        # default to model config with full ir setup
        self.model_config = model_config if model_config is not None else ModelConfig(ir_setup=True)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.data = DataPrep(self.model_config)
        self.classifier_model = ClassifierModel(self.model_config, self.data, self.device)
        self.embedding_model = SentenceTransformer(self.model_config.embedding_model_checkpoint) 
        self.gen_model = GenModel(self.model_config)
        self.category_model = None
        
    
    # main api method
    def match_pipeline(self, topic: str, top_k: int, mode: str ='normal') -> List[str]:

        # start off will all doc indexes
        doc_set = [i for i in range(len(self.data.index2docid))]
        
        # get topic representations for pipeline filters
        pipe_topic = PipeTopic(
            topic_text=topic, 
            embedding_vec=self.get_embeddings([topic]),             # 1 x embedding_dim (default=384)
            category_vec=self.get_categories(topic)[np.newaxis, :]  # 1 x 14
        )

        # first filter, category + embedding similarity
        doc_set = self.sim_filter(pipe_topic, doc_set, top_n=10000)

        # second filter, SVM
        doc_set = self.svm_filter(pipe_topic, doc_set, top_n=100)

        # third filter, LM
        doc_set = self.gen_filter(pipe_topic, doc_set, top_n=min(top_k, 10))

        return self.get_return_data(doc_set)


    # ------------------------------------------------------------------------------------------ #
    # filtering methods
    # ------------------------------------------------------------------------------------------ #

    def sim_filter(self, pipe_topic: PipeTopic, doc_set: List[int], top_n: int = 1000) -> List[int]:
        """
        filter documents by similarity to topic
        doing this with loop and cosine similarity instead of linear kernel because of memory issues
        """
        logger.info(f"Running similarity filter on {len(doc_set)} documents")

       
        norm_topic_cat = norm(pipe_topic.category_vec)
        norm_topic_emb = norm(pipe_topic.embedding_vec)
        cosine_dists = []
        for doc_id in doc_set:
            cat_dist = np.dot(pipe_topic.category_vec, self.data.doc_categories_df.iloc[doc_id].values) / (norm_topic_cat * norm(self.data.doc_categories_df.iloc[doc_id].values))
            emb_dist = np.dot(pipe_topic.embedding_vec, self.data.doc_embeddings_df.iloc[doc_id].values) / (norm_topic_emb * norm(self.data.doc_embeddings_df.iloc[doc_id].values))
            combined_dist = cat_dist + emb_dist
            cosine_dists.append(combined_dist[0])

        # return top n doc indices by combined similiarity, biggest to smallest
        return list(np.argsort(cosine_dists))[:min(len(doc_set), top_n)]
    
       


    def svm_filter(self, topic: PipeTopic, doc_set: List[int], top_n: int = 100) -> List[int]:
        logger.info(f"Running svm filter on {len(doc_set)} documents")

        # build training data and prediction vector of single positive class for SVM
        x = np.concatenate([topic.embedding_vec, self.data.doc_embeddings_df.iloc[doc_set].values], axis=0)
        y = np.zeros(len(doc_set) + 1)
        y[0] = 1

        # define and fit SVM
        clf = svm.LinearSVC(class_weight='balanced', verbose=False, max_iter=10000, tol=1e-6, C=0.1)
        clf.fit(x, y) 
        
        # infer for similarities
        similarities = clf.decision_function(x)

        # get top n doc indices by similiarity, biggest to smallest
        result = list(np.argsort(-similarities)[:min(len(doc_set) + 1, top_n + 1)])

        # remove topic from result
        result.remove(0)

        # indexes got shifted by 1 because topic was included in doc_set
        return [(r - 1) for r in result]
    


    def gen_filter(self, topic: PipeTopic, doc_set: List[int], top_n: int = 10) -> List[int]:
        """
            gen model supplies a ranking of remaming docs by evaluating the pairs of topic and doc texts

            in order to overcome the context length limitation, we need to do a kind of left-binary search over multiple 
            prompts to arrive at a ranking that meets the number of documents requirement (top_n)

            may take a few minutes to run through all queries and subqueries depending on size of doc_set
        
        """
        logger.info(f"Running gen filter on {len(doc_set)} documents")

        assert top_n > 0, "top_n must be greater than 0"

        ranked_docs = doc_set
        iters = 0
        while (len(ranked_docs) > top_n) and (iters < 10) and (len(ranked_docs) // 2 > top_n):
            query_prompts = self.get_subqueries(topic, ranked_docs)

            logger.info(f"running gen model on {len(query_prompts)} subqueries")

            # get gen model response for each query_prompt
            subrankings = []
            for prompt in query_prompts:
                subrank = self.gen_model.gen_response(prompt)

                # keep the top half of each subranking
                subrankings.extend(subrank[:len(subrank) // 2])


            ranked_docs = subrankings
            iters += 1
        
        return ranked_docs[:min(len(ranked_docs), top_n)]

    # ------------------------------------------------------------------------------------------ #
    # filter helper methods
    # ------------------------------------------------------------------------------------------ #
    def get_embeddings(self, texts: List[str]) -> List[float]:
        return self.embedding_model.encode(texts)
    
    def get_categories(self, text: str) -> str:
        if self.category_model is None:
            self.category_model = pipeline(
                'zero-shot-classification', 
                model=self.model_config.category_model_checkpoint, 
                device=0
            )
        output = self.category_model(text, candidate_labels=CT_CATEGORIES)
        score_dict = {output['labels'][i]:output['scores'][i] for i in range(len(output['labels']))}

        # to be consistent with doc category vecs 
        sorted_keys = sorted(score_dict.keys())
        return np.array([score_dict[k] for k in sorted_keys])



    def get_gen_query_prompt(self, topic: PipeTopic, doc_set: List[int]) -> str:
        query_prompt = f"{GEN_INIT_PROMPT}Patient description: {topic.topic_text}\n"
        
        for i, doc_text in enumerate(self.data.doc_texts_df.iloc[doc_set].values):
            query_prompt += f"NCTID: {doc_set[i]}, "
            query_prompt += f"Eligbility Criteria: {doc_text[0]}\n"

            # not really token length bc not tokenized yet but close enough if we undershoot
            prompt_len = len(query_prompt.split()) 
            if prompt_len > self.model_config.max_query_length:
                break
    
        return query_prompt, i       


    def get_subqueries(self, topic: PipeTopic, doc_set: List[int]) -> List[str]:
        query_prompts = []
        i = 0
        while i < len(doc_set) - 1:

            # break the querying over remaining doc set into multiple prompts
            query_prompt, used_i = self.get_gen_query_prompt(topic, doc_set[i:])
            query_prompts.append(query_prompt)
            i += used_i

        return query_prompts


    def get_return_set(self, doc_set: List[int]) -> List[Tuple[str, str]]:
        return_set = []
        for idx in doc_set:
            nctid = self.data.index2docid.iloc[doc_set].values[0]
            return_set.append((nctid, self.data.doc_texts_df.iloc[idx].values[0]))
        return return_set



    # ------------------------------------------------------------------------------------------ #
    # data prep methods that rely on model in CTMatch object (not run during routine program) 
    # ------------------------------------------------------------------------------------------ #

    def prep_ir_text(self, doc: Dict[str, List[str]], max_len: int = 512) -> str:
        inc_text = ' '.join(doc['elig_crit']['include_criteria'])
        exc_text = ' '.join(doc['elig_crit']['exclude_criteria'])
        all_text = f"Inclusion Criteria: {inc_text}, Exclusion Criteria: {exc_text}"
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
                    logger.info(f"Prepping doc {i}")

                ir_data_entry = dict()
                ir_data_entry['id'] = doc['id']
                doc_text = self.prep_ir_text(doc)
                ir_data_entry['doc_text'] = doc_text
                yield ir_data_entry


    def save_texts(self) -> Dict[int, str]:
        idx2id = dict()
        with open(Path(self.model_config.ir_save_path).parent / 'texts', 'w', encoding='utf-8') as wf:
            for i, doc in enumerate(get_processed_data(self.model_config.ir_save_path)):
                idx2id[i] = doc['id']
                if i % 10000 == 0:
                    logger.info(f"Prepping doc {i}")

                wf.write(doc['doc_text'])
                wf.write('\n')
        return idx2id

