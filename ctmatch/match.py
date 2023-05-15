
import logging
from typing import Any, Dict, List, Optional, Tuple


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
from .utils.ctmatch_utils import get_processed_data, exclusive_argmax
from .models.gen_model import GenModel
from .pipeconfig import PipeConfig
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
    
    def __init__(self, pipe_config: Optional[PipeConfig] = None) -> None:
        # default to model config with full ir setup
        self.pipe_config = pipe_config if pipe_config is not None else PipeConfig(ir_setup=True)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.data = DataPrep(self.pipe_config)
        self.classifier_model = ClassifierModel(self.pipe_config, self.data, self.device)
        self.embedding_model = SentenceTransformer(self.pipe_config.embedding_model_checkpoint) 
        self.gen_model = GenModel(self.pipe_config)
        self.category_model = None
        self.filters: Optional[List[str]] = pipe_config.filters

        # filter params
        self.sim_top_n = 10000
        self.svm_top_n = 1000
        self.classifier_top_n = 100
        self.gen_top_n = 10


    # main api method
    def match_pipeline(self, topic: str, top_k: int = 10, doc_set: Optional[List[int]] = None) -> List[str]:

        if doc_set is None:
            # start off will all doc indexes
            doc_set = [i for i in range(len(self.data.index2docid))]
        else:
            self.reset_filter_params(len(doc_set))
 
        # get topic representations for pipeline filters
        pipe_topic = self.get_pipe_topic(topic)

        if self.filters is None or ('sim' in self.filters):
            # first filter, category + embedding similarity
            doc_set = self.sim_filter(pipe_topic, doc_set, top_n=self.sim_top_n)

        if self.filters is None or ('svm' in self.filters):
            # second filter, SVM
            doc_set = self.svm_filter(pipe_topic, doc_set, top_n=self.svm_top_n)

        if self.filters is None or ('classifier' in self.filters):
            # third filter, classifier-LM (reranking)
            doc_set = self.classifier_filter(pipe_topic, doc_set, top_n=self.classifier_top_n)

        if self.filters is None or ('gen' in self.filters):
            # fourth filter, generative-LM
            doc_set = self.gen_filter(pipe_topic, doc_set, top_n=top_k)

        return self.get_return_data(doc_set[:min(top_k, len(doc_set))])


    def reset_filter_params(self, val: int) -> None:
        self.sim_top_n = self.svm_top_n = self.classifier_top_n = self.gen_top_n = val
  

    # ------------------------------------------------------------------------------------------ #
    # filtering methods
    # ------------------------------------------------------------------------------------------ #

    def sim_filter(self, pipe_topic: PipeTopic, doc_set: List[int], top_n: int) -> List[int]:
        """
        filter documents by similarity to topic
        doing this with loop and cosine similarity instead of linear kernel because of memory issues
        """
        logger.info(f"running sim filter on {len(doc_set)} docs")

        topic_cat_vec = exclusive_argmax(pipe_topic.category_vec)
        norm_topic_emb = norm(pipe_topic.embedding_vec)
        cosine_dists = []
        for doc_idx in doc_set:
            doc_cat_vec = self.redist_other_category(self.data.doc_categories_df.iloc[doc_idx].values)
        
            # only consider strongest predicted category
            doc_cat_vec = exclusive_argmax(doc_cat_vec)
            doc_emb_vec = self.data.doc_embeddings_df.iloc[doc_idx].values

            topic_argmax = np.argmax(topic_cat_vec)
            doc_argmax = np.argmax(doc_cat_vec)
            cat_dist = 0. if (topic_argmax == doc_argmax) else 1.
            emb_dist = np.dot(pipe_topic.embedding_vec, doc_emb_vec) / (norm_topic_emb * norm(doc_emb_vec))
            combined_dist = cat_dist + emb_dist
            cosine_dists.append(combined_dist)

        sorted_indices = list(np.argsort(cosine_dists))[:min(len(doc_set), top_n)]

        # return top n doc indices by combined similiarity, biggest to smallest
        return [doc_set[i] for i in sorted_indices]
    

    def svm_filter(self, topic: PipeTopic, doc_set: List[int], top_n: int) -> List[int]:
        """
           filter documents by training an SVM on topic and doc embeddings
        """
        logger.info(f"running svm filter on {len(doc_set)} documents")

        # build training data and prediction vector of single positive class for SVM
        topic_embedding_vec = topic.embedding_vec[np.newaxis, :]
        x = np.concatenate([topic_embedding_vec, self.data.doc_embeddings_df.iloc[doc_set].values], axis=0)
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
        return [doc_set[(r - 1)] for r in result]



    def classifier_filter(self, pipe_topic: PipeTopic, doc_set: List[int], top_n: int) -> List[int]:
        """
        filter documents by classifier no relevance prediction
        """
        logger.info(f"running classifier filter on {len(doc_set)} documents")

        # get doc texts
        doc_texts = [v[0] for v in self.data.doc_texts_df.iloc[doc_set].values]
    
        # sort by reverse irrelevant prediction
        neg_predictions = np.asarray([self.classifier_model.run_inference_single_example(pipe_topic.topic_text, dtext, return_preds=True)[0] for dtext in doc_texts])
       
        # return top n doc indices by classifier, biggest to smallest
        sorted_indices = list(np.argsort(neg_predictions)[:min(len(doc_set), top_n)])
        return [doc_set[i] for i in sorted_indices]
       


    def gen_filter(self, topic: PipeTopic, doc_set: List[int], top_n: int = 10) -> List[int]:
        """
            gen model supplies a ranking of remaming docs by evaluating the pairs of topic and doc texts

            in order to overcome the context length limitation, we need to do a kind of left-binary search over multiple 
            prompts to arrive at a ranking that meets the number of documents requirement (top_n)

            may take a few minutes to run through all queries and subqueries depending on size of doc_set
        
        """
        logger.info(f"running gen filter on {len(doc_set)} documents")

        assert top_n > 0, "top_n must be greater than 0"

        ranked_docs = doc_set
        iters = 0
        while (len(ranked_docs) > top_n) and (iters < 10) and (len(ranked_docs) // 2 > top_n):
            query_prompts = self.get_subqueries(topic, ranked_docs)

            logger.info(f"calling gen model on {len(query_prompts)} subqueries")

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

    def get_pipe_topic(self, topic):
        pipe_topic = PipeTopic(
            topic_text=topic, 
            embedding_vec=self.get_embeddings([topic])[0],             # 1 x embedding_dim (default=384)
            category_vec=self.get_categories(topic)                    # 1 x 14
        )
        return pipe_topic
    

    def get_embeddings(self, texts: List[str]) -> List[float]:
        return self.embedding_model.encode(texts)
    
    def get_categories(self, text: str) -> str:
        if self.category_model is None:
            self.category_model = pipeline(
                'zero-shot-classification', 
                model=self.pipe_config.category_model_checkpoint, 
                device=0
            )
        output = self.category_model(text, candidate_labels=CT_CATEGORIES)
        score_dict = {output['labels'][i]:output['scores'][i] for i in range(len(output['labels']))}

        # to be consistent with doc category vecs 
        sorted_keys = sorted(score_dict.keys())
        return self.redist_other_category(np.array([score_dict[k] for k in sorted_keys]))

    def redist_other_category(self, category_vec: np.ndarray, other_dim:int = 8) -> np.ndarray:
        """
            redistribute 'other' category weight to all other categories
        """
        other_wt = category_vec[other_dim] 
        other_wt_dist = other_wt / (len(category_vec) - 1)
        redist_cat_vec = category_vec + other_wt_dist
        redist_cat_vec[other_dim] = 0
        return redist_cat_vec
        

    def get_gen_query_prompt(self, topic: PipeTopic, doc_set: List[int]) -> str:
        query_prompt = f"{GEN_INIT_PROMPT}Patient description: {topic.topic_text}\n"
        
        for i, doc_text in enumerate(self.data.doc_texts_df.iloc[doc_set].values):
            query_prompt += f"NCTID: {doc_set[i]}, "
            query_prompt += f"Eligbility Criteria: {doc_text[0]}\n"

            # not really token length bc not tokenized yet but close enough if we undershoot
            prompt_len = len(query_prompt.split()) 
            if prompt_len > self.pipe_config.max_query_length:
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


    def get_return_data(self, doc_set: List[int]) -> List[Tuple[str, str]]:
        return_data = []
        for idx in doc_set:
            nctid = self.data.index2docid.iloc[idx].values[0]
            return_data.append((nctid, self.data.doc_texts_df.iloc[idx].values[0]))
        return return_data



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
        with open(self.pipe_config.ir_save_path, 'w') as wf:
            for ir_data in self.prep_ir_data():
                ir_data['categories'] = str(category_data[ir_data['id']])
                wf.write(json.dumps(ir_data))
                wf.write('\n')


    def prep_ir_data(self):
        for data_path in self.pipe_config.processed_data_paths:
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
        with open(Path(self.pipe_config.ir_save_path).parent / 'texts', 'w', encoding='utf-8') as wf:
            for i, doc in enumerate(get_processed_data(self.pipe_config.ir_save_path)):
                idx2id[i] = doc['id']
                if i % 10000 == 0:
                    logger.info(f"Prepping doc {i}")

                wf.write(doc['doc_text'])
                wf.write('\n')
        return idx2id
    
