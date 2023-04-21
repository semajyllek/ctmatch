
from transformers import AutoTokenizer, AutoModelForCausalLM
from .scripts.gen_categories import gen_single_category_vector
from .modelconfig import ModelConfig
from .match import TfidfVectorizer


class PipeTopic:
    def __init__(
        self, 
        topic: str,
        tokenizer: AutoTokenizer, 
        lm_model: AutoModelForCausalLM, 
        tfidf_model: TfidfVectorizer,
        model_config: ModelConfig
    ):
        self.model_config = model_config
        self.topic = topic
        self.topic_tokenized = self.get_tokenized_topic(tokenizer=tokenizer)
        self.topic_category_vector = gen_single_category_vector(topic)
        self.topic_tfidf_vector = tfidf_model.transform([self.topic]).toarray()
        self.topic_embedding_vector = lm_model(**self.topic_tokenized).last_hidden_state
    
    
    def get_tokenized_topic(self, tokenizer: AutoTokenizer):
        return tokenizer(
            self.topic, 
            padding = self.model_config.padding,
            max_length = self.model_config.max_length, 
            truncate = self.model_config.truncatation,
            return_tensors='pt'
        )
    



