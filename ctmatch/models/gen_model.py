
from transformers import BioGptTokenizer, BioGptForCausalLM


from ctmatch.modelconfig import ModelConfig
from transformers import set_seed
from numpy.linalg import norm
import numpy as np
import torch

from ctmatch.utils.ctmatch_utils import cosine_sim



class GenModel:
    def __init__(self, model_config: ModelConfig):
        self.tokenizer = None
        self.model = None
        self.model_config = model_config
        self.add_gen_model(model_config.gen_model)
    
    def add_gen_model(self, model_name='biogpt') -> None:
        if model_name == 'biogpt':
            self.tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
            self.model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")
        elif model_name == 'gpt2':
            from transformers import GPT2Tokenizer, GPT2LMHeadModel
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        else:
            raise ValueError(f"Model name {model_name} not supported")


    def build_rel_prompt(self, topic, document) -> str:
        neg_pair = self.get_random_example(target_label='0')
        neutral_pair = self.get_random_example(tagret_label='1')
        pos_pair = self.get_random_example(tagret_label='2')
        p_neg = f"topic: {neg_pair['topic']}, trial document: {neg_pair['doc']}, relevance score: 0"
        p_neutr = f"topic: {neutral_pair['topic']}, trial document: {neutral_pair['doc']}, relevance score: 1"
        p_pos = f"topic: {pos_pair['topic']}, trial document: {pos_pair['doc']}, relevance score: 2"
        prompt = f"{p_neg}, {p_pos}, {p_neutr}, topic: {topic}, trial document: {document}, relevance score: " 
        return prompt


    def generate_relevance(self, topic, document) -> int:
        prompt = self.build_rel_prompt(topic, document)
        return self.gen_response(prompt)
            

    def gen_response(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        set_seed(self.model_config.seed)
        with torch.no_grad():
            beam_output = self.model.generate(
                            **inputs,
                            min_length=100,
                            max_length=1024,
                            num_beams=5,
                            early_stopping=True
                        )
        return self.tokenizer.decode(beam_output[0], skip_special_tokens=True)


    def get_embedding(self, s: str):
        input = self.tokenizer(s, return_tensors='pt')
        with torch.no_grad():
            embedding = self.model(input['input_ids']).hidden_states[-1]

        return norm(embedding)

    def gen_sim_pos_topic_embedding(self, doc_text, pos_example, debug=False):
        prompt = f"here is a clinical trial document: {pos_example['doc']}. here is a topic that recieves a 2, or 'relevant' score for this document: {pos_example['topic']}, "
        prompt += f" here is another clinical trial document: {doc_text}, here is a topic that recieves a 2, or 'relevant' score for this document: "
        sim_pos_topic = self.gen_response(prompt)
        if debug:
            print(sim_pos_topic)

        return norm(self.get_embedding(sim_pos_topic))
    
    def gen_sim_neg_topic_embedding(self, doc_text, neg_example, debug=False):
        prompt = f"here is a clinical trial document: {neg_example['doc']}. here is a topic that recieves a 0, or 'not relevant' score for this document: {neg_example['topic']}, "
        prompt += f" here is another clinical trial document: {doc_text}, here is a topic that recieves a 0, or 'not relevant' score for this document: "
        sim_neg_topic = self.gen_response(prompt)
        if debug:
            print(sim_neg_topic)

        return norm(self.get_embedding(sim_neg_topic))


    def gen_relevance(self, topic, document, pos_example, neg_example) -> int:
        sim_pos_topic_embedding = self.gen_sim_pos_topic_embedding(document, pos_example)
        sim_neg_topic_embedding = self.gen_sim_neg_topic_embedding(document, neg_example)
        topic_embedding = norm(self.get_embedding(topic))
        return self.infer_rel_from_topic_similarity(sim_pos_topic_embedding, sim_neg_topic_embedding, topic_embedding)

    def infer_rel_from_topic_similarity(self, pos_topic, neg_topic, topic, neutral_margin: float = 0.001) -> int:
        pos_dist = cosine_sim(pos_topic, topic)
        neg_dist = cosine_sim(neg_topic, topic)
        if pos_dist < neg_dist:

            # it's closer to the positive example, but close enough to not be neutral?
            if pos_dist < neutral_margin:
                return {'relevance': 1, 'distance': pos_dist}
            return {'relevance': 2, 'distance': pos_dist}
        else:
            # closer to negative example, but close enough to not be neutral?
            if neg_dist < neutral_margin:
                return {'relevance': 1, 'distance': neg_dist}
            return {'relevance':0, 'distance': neg_dist}
        
