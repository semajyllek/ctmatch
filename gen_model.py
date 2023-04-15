
from ctmatch.model_config import ModelConfig
from transformers import set_seed
import random
import torch



class GenModel:
    def __init__(self, model_config: ModelConfig):
        self.gen_tokenizer = None
        self.gen_model = None
        self.add_gen_model(model_config.gen_model)
        
def add_gen_model(self, model_name='biogpt') -> None:
    if model_name == 'biogpt':
        from transformers import BioGptTokenizer, BioGptForCausalLM,
        self.gen_tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
        self.gen_model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")
    elif model_name == 'gpt2':
        from transformers import GPT2Tokenizer, GPT2LMHeadModel
        self.gen_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.gen_model = GPT2LMHeadModel.from_pretrained("gpt2")
    else:
        raise ValueError(f"Model name {model_name} not supported")


def get_random_example(self, target_label: str):
    return self.ct_dataset_df.where(self.ct_dataset_df['label'] == target_label).sample(1)

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
	inputs = self.gen_tokenizer(prompt, return_tensors="pt")
	set_seed(self.model_config.seed)
	with torch.no_grad():
		beam_output = self.gen_model.generate(
						**inputs,
						min_length=100,
						max_length=1024,
						num_beams=5,
						early_stopping=True
					)
	return self.gen_tokenizer.decode(beam_output[0], skip_special_tokens=True)

def infer_relevance_from_gen_topic(self, topic, document) -> int:
    pos_example = self.ct_dataset_df.where(self.ct_dataset_df['label'] == '2').sample(1)
    neg_example = self.ct_dataset_df.where(self.ct_dataset_df['label'] == '0').sample(1)
    
	# gen pos topic example
	prompt = f"here is a clinical trial document: {pos_example['doc']}. here is a topic that recieves a 2, or relevant score for this document: {pos_example['topic']}, "
    prompt += f" here is another clinical trial document: {document}, here is a topic that recieves a 2, or relevant score for this document: "
    pseudo_pos_topic = self.gen_response(prompt)
    
	# gen neg topic example
	prompt = f"here is a clinical trial document: {neg_example['doc']}. here is a topic that recieves a 0, or not relevant score for this document: {neg_example['topic']}, "
    prompt += f" here is another clinical trial document: {document}, here is a topic that recieves a 0, or relevant score for this document: "
    pseudo_neg_topic = self.gen_response(prompt)
    
	return self.infer_rel_from_topic_similarity(pseudo_pos_topic, pseudo_neg_topic, topic)