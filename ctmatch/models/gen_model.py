
from typing import List
from ctmatch.modelconfig import ModelConfig
import openai

INIT_PROMPT = init_prompt_starter = "I will give you a patient description and a set of clinical trial documents. Each document will have a NCTID. I would like you to return the set of NCTIDs ranked from most to least relevant for patient in the description.\n"


class GenModel:
    def __init__(self, model_config: ModelConfig):
        self.tokenizer = None
        self.model = None
        self.model_config = model_config
        openai.api_key = "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"

    def gen_response(self, query_prompt):
        """
        uses openai model to return a ranking of ids
        """
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=query_prompt,
            temperature=0,
            max_tokens=200,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
  
        return self.post_process_response(response['choices'][0]['text'])

    def post_process_response(self, response):
        """
        could be:
        NCTID 6, NCTID 7, NCTID 5
        NCTID: 6, 7, 5
        6, 7, 5
        """
        s = []
        for substr in response.split(','):
            s.append(int(self.remove_leading_text(substr)))
        return s

    def remove_leading_text(self, s: str) -> str:
        i = 0
        ch = s[i]
        while not ch.isdigit():
            i += 1
            ch = s[i]
        return s[i:]
    


    def get_query_prompt(self, topic, df, max_n=10) -> List[str]:
        query_prompt = f"{INIT_PROMPT}Patient description: {topic}\n"
    
        for i, (_, doc, label) in df.iterrows():
            if i > max_n:
                break
            query_prompt += f"NCTID: {i}, "
            query_prompt += f"Eligbility Criteria: {doc}\n"

        return query_prompt
	