
from typing import List

from ..modelconfig import ModelConfig
import openai




class GenModel:
    def __init__(self, model_config: ModelConfig) -> None:
        openai.api_key = model_config.openai_api_key
        self.model_config = model_config


    def gen_response(self, query_prompt: str) -> List[int]:
        """
        uses openai model to return a ranking of ids
        """
        if self.model_config.gen_model_checkpoint == 'text-davinci-003':
            response = openai.Completion.create(
                model=self.model_config.gen_model_checkpoint,
                prompt=query_prompt,
                temperature=0,
                max_tokens=200,
                top_p=1,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
        else:
            # for gpt-3.5-turbo
            response = openai.ChatCompletion.create(
                model=self.model_config.gen_model_checkpoint,
                messages = {'role': 'user', 'content' : query_prompt},
                temperature=0.4,
                max_tokens=200,
                top_p=1,
                frequency_penalty=0.2,
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
        return [int(self.remove_leading_text(substr)) for substr in response.split(',')]

    def remove_leading_text(self, s: str) -> str:
        i = 0
        ch = s[i]
        while not ch.isdigit():
            i += 1
            ch = s[i]
        return s[i:]
    



	