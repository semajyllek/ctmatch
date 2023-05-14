import gradio as gr

from typing import Optional
from ctmatch.match import CTMatch, PipeConfig



def ctmatch_web_api(topic_query: str, top_k: int = 10, openai_api_key: Optional[str] = None) -> None:
    pipe_config = PipeConfig(
        openai_api_key = openai_api_key,
        ir_setup=True,
        filters=['svm', "classifier"]
    )
    
    ctm = CTMatch(pipe_config)
    return ctm.match_pipeline(topic_query, top_k=top_k)


demo = gr.Interface(fn=ctmatch_web_api, inputs="text", outputs="text")
demo.launch()   