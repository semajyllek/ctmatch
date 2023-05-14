import gradio as gr

from ctmatch.match import CTMatch, PipeConfig



def ctmatch_web_api(topic_query: str) -> str:
    progress=gr.Progress()
    progress(0, desc="loading CTMatch...")
    
    pipe_config = PipeConfig(
        ir_setup=True,
        progress=progress,
        filters=['svm', "classifier"]
    )
    
    ctm = CTMatch(pipe_config)
    return '\n'.join([f"{nid}: {txt}" for nid, txt in ctm.match_pipeline(topic_query, top_k=5, progress=gr.Progress())])


with gr.Blocks(css=".gradio-container {background-color: #00CED1}") as demo:
    name = gr.Textbox(lines=5, label="patient description", placeholder="Patient is a 45-year-old man with a history of anaplastic astrocytoma...")
    output = gr.Textbox(lines=10, label="matching trials")
    greet_btn = gr.Button("match")
    greet_btn.click(fn=ctmatch_web_api, inputs=name, outputs=output, api_name="match")

demo.queue().launch(share=True)