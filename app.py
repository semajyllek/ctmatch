
from ctmatch.match import CTMatch, PipeConfig
import gradio as gr


if __name__ == "__main__":
    pipe_config = PipeConfig(
        classifier_model_checkpoint='semaj83/scibert_finetuned_pruned_ctmatch',
        ir_setup=True,
        filters=["svm", "classifier"],
    )

    CTM = CTMatch(pipe_config)

    def ctmatch_web_api(topic_query: str) -> str:
        return '\n\n'.join([f"{nid}: {txt}" for nid, txt in CTM.match_pipeline(topic_query, top_k=5)])

    with gr.Blocks(css=".gradio-container {background-color: #00CED1}") as demo:
        name = gr.Textbox(lines=5, label="patient description", placeholder="Patient is a 45-year-old man with a history of anaplastic astrocytoma...")
        output = gr.Textbox(lines=10, label="matching trials")
        greet_btn = gr.Button("match")
        greet_btn.click(fn=ctmatch_web_api, inputs=name, outputs=output, api_name="match")

    demo.queue().launch(share=True, debug=True)