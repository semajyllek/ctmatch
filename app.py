from ctmatch.match import CTMatch, PipeConfig
import gradio as gr


pipe_config = PipeConfig(
    classifier_model_checkpoint='semaj83/scibert_finetuned_pruned_ctmatch',
    ir_setup=True,
    filters=["svm", "classifier"],
)

CTM = CTMatch(pipe_config)


def ctmatch_web_api(topic_query: str, topK: int = 5) -> str:
    return '\n\n'.join([f"{nid}: {txt}" for nid, txt in CTM.match_pipeline(topic=topic_query, top_k=int(topK))])


if __name__ == "__main__":
  

    with gr.Blocks(css=".gradio-container {background-color: #00CED1}") as demo:
        name = gr.Textbox(lines=5, label="patient description", placeholder="Patient is a 45-year-old man with a history of anaplastic astrocytoma...")
        topK = gr.Number(label='topK', info='number of documents to return, <= 50', value=5)
        output = gr.Textbox(lines=10, label="matching trials")
        greet_btn = gr.Button("match")
        greet_btn.click(fn=ctmatch_web_api, inputs=[name, topK], outputs=output, api_name="match")

    demo.queue().launch(debug=True)