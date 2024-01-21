import gradio as gr

from conversation import *
from index import *

def handle_file_upload(file_path: str) -> str:
    return file_path

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            file = gr.components.File(
                label='Upload your pdf file',
                file_count='single',
                file_types=['.pdf'],
                interactive=True)
            with gr.Row():
                upload = gr.components.Button(
                    value='Upload', variant='primary')
                index_clear_btn = gr.components.Button(
                    value='Clear', variant='stop')
        label = gr.components.Textbox()
    file.upload(handle_file_upload, [file], [file], trigger_mode='once')
    chatbot = gr.Chatbot(label='Talk to the Douments')
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])
    upload.click(create_index, [file], [label])
    index_clear_btn.click(clear_index, [], [label, file])
    msg.submit(
        handle_user_query,
        [msg, chatbot],
        [msg, chatbot]
    ).then(
        create_llm_conversation,
        [chatbot],
        [chatbot]
    )

demo.queue()
