from typing import Annotated
import gradio as gr
from fastapi import FastAPI, Form

from gradio_ui import demo
from database_api import *
from conversation import *
from twilio_api import *

app = FastAPI()


@app.get('/')
async def root():
    return 'Gradio app is running at /gradio', 200


@app.post('/twilio')
async def handle_twilio(Body: Annotated[str, Form()], From: Annotated[str, Form()], ProfileName: Annotated[str, Form()]):
    try:
        query = Body
        sender_id = From
        user_name = ProfileName
        user = get_user(sender_id)
        if user:
            response = create_llm_conversation_backend(
                user['messages'][-2:], query)
        else:
            response = create_llm_conversation_backend([], query)
        if user:
            update_messages(sender_id, query,
                            response, user['messageCount'])
        else:
            message = {
                'query': query,
                'response': response,
                'createdAt': datetime.now().strftime('%d/%m/%Y, %H:%M')
            }
            user = {
                'userName': user_name,
                'senderId': sender_id,
                'messages': [message],
                'messageCount': 1,
                'mobile': sender_id.split(':')[-1],
                'channel': 'WhatsApp',
                'is_paid': False,
                'created_at': datetime.now().strftime('%d/%m/%Y, %H:%M')
            }
            create_user(user)
        send_message(sender_id, response)
    except:
        pass

    return 'OK', 200


app = gr.mount_gradio_app(app, demo, path='/gradio')
