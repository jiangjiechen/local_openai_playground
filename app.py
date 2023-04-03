import logging
import os
import json
import gradio as gr 
import openai 
import time
import sys
from utils import init_logger, GPT3_NAME_AND_COST


def prompt_gpt3(prompt_input: str, model_name='text-davinci-003', 
                max_tokens=128, stream=True, **kwargs):
    response = openai.Completion.create(
        model=model_name,
        prompt=prompt_input,
        max_tokens=max_tokens,
        stream=stream,
        **kwargs
    )

    if stream:
        for chunk in response:
            text = chunk['choices'][0]['text']
            if chunk['choices'][0]['finish_reason'] is None:
                yield text
    else:
        # handle error batch
        text = response if isinstance(response, str) else response['choices'][0]['text']
        return text


def run(model_name, input_prompt, temperature, max_new_tokens):
    prev_text = ''
    for chunk in prompt_gpt3(
        input_prompt, # must be a list
        model_name=model_name,
        max_tokens=int(max_new_tokens), 
        temperature=temperature, 
        stream=True
    ):
        prev_text += chunk
        yield prev_text
    
    info = {
        'model_name': model_name,
        'input_prompt': input_prompt,
        'temperature': temperature,
        'max_new_tokens': int(max_new_tokens),
        'model_output': prev_text,
    }
    logger.info(json.dumps(info, ensure_ascii=False) + '\n')
    

openai.api_key = os.environ['OPENAI_API_KEY']
iface = gr.Interface(
    fn=run, 
    inputs=[
        # gr.Textbox(placeholder='Your OpenAI API KEY here', type='password', max_lines=500), # apikey
        gr.Dropdown(list(GPT3_NAME_AND_COST.keys()), value='text-davinci-003'), # model_name
        gr.Textbox(placeholder="Your prompt here"), # input_prompt
        gr.Slider(0, 2, value=0.9, label='Temperature'),  # temperature
        gr.Number(value=100), # max_new_tokens
        ],
    outputs=gr.Textbox(label='Output'),
    title="Local OpenAI Playground",
    layout="horizontal",
    allow_flagging='auto',
    flagging_options=['👍', '👎'],
)

logger = init_logger('logs/playground.log')
iface.launch(
    enable_queue=True,
)