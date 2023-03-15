import logging
import os
import json
import gradio as gr 
import openai 
import time
import sys
from utils import init_logger, GPT3_NAME_AND_COST


def prompt_gpt3(apikey: str, prompt_input: str, model_name='code-davinci-002', 
                max_tokens=128, **kwargs):
    openai.api_key = apikey
    i = 0
    while i < 3:
        try:
            response = openai.Completion.create(
                model=model_name,
                prompt=prompt_input,
                max_tokens=max_tokens,
                **kwargs
            )
            break
        except:
            error = sys.exc_info()[0]
            if error == openai.error.InvalidRequestError: 
                # something is wrong: e.g. prompt too long
                print(f"InvalidRequestError\nPrompt:\n\n{prompt_input}\n\n")
            print("API error:", error)
            time.sleep(1)
            i += 1
            response = str(error)

    if isinstance(response, str): # error batch
        text = response
    else:
        text = response['choices'][0]['text']
    return text


def run(apikey, model_name, input_prompt, temperature, max_new_tokens):
    response = prompt_gpt3(
        apikey, 
        input_prompt, # must be a list
        model_name=model_name,
        max_tokens=int(max_new_tokens), 
        temperature=temperature, 
    )
    info = {
        'model_name': model_name,
        'input_prompt': input_prompt,
        'temperature': temperature,
        'max_new_tokens': max_new_tokens,
        'model_output': response,
    }
    logger.info(json.dumps(info, ensure_ascii=False) + '\n')
    return response


iface = gr.Interface(
    fn=run, 
    inputs=[
        gr.Textbox(placeholder='Your OpenAI API KEY here', type='password', max_lines=500), # apikey
        gr.Dropdown(list(GPT3_NAME_AND_COST.keys()), value='text-davinci-003'), # model_name
        gr.Textbox(placeholder="Your prompt here"), # input_prompt
        gr.Slider(0, 1, value=0.9),  # temperature
        gr.Number(value=100), # max_new_tokens
        ],
    outputs="text",
    title="Local OpenAI Playground",
    layout="horizontal",
    enable_queue=True,
    allow_flagging=True,
    flagging_options=['Interesting!', 'Wrong!'],
)

logger = init_logger('logs/playground.log')
iface.launch()