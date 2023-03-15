import logging
import os
import json
import gradio as gr 
import openai 
import time
import sys


GPT3_NAME_AND_COST = {
    'text-davinci-003': 0.02,
    'text-davinci-002': 0.02,
    'code-davinci-002': 0,
    'text-davinci-001': 0.02,
    'davinci': 0.02,
    'text-curie-001': 0.002,
    'curie': 0.002,
    'babbage': 0.0005,
    'ada': 0.0004,
}


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


def init_logger(log_file=None, log_file_level=logging.NOTSET, from_scratch=False):
    from coloredlogs import ColoredFormatter

    fmt = "[%(asctime)s %(levelname)s] %(message)s"
    log_format = ColoredFormatter(fmt=fmt)
    # log_format = logging.Formatter()
    logger = logging.getLogger()
    logger.setLevel(log_file_level)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != '':
        if from_scratch and os.path.exists(log_file):
            logger.warning('Removing previous log file: %s' % log_file)
            os.remove(log_file)
        path = os.path.dirname(log_file)
        os.makedirs(path, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger


def run(model_name, input_prompt, temperature, max_new_tokens):
    response = prompt_gpt3(
        'sk-', 
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
        # gr.Textbox(placeholder='Your OpenAI APIKEY here', type='password', invisible=True), # apikey
        gr.Dropdown(list(GPT3_NAME_AND_COST.keys()), value='text-davinci-003'), # model_name
        gr.Textbox(placeholder="Your prompt here", max_lines=500), # input_prompt
        gr.Slider(0, 1, value=0.9),  # temperature
        gr.Number(value=100), # max_new_tokens
        ],
    outputs=gr.Textbox(max_lines=500),
    title="Local OpenAI Playground",
    layout="horizontal",
    enable_queue=True,
    allow_flagging=True,
    flagging_options=['Interesting!', 'Wrong!'],
)

logger = init_logger('logs/playground-public.log', logging.INFO)
iface.launch(share=True)