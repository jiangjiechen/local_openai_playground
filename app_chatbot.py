import gradio as gr
import random
import time
import openai
import os
import sys
from utils import init_logger, GPT3_NAME_AND_COST


def prompt_chatgpt(system_input, user_input, history=[], temperature=0.7, 
                   max_tokens=256, model_name='gpt-3.5-turbo', stream=True):
    '''
    :param system_input: "You are a helpful assistant/translator."
    :param user_input: you texts here
    :param history: ends with assistant output.
                    e.g. [{"role": "system", "content": xxx},
                          {"role": "user": "content": xxx},
                          {"role": "assistant", "content": "xxx"}]
    return: assistant_output, (updated) history, money cost
    '''
    if len(history) == 0:
        if system_input.strip() == '': system_input = 'You are a helpful assistant.'
        history = [{"role": "system", "content": system_input}]
    history.append({"role": "user", "content": user_input})

    completion = openai.ChatCompletion.create(
        model=model_name,
        messages=history,
        max_tokens=int(max_tokens),
        stream=stream,
        temperature=temperature,
    )
    if stream:
        assistant_output = ""
        for chunk in completion:
            word = chunk['choices'][0]['delta']
            if len(word) > 0 and word.get('role') is None:
                assistant_output += word['content']
                yield assistant_output, history
    else:
        assistant_output = completion['choices'][0]['message']['content']
        history.append({"role": "assistant", "content": assistant_output})
        # total_tokens = completion['usage']['total_tokens']
        yield assistant_output, history


def convert_chatgpt_history(x, backward=False):
    # input: [{'role': 'user', 'content': 'xxx'}, {'role': 'assistant', 'content': 'yyy'}]
    # output: [[xxx, yyy], ...]
    
    output = []
    if backward:    # output -> input
        for i in range(len(x)):
            output.append({"role": 'user', "content": x[i][0]})
            if x[i][1] is not None:
                output.append({"role": 'assistant', "content": x[i][1]})
    else:           # input -> output
        for i in range(len(x)):
            if i % 2 == 0:
                output.append([x[i]['content']])
            else:
                output[-1].append(x[i]['content'])
    return output


def regenerate(model_name, sys_in, history, temperature, max_tokens):
    history = history[:-1]
    if len(history) > 0:
        history[-1][1] = None
    else:
        # for empty input check in `bot()`
        history.append(['', None])
    yield from bot(model_name, sys_in, history, temperature, max_tokens)


def bot(model_name, sys_in, history, temperature, max_tokens):
    yield history, disable_btn, disable_btn, disable_btn
    user_input = history[-1][0]
    if user_input.strip() == '': # empty input check
        history = history[:-1]
        yield history, enable_btn, enable_btn, enable_btn
        return

    chatgpt_history = convert_chatgpt_history(history[:-1], backward=True)
    
    chatgpt_history = [{"role": "system", "content": sys_in}] + chatgpt_history
    for i, (output, chatgpt_history) in enumerate(prompt_chatgpt(sys_in, user_input, chatgpt_history, 
                                                                 temperature=temperature, max_tokens=max_tokens,
                                                                 model_name=model_name)):
        item = {"role": "assistant", "content": output}
        if i == 0:
            chatgpt_history += [item]
        else:
            chatgpt_history = chatgpt_history[:-1] + [item]
        yield convert_chatgpt_history(chatgpt_history[1:]), disable_btn, disable_btn, disable_btn

    logger.info(chatgpt_history)
    yield convert_chatgpt_history(chatgpt_history[1:]), enable_btn, enable_btn, enable_btn


with gr.Blocks(title='Local OpenAI Chatbot') as demo:
    gr.Markdown('''<h1><center>Local OpenAI Chatbot</center></h1>''')

    enable_btn = gr.Button.update(interactive=True)
    disable_btn = gr.Button.update(interactive=False)
    no_change_btn = gr.Button.update()

    model_name = gr.Dropdown(['gpt-3.5-turbo', 'gpt-4'], label='Models', value='gpt-3.5-turbo')
    system_input = gr.Textbox(placeholder="e.g., You are a helpful assistant.", max_lines=500, label='System')
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label='User Input')
    with gr.Row():
        temperature = gr.Slider(0, 1, value=0.7, label='Temperature')
        max_tokens = gr.Number(value=256, label='Max Tokens') # max_new_tokens
    
    with gr.Row():
        btn_send = gr.Button(value="Submit", variant="primary", interactive=True)    
        # btn_stop = gr.Button(value='Stop', interactive=True)
        btn_regenerate = gr.Button(value='Regenerate', interactive=True)
        
    btn_clear = gr.Button(value="Clear", interactive=True)
    btn_list = [btn_send, btn_clear, btn_regenerate]
    
    def user(user_message, history):
        return "", history + [[user_message, None]]
    
    msg.submit(user, [msg, chatbot], [msg, chatbot]).then(
        bot, [model_name, system_input, chatbot, temperature, max_tokens], [chatbot] + btn_list
    )
    btn_send.click(user, [msg, chatbot], [msg, chatbot]).then(
        bot, [model_name, system_input, chatbot, temperature, max_tokens], [chatbot] + btn_list
    )
    btn_regenerate.click(user, [msg, chatbot], [msg, chatbot]).then(
        regenerate, [model_name, system_input, chatbot, temperature, max_tokens], [chatbot] + btn_list
    )
    btn_clear.click(lambda: None, None, chatbot)
    

if __name__ == "__main__":
    openai.api_key = os.environ['OPENAI_API_KEY']
    logger = init_logger('logs/chatbot.log')
    demo.queue(concurrency_count=3).launch()