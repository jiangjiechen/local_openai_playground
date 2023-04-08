import gradio as gr
import random
import time
import openai
import os
import sys
from utils import init_logger, GPT3_NAME_AND_COST


def clean_br(history):
    for x in history:
        x['content'] = x['content'].replace('<br>', '')
    return history


def prompt_chatgpt(system_input, user_input, history=[], model_name='gpt-3.5-turbo', max_tokens=128, stream=True):
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

    history = clean_br(history)
    completion = openai.ChatCompletion.create(
        model=model_name,
        messages=history,
        max_tokens=max_tokens,
        stream=stream,
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


def bot(model_name, sys_in, history):
    user_input = history[-1][0]

    chatgpt_history = convert_chatgpt_history(history[:-1], backward=True)
    
    chatgpt_history = [{"role": "system", "content": sys_in}] + chatgpt_history
    for i, (output, chatgpt_history) in enumerate(prompt_chatgpt(sys_in, user_input, chatgpt_history, model_name)):
        item = {"role": "assistant", "content": output}
        if i == 0:
            chatgpt_history += [item]
        else:
            chatgpt_history = chatgpt_history[:-1] + [item]
        yield convert_chatgpt_history(chatgpt_history[1:])


with gr.Blocks(title='Local OpenAI Chatbot') as demo:
    model_name = gr.Dropdown(['gpt-3.5-turbo', 'gpt-4'], label='Models', value='gpt-3.5-turbo')
    system_input = gr.Textbox(placeholder="e.g., You are a helpful assistant.", max_lines=500, show_progress=False, label='System')
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label='User Input')
    with gr.Row():
        btn_send = gr.Button(value="Submit", variant="primary", interactive=True)
        btn_clear = gr.Button(value="Clear", interactive=True)
        btn_list = [btn_send, btn_clear]

    def user(user_message, history):
        return "", history + [[user_message, None]]
    
    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, [model_name, system_input, chatbot], [chatbot]
    )
    btn_send.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, [model_name, system_input, chatbot], [chatbot]
    )
    btn_clear.click(lambda: None, None, chatbot, queue=False)


if __name__ == "__main__":
    openai.api_key = os.environ['OPENAI_API_KEY']
    logger = init_logger('logs/chatbot.log')
    demo.launch(enable_queue=True)