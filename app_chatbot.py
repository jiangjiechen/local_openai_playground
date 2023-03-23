import gradio as gr
import random
import time
import openai
import os
import sys
from utils import init_logger


def clean_br(history):
    for x in history:
        x['content'] = x['content'].replace('<br>', '')
    return history


def prompt_chatgpt(system_input, user_input, history=[], model_name='gpt-3.5-turbo'):
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

    for _ in range(3):
        try:
            history = clean_br(history)
            completion = openai.ChatCompletion.create(
                model=model_name,
                messages=history,
            )
            break
        except:
            error = sys.exc_info()[0]
            print("API error:", error)
            time.sleep(1)

    assistant_output = completion['choices'][0]['message']['content']
    history.append({"role": "assistant", "content": assistant_output})
    # total_tokens = completion['usage']['total_tokens']

    return assistant_output, history


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


with gr.Blocks() as demo:
    api_key = gr.Textbox(placeholder='Your OpenAI API KEY here', type='password', show_progress=False, label='Your OpenAI API KEY')
    system_input = gr.Textbox(placeholder="e.g., You are a helpful assistant.", max_lines=500, show_progress=False, label='System')
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label='User Input')
    clear = gr.Button("Clear")

    def user(user_message, history):
        return "", history + [[user_message, None]]

    def bot(key, sys_in, history):
        # bot_message = random.choice(["Yes", "No"])
        openai.api_key = key
        user_input = history[-1][0]

        chatgpt_history = convert_chatgpt_history(history[:-1], backward=True)
        
        chatgpt_history = [{"role": "system", "content": sys_in}] + chatgpt_history
        _, chatgpt_history = prompt_chatgpt(sys_in, user_input, chatgpt_history)
        logger.info(chatgpt_history)
        
        history = convert_chatgpt_history(chatgpt_history[1:])
        return history
    
    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, [api_key, system_input, chatbot], [chatbot]
    )
    clear.click(lambda: None, None, chatbot, queue=False)


if __name__ == "__main__":
    logger = init_logger('logs/chatbot.log')
    demo.launch()