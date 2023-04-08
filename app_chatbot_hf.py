import gradio as gr
import openai
from utils import init_logger


def clean_br(history):
    for x in history:
        x['content'] = x['content'].replace('<br>', '')
    return history


def beautify_output(output):
    return output.replace('[tab]', '    ').replace('[newline]', '\n').replace('</s>', '').strip()


def openai_history_to_prompt(history):
    # Support an example dialogue history like (ChatGLM style)
    # ${instruction}
    # [Round 1]
    # 问：${user_input}
    # 答：${assistant_output}
    # [Round 2]
    # ...
    prompt = history[0]['content'] + '\n' # system
    for i, hist in enumerate(history[1:-1]):
        if hist['role'] == 'user':
            prompt += f"[Round {int(i/2)}]\n问：{hist['content']}\n"
        else:
            prompt += f"[Round {int(i/2)}]\n答：{hist['content']}\n"
    prompt += f"[Round {int(len(history)/2)}]\n问：{history[-1]['content']}\n答："
    return prompt


def prompt_chatgpt(system_input, user_input, history=[], model_name='some-model', max_tokens=128):
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

    # history = clean_br(history)
    completion = openai.Completion.create(
        model=model_name,
        prompt=openai_history_to_prompt(history),
        max_tokens=max_tokens,
        temperature=0.7, # TODO
        stream=True,
        stop=['[Round', '.\n[']
    )
    yield from lookahead_for_stop(completion)


def lookahead_for_stop(completion):
    # completion: generator
    assistant_output = ''
    current_word = next(completion)['choices'][0]['text']
    while True:
        try:
            next_word = next(completion)['choices'][0]['text']
        except StopIteration:
            break
        
        if '[Round' in next_word:
            break
        elif current_word == "[" and next_word.startswith("Round"):
            current_word = ''
            break
        elif current_word.endswith("[") and next_word.startswith("Round"):
            current_word = current_word[:-1]
            break
        else:
            assistant_output += current_word
            yield beautify_output(assistant_output)
            current_word = next_word

    assistant_output += current_word
    yield beautify_output(assistant_output)


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
    yield history, disable_btn, disable_btn

    user_input = history[-1][0]
    chatgpt_history = convert_chatgpt_history(history[:-1], backward=True)
    
    output = ''
    chatgpt_history = [{"role": "system", "content": sys_in}] + chatgpt_history
    for i, output in enumerate(prompt_chatgpt(sys_in, user_input, chatgpt_history, model_name)):
        item = {"role": "assistant", "content": output}
        if i == 0:
            chatgpt_history += [item]
        else:
            chatgpt_history = chatgpt_history[:-1] + [item]
        # chatgpt_history = clean_br(chatgpt_history)
        yield convert_chatgpt_history(chatgpt_history[1:]), disable_btn, disable_btn
    
    logger.info(chatgpt_history)
    yield convert_chatgpt_history(chatgpt_history[1:]), enable_btn, enable_btn


with gr.Blocks(title='Local HuggingFace Chatbot') as demo:
    enable_btn = gr.Button.update(interactive=True)
    disable_btn = gr.Button.update(interactive=False)
    no_change_btn = gr.Button.update()

    model_name = gr.Dropdown(['Bloomz-7b1'], label='Models', value='Bloomz-7b1')
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
        bot, [model_name, system_input, chatbot], [chatbot] + btn_list
    )
    btn_send.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, [model_name, system_input, chatbot], [chatbot] + btn_list
    )
    btn_clear.click(lambda: None, None, chatbot, queue=False)


if __name__ == "__main__":
    openai.api_base = "http://127.0.0.1:34269/v1" # start your own hf model server
    openai.api_key = 'test' # random string to pass the check
    logger = init_logger('logs/chatbot.log')
    demo.launch(enable_queue=True, share=True)