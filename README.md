# Local OpenAI Playground


An easy-to-use local entry for accessing OpenAI and (deployed) HuggingFace LLMs with OpenAI-style API. Support basic chatbot functions with `gradio`.

## Examples

![chatbot](/images/chatbot.png)

![demo](/images/demo.jpg)


## How to run

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Run the server 

For InstructGPT or GPT-3, run

```bash
python3 app.py
```

For ChatGPT, GPT-4, run

```bash
python3 app_chatbot.py
```

For your own HuggingFace models (see [basaran](https://github.com/hyperonym/basaran)), run

```bash
python3 app_chatbot_hf.py
```

3. Open the browser and go to `YOUR URL`

4. Input your **OpenAI API key** (or run `export OPENAI_API_KEY=xxx` beforehand) and everything and click `SUBMIT` or press ENTER (For ChatBot)

5. Enjoy your local OpenAI Playground!


## How to flag

You can flag interesting/error cases by clicking the `Flag` button. The flagged cases will be stored in `flagged/some_file.csv` and you can check them later.

