# Local OpenAI Playground


Local entry for accessing OpenAI LLMs with OpenAI API.

![demo](/images/demo.jpg)

![chatbot](/images/chatbot.jpg)

## How to run

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Run the server 

For InstructGPT, Codex or GPT-3, run

```bash
python3 app.py
```

For ChatGPT, run

```bash
python3 app_chatbot.py
```

3. Open the browser and go to `YOUR URL`

4. Input your **OpenAI API key** (or run `export OPENAI_API_KEY=xxx`) and everything and click `SUBMIT` or press ENTER (For ChatBot)

5. Enjoy your local OpenAI Playground!


## How to flag

You can flag interesting/error cases by clicking the `Flag` button. The flagged cases will be stored in `flagged/some_file.csv` and you can check them later.

