import json
import openai
import os
from huggingface_hub import InferenceClient

def queryLLM(prompt, config, model):
    global f_query

    with open(config) as jfile:
        jdata = json.load(jfile)
        response = None
        try:
            print(f"Querying {model}. Available models: {[m for m in f_query.keys()]}.")
            print()
            response = f_query[model](prompt, jdata[model])
        except Exception as inst:
            print(f"[error] Querying model {model} has failed. Error: {inst}")

    return response

def queryllama(prompt, jdata):
    client = InferenceClient(model="meta-llama/Llama-2-70b-chat-hf", token=jdata['key'])
    completion = client.text_generation(prompt, 
                                        max_new_tokens=2000,
                                        temperature=0.7)
    return completion

def queryllamacode(prompt, jdata):
    client = InferenceClient(model="codellama/CodeLlama-13b-hf", token=jdata['key'])
    completion = client.text_generation(prompt, 
                                        max_new_tokens=2000,
                                        temperature=0.7)
    return completion

def queryllamacodeinstruct(prompt, jdata):
    client = InferenceClient(model="codellama/CodeLlama-34b-Instruct-hf", token=jdata['key'])
    completion = client.text_generation(prompt, 
                                        max_new_tokens=2000,
                                        temperature=0.7)
    return completion

def querygpt(prompt, jdata):
    """
    https://platform.openai.com/docs/guides/gpt/chat-completions-api
    """
    openai.organization = jdata['organization']
    openai.api_key = jdata['key']
    model_name = jdata['name']

    completion = openai.ChatCompletion.create(
                                                model = model_name,
                                                messages = [
                                                    {'role': 'user', 'content': f'{prompt}'}
                                                ],
                                                temperature = 0.,
                                                )

    return completion['choices'][0]['message']['content']

def querygpt4local(prompt, jdata):
    """
    GPT-4 Azure endpoint required (stored locally)
    """
    openai.api_type = jdata['api_type']
    openai.api_base = jdata['api_base']
    openai.api_version = jdata['api_version']
    openai.api_key = jdata['key']

    message_text = [{"role":"system","content":""}, \
                    {"role":"user","content": prompt}]

    completion = openai.ChatCompletion.create(
                                                engine="gpt-4",
                                                messages = message_text,
                                                temperature=0.,
                                                max_tokens=1000,
                                                frequency_penalty=0,
                                                presence_penalty=0,
                                                stop=None
                                                )

    return completion['choices'][0]['message']['content']

f_query = {'gpt35':querygpt, 'gpt4':querygpt4local, 'gpt4local': querygpt4local, \
           'llama':queryllama, 'llamacode':queryllamacode, 'llamacodeinstruct':queryllamacodeinstruct}

def read_prompts(file_, tag):
    with open(file_, 'r') as f:
        s = f.read()
        start = f"<{tag}>"
        end = f"</{tag}>"
        prompt = s[s.index(start) + len(start): s.index(end)]
    assert len(prompt) > 0
    return prompt
