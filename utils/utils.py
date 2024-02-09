'''
File for utility functions
'''
import json
import random
from collections import Counter
import asyncio
import time
import re
import torch
import logging
from openai import (
    APIConnectionError,
    APIError,
    BadRequestError,
    RateLimitError,
    Timeout,
)

ERROR_ERRORS_TO_MESSAGES = {
    BadRequestError: "OpenAI API Invalid Request: Prompt was filtered",
    RateLimitError: "OpenAI API rate limit exceeded.",
    APIConnectionError: "OpenAI API Connection Error: Error Communicating with OpenAI",  # noqa E501
    Timeout: "OpenAI APITimeout Error: OpenAI Timeout",
    APIError: "OpenAI API error: {e}",
}


def check_safety(response: dict):
    '''
    Check if response is safe
        Parameters:
            response (dict): response from GPT
            safe_dic (dict): dictionary of safe categories
        Returns:
            is_safe (bool): whether response is safe
    '''

    safe_dic = {'hate': {'filtered': False, 'severity': 'safe'},
                'self_harm': {'filtered': False, 'severity': 'safe'},
                'sexual': {'filtered': False, 'severity': 'safe'},
                'violence': {'filtered': False, 'severity': 'safe'}}
    try:
        if response['prompt_filter_results'][0]['content_filter_results'] != safe_dic:
            return False
        for choice in response['choices']:
            if choice['content_filter_results'] != safe_dic:
                return False
        return True
    except Exception:
        return False


async def _throttled_openai_chat_completion_acreate(
    client,
    model: str,
    messages: list,
    temperature: float,
    top_p=1.0,
    n=3,
    frequency_penalty=0,
    presence_penalty=0,
    stop=None,
    initial_delay: float = 10,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 4,

):
    if not messages:
        return {}
    num_retries = 0
    delay = initial_delay
    for _ in range(max_retries):
        try:
            return await client.chat.completions.create(
                                                        model=model,
                                                        messages=messages,
                                                        temperature=temperature,
                                                        top_p=top_p,
                                                        n=n,
                                                        frequency_penalty=frequency_penalty,
                                                        presence_penalty=presence_penalty,
                                                        stop=stop
                                                    )
        except Exception as e:
            if isinstance(e, APIError):
                logging.warning(ERROR_ERRORS_TO_MESSAGES[type(e)].format(e=e))
            elif isinstance(e, BadRequestError):
                logging.warning(ERROR_ERRORS_TO_MESSAGES[type(e)])
                return {
                    "choices": [
                        {
                            "message": {
                                "content": "Invalid Request: Prompt was filtered"
                            }
                        }
                    ]
                }
            else:

                logging.warning(e)

            num_retries += 1
            print("num_retries=", num_retries)

            # Check if max retries has been reached
            if num_retries > max_retries:
                raise Exception(
                    f"Maximum number of retries ({max_retries}) exceeded."
                )

            # Increment the delay
            delay *= exponential_base * (1 + jitter * random.random())
            print("new delay=", delay)
            await asyncio.sleep(delay)
        return {"choices": [{"message": {"content": ""}}]}


async def generate_from_openai_chat_completion(
        client,
        task_prompts: list[str],
        nshot_prompt: str = '',
        model: str = 'gpt-4',
        system_prompt: str = 'You are a helpful plan organizer.',
        n_choices: int = 3,
        temperature: float = 1.0,
) -> list:
    if system_prompt:
        messages = list()
        for task_prompt in task_prompts:
            if not task_prompt:
                messages.append({})
                continue
            messages.append([{
                              'role': 'system',
                              'content': system_prompt
                             },
                             {
                              "role": "user",
                              "content": nshot_prompt+task_prompt
                             }])
    else:
        messages = list()
        for task_prompt in task_prompts:
            # skip empty prompts
            if not task_prompt:
                messages.append({})
                continue
            messages.append([{
                              "role": "user",
                              "content": nshot_prompt+task_prompt,
                              }])
    async_responses = [
        _throttled_openai_chat_completion_acreate(
            client,
            model=model,
            messages=message,
            temperature=temperature,
            top_p=1,
            n=n_choices,
            frequency_penalty=0,
            stop=None
        ) for message in messages
    ]
    responses = await asyncio.gather(*async_responses)
    # Note: will never be none because it's set, but mypy doesn't know that.
    # await openai.aiosession.get().close()  # type: ignore
    all_responses = [json.loads(x.model_dump_json(indent=2)) if type(x) is not dict else x for x in responses]
    return all_responses


def can_reach(edges, start, end):
    '''
    Check if end node is reachable from start node
        Parameters:
            edges: list of edges
            start: start node
            end: end node
        Returns:
            reachable: boolean indicating whether end node is reachable from start node
    '''
    graph = {}
    for edge in edges:
        if edge[0] in graph:
            graph[edge[0]].append(edge[1])
        else:
            graph[edge[0]] = [edge[1]]

    # Using Depth First Search (DFS) to check if the end node is reachable
    def dfs(current, target, visited):
        if current == target:
            return True
        visited.add(current)
        for neighbor in graph.get(current, []):
            if neighbor not in visited:
                if dfs(neighbor, target, visited):
                    return True
        return False

    return dfs(start, end, set())


def get_all_dependencies(text):
    '''
    Get all dependencies from model outputs
        Parameters:
            text: string
        Returns:
            all_deps: set
    '''
    patterns = re.findall(r'(?:step)?\s*\d+"?\'?\s*->\s*"?\'?(?:step)?\s*\d+', text.lower())
    all_deps = set()
    if not patterns:
        return all_deps
    for pattern in patterns:
        prec, foll = pattern.split('->')
        prec_node = re.findall(r'\d+', prec)[0]
        foll_node = re.findall(r'\d+', foll)[0]
        # nodes = re.findall(r'\d+', dep)
        all_deps.add(tuple([prec_node, foll_node]))

    res_deps = deepcopy(all_deps)
    all_deps = list(all_deps)
    for i, dep in enumerate(all_deps):
        if can_reach(all_deps[:i]+all_deps[i+1:], dep[0], dep[1]):
            res_deps.remove(dep)
    return res_deps


def get_edge_list(contents):
    '''
    Get dependency responses and edge lists, and return the most frequent dependencies and its frequency
        Parameters:
            contents: list of response contents
        Returns:
            edge_list: list of edge lists
            most_freq_deps: tuple of most frequent dependencies
            count_freq: count frequency of the most frequent dependencies
    '''

    edge_list = list()
    for content in contents:
        try:
            edge_list.append(sorted(list(get_all_dependencies(content))))
        except Exception:
            continue

    try:
        most_freq_deps, count_freq = Counter([tuple(_) for _ in edge_list]).most_common(1)[0]
    except Exception:
        most_freq_deps, count_freq = tuple(), 0

    return edge_list, most_freq_deps, count_freq


def get_llama_response_sync(client,
                            task_prompt,
                            model):
    '''
    Query llama model synchronously
        Parameters:
            clients: client
            task_prompt: task prompt
            model: model name
        Returns:
            response: response from model
    '''
    for _ in range(3):
        try:
            response = client.text_generation(prompt=f'[INST]{task_prompt}[/INST]',
                                              do_sample=False,
                                              seed=0,
                                              max_new_tokens=4096)
            return response
        except Exception as e:
            time.sleep(60)
            print(e)
            print('sleep for 60 seconds')
    return ''


def batch_prompt_mistral(model,
                         tokenizer,
                         prompts):
    torch.cuda.manual_seed_all(2024)
    inputs = tokenizer([f'[INST]{prompt}[/INST]' for prompt in prompts], return_tensors="pt", padding=True).to('cuda')
    generate_ids = model.generate(**inputs, do_sample=False, max_length=4096, temperature=0.0)
    start_index = inputs.input_ids.shape[-1]
    outputs = tokenizer.batch_decode(generate_ids[:, start_index:],
                                     skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False,)
    return outputs


async def benchmark_model(model_name,
                          prompts,
                          client: None,
                          model=None,
                          tokenizer=None,
                          batch_size=1
                          ):
    '''
    Benchmark different models
        Parameters:
            model: model name
            prompts: list of prompts
        Returns:
            responses: list of responses
    '''
    if 'gpt' in model_name.lower():
        responses = []
        start_idx = 0
        while start_idx < len(prompts):
            responses.extend(await generate_from_openai_chat_completion(
                client,
                prompts[start_idx:start_idx+batch_size],
                model=model_name,
                system_prompt='',
                n_choices=1,
                temperature=0,
            ))
            start_idx += batch_size
    elif 'llama' in model_name.lower():
        responses = [get_llama_response_sync(client, prompt, model_name) for prompt in prompts]
    elif 'mistral' in model_name.lower():
        start_idx = 0
        responses = []
        while start_idx < len(prompts):
            responses.extend(batch_prompt_mistral(model, tokenizer, prompts[start_idx:start_idx+batch_size]))
            start_idx += batch_size
    else:
        raise NotImplementedError(f'the model {model_name} is not implemented')
    return responses
