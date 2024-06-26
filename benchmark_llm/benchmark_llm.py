'''
Benchmark GPT-3.5/4 on asynchow or baseline datasets
'''
import json
import numpy as np
import random
from datetime import timedelta
import re
import pickle
from copy import deepcopy
from openai import AsyncAzureOpenAI
import asyncio
from utils.utils import *
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from cohere import AsyncClient
from huggingface_hub import InferenceClient


async def prompt_model_for_all_templates(model_name,
                                         benchmark_dic,
                                         client=None,
                                         model=None,
                                         tokenizer=None,
                                         batch=30):
    random.seed(2024)
    sampled_idxs = random.sample([_ for _ in range(len(benchmark_dic['titles']))], 100)
    response_dic = dict()
    for key in benchmark_dic['prompts'].keys():
        prompts = [benchmark_dic['prompts'][key]['vanilla_prompts'][i] for i in sampled_idxs]
        response_dic[key] = list()
        responses = await benchmark_model(model_name=model_name,
                                          prompts=prompts,
                                          client=client,
                                          model=model,
                                          tokenizer=tokenizer,
                                          batch_size=batch)
        response_dic[key] = responses

    return response_dic


async def prompt_model_for_combined_templates(model_name,
                                              best_prompt,
                                              benchmark_dic,
                                              client=None,
                                              model=None,
                                              tokenizer=None,
                                              batch=30):
    '''
    Prompt model for combined (economic) templates
        Parameters:
            model: model to prompt
            benchmark_dic: benchmark dictionary
            best_prompt: best prompt template
            batch: batch size
        Returns:
            response_list: list of responses
    '''
    random.seed(2024)
    sampled_idxs = random.sample([_ for _ in range(len(benchmark_dic['titles']))], 100)
    prompts = [benchmark_dic['prompts'][best_prompt]['vanilla_combined_prompts'][i] for i in sampled_idxs]
    responses = await benchmark_model(model_name=model_name,
                                      prompts=prompts,
                                      model=model,
                                      client=client,
                                      tokenizer=tokenizer,
                                      batch_size=batch)

    return responses


def generate_nshot_prompt_graph(benchmark_dic,
                                best_template,
                                instruction,
                                nshot_idx,
                                cot,
                                graph,
                                combined=''):
    '''
    Generate n-shot prompt graph
        Parameters:
            benchmark_dic: benchmark dictionary
            best_template: best template
            instruction: instruction
            nshot_idx: n-shot idx
            cot: cot or not
            graph: graph
            combined: combined or not
        Returns:
            nshot_instr: n-shot instruction
    '''
    if cot:
        prompts = [benchmark_dic['prompts'][best_template][f'{graph}_cot{combined}_prompts'][i] for i in nshot_idx]
        nshot_instr = instruction['few_shot_cot']
        for i in range(len(prompts)):
            nshot_instr = nshot_instr.replace(f'[PROMPT{i}]',prompts[i])
    else:
        prompts = [benchmark_dic['prompts'][best_template][f'{graph}{combined}_prompts'][i] for i in nshot_idx]
        nshot_instr = instruction['few_shot']
        for i in range(len(prompts)):
            nshot_instr = nshot_instr.replace(f'[PROMPT{i}]',prompts[i])
    
    return nshot_instr


async def prompt_model_graph(model_name,
                            benchmark_dic,
                            sampled_idxs,
                            nshot_idx,
                            nshot_template_dic,
                            best_prompt,
                            cot,
                            nshot,
                            graph,
                            model=None,
                            tokenizer=None,
                            client=None,
                            batch=30):
    if cot:
        prompts = [benchmark_dic['prompts'][best_prompt][f'{graph}_cot_prompts'][i] for i in sampled_idxs]
    else:
        prompts = [benchmark_dic['prompts'][best_prompt][f'{graph}_prompts'][i] for i in sampled_idxs]
    if nshot:
        instruction = generate_nshot_prompt_graph(benchmark_dic,
                                            best_prompt,
                                            nshot_template_dic,
                                            nshot_idx,
                                            cot,
                                            graph)
        prompts = [instruction+prompt for prompt in prompts]

    responses = await benchmark_model(prompts=prompts,
                                      model_name=model_name,
                                      model=model,
                                      tokenizer=tokenizer,
                                      client=client,
                                      batch_size=batch)
    
    return responses


def calc_acc(responses,
             sampled_idxs,
             gold_timedelta):
    '''
    Calculate accuracy of cohere responses
        Parameters:
            responses: list of cohere responses
            sampled_idxs: list of sampled idxs
            gold_timedelta: list of gold timedelta
        Returns:
            res_dic: dictionary of results
    '''
    acc = 0
    total = 0
    res_dic = dict()
    time_delta_pred = list()
    bool_pred = list()
    invalids = list()
    eval_timedelta = [gold_timedelta[i] for i in sampled_idxs]
    if len(responses) != len(eval_timedelta):
        if type(responses[0]) == str:
            eval_responses = [responses[i] for i in sampled_idxs]
        elif type(responses[0]) == dict:
            eval_responses = [responses[i]['choices'][0]['message']['content'] for i in sampled_idxs]
        else:
            eval_responses = [responses[i].text if responses[i] else '' for i in sampled_idxs]
    else:
        if type(responses[0]) == str:
            eval_responses = responses
        elif type(responses[0]) == dict:
            eval_responses = [responses[i]['choices'][0]['message']['content'] for i in range(len(responses))]
        else:
            eval_responses = [responses[i].text if responses[i] else '' for i in range(len(responses))]
    for i, response in enumerate(eval_responses):
        if not response:
            time_delta_pred.append(None)
            bool_pred.append(False)
            invalids.append(i)
            total += 1
            continue
        timedelta_ans, is_correct = measure_perf(response, eval_timedelta[i])
        if is_correct:
            acc += 1
        time_delta_pred.append(timedelta_ans)
        bool_pred.append(is_correct)

    res_dic['time_delta_pred'] = time_delta_pred
    res_dic['bool_pred'] = bool_pred
    res_dic['acc'] = acc/total
    res_dic['invalids'] = invalids
    return res_dic


def eval_different_templates(response_dic,
                             benchmark_dic):
    '''
    Evaluate different templates
        Parameters:
            response_dic: dictionary of responses
            benchmark_dic: benchmark dictionary
        Returns:
            res_dic: dictionary of results
    '''
    random.seed(2024)
    sampled_idxs = random.sample([_ for _ in range(len(benchmark_dic['titles']))], 100)
    res_dic = dict()
    for key in benchmark_dic['prompts'].keys():
        res_dic[key] = dict()
        timedelta_pred, bool_pred, acc, invalids = calc_acc(response_dic[key], [benchmark_dic['task_time'][i] for i in sampled_idxs])
        res_dic[key]['acc'] = acc
        res_dic[key]['timedelta_pred'] = timedelta_pred
        res_dic[key]['bool_pred'] = bool_pred
        res_dic[key]['invalids'] = invalids
    templates, accs = list(), list()

    for template, val in res_dic.items():
        templates.append(template)
        accs.append(val['acc'])
    res_dic['best_template'] = templates[np.argmax(accs)]
    res_dic['worst_prompt'] = templates[np.argmin(accs)]
    return res_dic


def text_to_number_updated(sentence):
    # Updated mapping of number words to their numerical equivalents
    num_words = {
        "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, 'a': 1,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
        "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
        "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
        "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
        "ninety": 90, "hundred": 100, "thousand": 1000, "million": 1000000
    }

    # Helper function to convert a textual number expression to a number
    def text_number_to_num(text_num):
        parts = text_num.split()
        if 'and' in parts:
            parts.remove('and')

        total = 0
        current = 0

        for part in parts:
            if part in num_words:
                scale = num_words[part]
                if scale > 100:
                    current *= scale
                    total += current
                    current = 0
                elif scale == 100:
                    current *= scale
                else:
                    if current == 0:
                        current = scale
                    else:
                        current += scale
            else:
                # In case of numbers like 'forty-five'
                nums = part.split('-')
                for num in nums:
                    current += num_words.get(num, 0)

        return total + current

    # Regular expression pattern for matching text number expressions
    num_pattern = re.compile(r'\b(?:[a-zA-Z]+(?:-)?)+\b')

    # Find all matches
    matches = re.findall(num_pattern, sentence)

    # Process each match
    captured_patterns = {}
    for match in matches:
        number = text_number_to_num(match)
        if number > 0:
            captured_patterns[match] = number
            sentence = sentence.replace(match, str(number), 1)

    return sentence, captured_patterns


def measure_perf(response,
                 gold_timedelta):
    text_num_set = set(["an", "one", "two", "three", "four", "five", 'a',
        "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen",
        "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
        "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty",
        "ninety", "hundred", "thousand", "million", "half"])
    try:
        # if a model follows instruction
        # answer should be in double quotes, and either the first or the last one is the answer
        potential_answers = [re.findall(r'".*"', response)[0].lower(), re.findall(r'".*"', response)[-1].lower()]
        for i, answer in enumerate(potential_answers):
            if re.findall(r'\b\w+ and a half', answer):
                pattern = re.findall(r'\b\w+ and a half', answer)[0]
                prec_word = re.findall(r'\b\w+', pattern)[0]
                if prec_word not in text_num_set:
                    answer = answer.replace(pattern, f'and half {prec_word}')

                answer = answer.replace('half a ', '0.5')
                answer = text_to_number_updated(answer)[0].replace(' and half', '.5')
                potential_answers[i] = answer
    except Exception as e:
        # if a model does not follow instruction
        # try to get response after 'is'
        answer = response.split('is ')[-1].lower()
        if re.findall(r'\b\w+ and a half', answer):
            pattern = re.findall(r'\b\w+ and a half', answer)[0]
            prec_word = re.findall(r'\b\w+', pattern)[0]
            if prec_word not in text_num_set:
                answer = answer.replace(pattern, f'and half {prec_word}')

            answer = answer.replace('half a ', '0.5')
            answer = text_to_number_updated(answer)[0].replace(' and half', '.5')
            potential_answers = [answer]
    for answer in potential_answers:
        if '=' in answer:
            answer = answer.split('=')[-1]
        if ' or ' in answer:
            answer = answer.split(' or ')[-1]
        if '(' in answer:
            answer = answer.split('(')[0]
        timedelta_ans = [timedelta(), timedelta()]
        if ' to ' in answer:
            return [timedelta(), timedelta()], False
        try:
            time_spans = re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:min|minute|minutes|hr|hour|hours|sec|second|seconds|week|weeks|day|days|month|months|year|years|s|h|m|d|w)', answer)
            for time_span in time_spans:
                time = re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?', time_span)[-1].replace(',','')
                unit = re.findall(r'\b[a-z]+', time_span)[-1].strip()
                if unit in ['year', 'years', 'y']:
                    delta = [timedelta(days=float(time)*365), timedelta(days=float(time)*366)]
                elif unit in ['month', 'months', 'm']:
                    # define a loose range for month
                    # match other units in same format
                    delta = [timedelta(days=float(time)*28), timedelta(days=float(time)*31)]
                elif unit in ['week', 'weeks', 'w']:
                    delta = [timedelta(weeks=float(time)), timedelta(weeks=float(time))]
                elif unit in ['day', 'days', 'd']:
                    delta = [timedelta(days=float(time)), timedelta(days=float(time))]
                elif unit in ['hour', 'hours', 'h']:
                    delta = [timedelta(hours=float(time)), timedelta(hours=float(time))]
                elif unit in ['minute', 'min', 'minutes', 'mins']:
                    delta = [timedelta(minutes=float(time)), timedelta(minutes=float(time))]
                elif unit in ['second', 'sec', 'seconds', 'secs']:
                    delta = [timedelta(seconds=float(time)), timedelta(seconds=float(time))]
                else:
                    raise ValueError(f'unit not found: {time_span}')
                timedelta_ans[0] += delta[0]
                timedelta_ans[1] += delta[1]

            if gold_timedelta[0] <= timedelta_ans[0] <= gold_timedelta[1]:
                return timedelta_ans, True
        except Exception:
            continue
        if gold_timedelta[0] <= timedelta_ans[1] <= gold_timedelta[1]:
            return timedelta_ans, True
        return timedelta_ans, False

    return [timedelta(), timedelta()], False


def generate_nshot_prompt(benchmark_dic,
                          best_template,
                          instruction,
                          nshot_idx,
                          cot,
                          combined=''):
    if cot:
        prompts = [benchmark_dic['prompts'][best_template][f'cot{combined}_prompts'][i] for i in nshot_idx]
        nshot_instr = instruction['few_shot_cot']
        for i in range(len(prompts)):
            nshot_instr = nshot_instr.replace(f'[PROMPT{i}]',prompts[i])
    else:
        prompts = [benchmark_dic['prompts'][best_template][f'vanilla{combined}_prompts'][i] for i in nshot_idx]
        nshot_instr = instruction['few_shot']
        for i in range(len(prompts)):
            nshot_instr = nshot_instr.replace(f'[PROMPT{i}]',prompts[i])
    
    return nshot_instr


async def prompt_model_nshot_cot(model_name,
                                benchmark_dic,
                                nshot_idx,
                                nshot_template_dic,
                                best_prompt,
                                cot,
                                nshot,
                                combined='',
                                model=None,
                                tokenizer=None,
                                client=None,
                                batch=30):

    if cot:
        prompts = [benchmark_dic['prompts'][best_prompt][f'cot{combined}_prompts'][i] for i in range(len(benchmark_dic['prompts'][best_prompt][f'cot{combined}_prompts'])) if i not in nshot_idx]
    else:
        prompts = [benchmark_dic['prompts'][best_prompt][f'vanilla{combined}_prompts'][i] for i in range(len(benchmark_dic['prompts'][best_prompt][f'vanilla{combined}_prompts'])) if i not in nshot_idx]
    
    if nshot:
        instruction = generate_nshot_prompt(benchmark_dic,
                                            best_prompt,
                                            nshot_template_dic,
                                            nshot_idx,
                                            cot)
        prompts = [instruction+prompt for prompt in prompts]

    responses = await benchmark_model(prompts=prompts,
                                        model_name=model_name,
                                        model=model,
                                        tokenizer=tokenizer,
                                        client=client,
                                        batch_size=batch)

    return responses


async def prompt_model_graph_full_res(model_name,
                                      benchmark_dic,
                                      nshot_idx,
                                      nshot_template_dic,
                                      best_prompt,
                                      cot,
                                      nshot,
                                      graph,
                                      model=None,
                                      tokenizer=None,
                                      client=None,
                                      batch=30):
    response_list = list()
    exclude_idxs = []
    if nshot:
        exclude_idxs = deepcopy(nshot_idx)
    if cot:
        prompts = [benchmark_dic['prompts'][best_prompt][f'{graph}_cot_prompts'][i] if i not in exclude_idxs else '' for i in range(len(benchmark_dic['prompts'][best_prompt][f'{graph}_cot_prompts']))]
    else:
        prompts = [benchmark_dic['prompts'][best_prompt][f'{graph}_prompts'][i] if i not in exclude_idxs else '' for i in range(len(benchmark_dic['prompts'][best_prompt][f'{graph}_prompts']))]
    if nshot:
        instruction = generate_nshot_prompt_graph(benchmark_dic,
                                            best_prompt,
                                            nshot_template_dic,
                                            nshot_idx,
                                            cot,
                                            graph)
        prompts = [instruction+prompt for prompt in prompts]

    start_idx = 0
    while start_idx<len(prompts):
        responses = await generate_from_openai_chat_completion(prompts[start_idx:start_idx+batch],model=model,system_prompt='', temperature=0.0, n_choices=1)
        start_idx+=batch
        response_list.extend(responses)

    return response_list


def generate_nshot_prompt_bag(benchmark_dic,
                              best_template,
                              best_graph,
                              instruction,
                              nshot_idx,
                              combined=''):
    '''
    Generate n-shot prompt for BaG
        Parameters:
            benchmark_dic: benchmark dictionary
            best_template: best template
            instruction: instruction
            nshot_idx: n-shot idx
            cot: cot or not
            combined: combined or not
        Returns:
            nshot_instr: n-shot instruction

    '''
    prompts = [benchmark_dic['prompts'][best_template][f'bag_cot{combined}_prompts'][i] for i in nshot_idx]
    nshot_instr = instruction['few_shot_cot']
    for i in range(len(prompts)):
        nshot_instr = nshot_instr.replace(f'[PROMPT{i}]',prompts[i])
    
    return nshot_instr


async def prompt_model_bag_full_res(model_name,
                                    benchmark_dic,
                                    nshot_idx,
                                    nshot_template_dic,
                                    best_prompt,
                                    nshot,
                                    best_graph,
                                    combined='',
                                    model=None,
                                    tokenizer=None,
                                    client=None,
                                    batch=30):
    '''
    Prompt model for BaG
        Parameters:
            model: model to prompt
            benchmark_dic: benchmark dictionary
            nshot_idx: n-shot idx
            nshot_template_dic: n-shot template dictionary
            best_prompt: best prompt template
            nshot: n-shot or not
            combined: combined or not
            batch: batch size
        Returns:
            response_list: list of responses
    '''
    exclude_idxs = deepcopy(nshot_idx)
    prompts = [benchmark_dic['prompts'][best_prompt][f'bag_cot{combined}_prompts'][i] if i not in exclude_idxs else '' for i in range(len(benchmark_dic['prompts'][best_prompt][f'bag_cot{combined}_prompts']))]
    if nshot:
        instruction = generate_nshot_prompt_bag(benchmark_dic,
                                                best_prompt,
                                                best_graph,
                                                nshot_template_dic,
                                                nshot_idx)
        prompts = [instruction+prompt for prompt in prompts]

    responses = await benchmark_model(prompts=prompts,
                                      model=model,
                                      model_name=model_name,
                                      tokenizer=tokenizer,
                                      client=client,
                                      batch_size=batch)

    return responses


async def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',
                        type=str,
                        default=None,
                        required=True,
                        help='model name')
    parser.add_argument('--benchmark_path',
                        type=str,
                        default=None,
                        required=True,
                        help='Path to benchmark data')
    parser.add_argument('--api_key',
                        type=str,
                        default=None,
                        required=False,
                        help='OpenAI/huggingface API key')
    parser.add_argument('--batch',
                        type=int,
                        default=30,
                        required=False,
                        help='batch size')
    parser.add_argument('--task',
                        type=str,
                        default='vary_prompt_template',
                        required=False,
                        help='task name')
    parser.add_argument('--save_path',
                        type=str,
                        default=None,
                        required=False,
                        help='path to save results')
    parser.add_argument('--best_prompt_template',
                        type=str,
                        default=None,
                        required=False,
                        help='best prompt template')
    parser.add_argument('--api_version',
                        type=str,
                        default=None,
                        required=False,
                        help='OpenAI Azure API version')
    parser.add_argument('--azure_endpoint',
                        type=str,
                        default=None,
                        required=False,
                        help='Azure endpoint')
    parser.add_argument('--best_graph',
                        type=str,
                        default=None,
                        required=False,
                        help='best graph')
    args = parser.parse_args()

    # Load benchmark data
    benchmark_dic = json.load(open(args.benchmark_path, 'r'))
    # Load model
    if 'mistral' in args.model_name:
        client = None
        model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", device_map='auto')
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
        tokenizer.pad_token = tokenizer.eos_token
    elif 'gpt' in args.model_name:
        client = AsyncAzureOpenAI(api_version=args.api_version,
                                  api_key=args.api_key,
                                  azure_endpoint=args.azure_endpoint)
        model = None
        tokenizer = None
    elif 'command' in args.model_name.lower():
        client = AsyncClient(args.api_key)
        model = None
        tokenizer = None
    elif 'llama' in args.model_name.lower():
        client = InferenceClient(model=args.model_name,
                                 token=args.api_key)
        model = None
        tokenizer = None
    
    if args.task == 'vary_prompt_template':
        # Vary prompt template
        response_dic = await prompt_model_for_all_templates(model_name=args.model_name,
                                                            benchmark_dic=benchmark_dic,
                                                            client=client,
                                                            model=model,
                                                            tokenizer=tokenizer,
                                                            batch=args.batch)
        # Evaluate different templates
        res_dic = eval_different_templates(response_dic=response_dic,
                                           benchmark_dic=benchmark_dic)
        res_dic['responses'] = response_dic
        os.mkdir(args.save_path)
        with open(f'{args.save_path}/{args.task}_{args.model_name}.pkl', 'wb') as f:
            pickle.dump(res_dic, f)
    elif args.task == 'vary_economic_prompt':
        # Vary economic prompt
        response_dic = await prompt_model_for_combined_templates(model_name=args.model_name,
                                                                 best_prompt=args.best_prompt_template,
                                                                 benchmark_dic=benchmark_dic,
                                                                 client=client,
                                                                 model=model,
                                                                 tokenizer=tokenizer,
                                                                 batch=args.batch)
        # Evaluate different templates
        res_dic = calc_acc(response_dic=response_dic,
                            sampled_idxs=sampled_idxs,
                            gold_timedelta=[benchmark_dic['task_time'][i] for i in sampled_idxs])
        res_dic['responses'] = response_dic
        os.mkdir(args.save_path)
        with open(f'{args.save_path}/{args.task}_{args.model_name}.pkl', 'wb') as f:
            pickle.dump(res_dic, f)
    elif args.task == 'vary_graph':
        res_dic = dict()
        nshot_idx = benchmark_dic['nshot_instructions']['idxs']
        random.seed(0)
        sampled_idxs = random.sample([_ for _ in range(len(benchmark_dic['titles'])) if _ not in nshot_idx], 100)
        nshot_template_dic = benchmark_dic['nshot_instructions']
        response_dic = await prompt_model_graph(model_name=args.model_name,
                                                benchmark_dic=benchmark_dic,
                                                sampled_idxs=sampled_idxs,
                                                nshot_idx=nshot_idx,
                                                nshot_template_dic=nshot_template_dic,
                                                best_prompt=args.best_prompt_template,
                                                cot=False,
                                                nshot=True,
                                                graph='graph',
                                                model=model,
                                                tokenizer=tokenizer,
                                                client=client,
                                                batch=args.batch)
        # Evaluate different graphs
        for graph in ['adjacency_list', 'adjacency_matrix', 'csr', 'edge_list']:
            res_dic[graph] = calc_acc(response_dic=response_dic,
                                      sampled_idxs=sampled_idxs,
                                      gold_timedelta=[benchmark_dic['task_time'][i] for i in sampled_idxs])
        res_dic['responses'] = response_dic
        os.mkdir(args.save_path)
        with open(f'{args.save_path}/{args.task}_{args.model_name}.pkl', 'wb') as f:
            pickle.dump(res_dic, f)
    elif args.task == 'vary_shot_cot':
        response_dic = dict()
        for cot in [True, False]:
            for nshot in [True, False]:
                response_dic[f'cot_{cot}_nshot_{nshot}'] = await prompt_model_nshot_cot(model_name=args.model_name,
                                                                                        benchmark_dic=benchmark_dic,
                                                                                        nshot_idx=benchmark_dic['nshot_instructions']['idxs'],
                                                                                        nshot_template_dic=benchmark_dic['nshot_instructions'],
                                                                                        best_prompt=args.best_prompt_template,
                                                                                        cot=cot,
                                                                                        nshot=nshot,
                                                                                        model=model,
                                                                                        tokenizer=tokenizer,
                                                                                        client=client,
                                                                                        batch=args.batch)
        res_dic = dict()
        for key, val in response_dic.items():
            res_dic[key] = calc_acc(response_dic=val,
                                    sampled_idxs=[i for i in range(len(benchmark_dic['titles'])) if i not in benchmark_dic['nshot_instructions']['idxs']],
                                    gold_timedelta=benchmark_dic)
        res_dic['responses'] = response_dic
        os.mkdir(args.save_path)
        with open(f'{args.save_path}/{args.task}_{args.model_name}.pkl', 'wb') as f:
            pickle.dump(res_dic, f)
    elif args.task == 'explicit_graph':
        response_dic = await prompt_model_graph_full_res(model_name=args.model_name,
                                                            benchmark_dic=benchmark_dic,
                                                            nshot_idx=benchmark_dic['nshot_instructions']['idxs'],
                                                            nshot_template_dic=benchmark_dic['nshot_instructions'],
                                                            best_prompt=args.best_prompt_template,
                                                            cot=False,
                                                            nshot=True,
                                                            graph=args.best_graph,
                                                            model=model,
                                                            tokenizer=tokenizer,
                                                            client=client,
                                                            batch=args.batch)
        res_dic = calc_acc(response_dic=response_dic,
                            sampled_idxs=sampled_idxs,
                            gold_timedelta=[benchmark_dic['task_time'][i] for i in sampled_idxs])
        res_dic['responses'] = response_dic
        os.mkdir(args.save_path)
        with open(f'{args.save_path}/{args.task}_{args.model_name}.pkl', 'wb') as f:
            pickle.dump(res_dic, f)
    elif args.task == 'bag':
        response_dic = await prompt_model_bag_full_res(model_name=args.model_name,
                                                        benchmark_dic=benchmark_dic,
                                                        nshot_idx=benchmark_dic['nshot_instructions']['idxs'],
                                                        nshot_template_dic=benchmark_dic['bag']['templates'][args.best_graph],
                                                        best_prompt=args.best_prompt_template,
                                                        best_graph=args.best_graph,
                                                        nshot=True,
                                                        combined='',
                                                        model=model,
                                                        tokenizer=tokenizer,
                                                        client=client,
                                                        batch=args.batch)
        res_dic = calc_acc(response_dic=response_dic,
                            sampled_idxs=sampled_idxs,
                            gold_timedelta=[benchmark_dic['task_time'][i] for i in sampled_idxs])
        res_dic['responses'] = response_dic
        os.mkdir(args.save_path)
        with open(f'{args.save_path}/{args.task}_{args.model_name}.pkl', 'wb') as f:
            pickle.dump(res_dic, f)
    else:
        raise ValueError(f'Invalid task: {args.task}')

if __name__ == '__main__':
    asyncio.run(main())
