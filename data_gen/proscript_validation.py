'''
Validate dependency annotation in proscript
'''
import re
from utils.utils import *
import argparse
import random
import asyncio
import json
from openai import AsyncAzureOpenAI


def gen_proscript_dep_prompt(script):
    '''
    Generate prompt for validating dependency annotation in proscript
        Parameters:
            script: proscript
        Returns:
            prompt: prompt for validating dependency annotation in proscript
    '''
    title = script['scenario']
    steps = [step.split(': ')[1] for i, step in enumerate(script['flatten_input_for_edge_prediction'].split('; '))][:-2]
    prompt = f'Here are randomly ordered steps needed to \'{title}\'. '
    str_steps = '; '.join([f'step{ii+1}: {s}' for ii, s in enumerate(steps)])
    prompt += f'\n{str_steps}\n'
    prompt += 'Assume infinite resources are available and that steps should be parallelized where possible. For each step, does it logically need to follow others considering the nature of the task? Let’s think step by step then finally answer in dot language for all necessary constraints, each constraint per line in the format of "preceding step index" -> "following step index"'
    return prompt


def get_gold_edges_proscript(script,
                             step_num):
    '''
    Get gold edges from proscript
        Parameters:
            script: dict
        Returns:
            edges: set
    '''
    edges = set()
    for edge in script['gold_edges_for_prediction']:
        left, right = re.findall(r'\d+', edge)
        if int(left) in [step_num, step_num-1] or int(right) in [step_num, step_num-1]:
            continue
        edges.add((str(int(left)+1), str(int(right)+1)))

    return edges


def eval_proscript(responses,
                   proscript_test,
                   idxs: list):
    total = 0
    recall = 0
    prec = 0
    f1 = 0

    for i, response in enumerate(responses):
        step_num = len(proscript_test[idxs[i]]['flatten_input_for_edge_prediction'].split('; '))-1
        gold_deps = get_gold_edges_proscript(proscript_test[idxs[i]], step_num)
        contents = [choice['message']['content'] for choice in response['choices']]
        edge_list, pred_deps, freq_counts = get_edge_list(contents)
        # print(contents)
        if freq_counts < 4:
            continue
        total += 1
        intersec = len(gold_deps.intersection(set(pred_deps)))
        if intersec != len(gold_deps):
            print(i, gold_deps, pred_deps, freq_counts)
        if not gold_deps:
            recall += 1
        else:
            recall += intersec/len(gold_deps)
        prec += intersec/len(pred_deps)
        f1 += 2*intersec/(len(gold_deps)+len(pred_deps))
    if total:
        recall = recall/total
        prec = prec/total
        f1 = f1/total
    return recall, prec, f1


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--proscript_test_path',
                        type=str,
                        default=None,
                        required=True,
                        help='Path to proscript test data')
    parser.add_argument('--proscript_dev_path',
                        type=str,
                        default=None,
                        required=True,
                        help='Path to proscript dev data')
    parser.add_argument('--api_key',
                        type=str,
                        default=None,
                        required=True)
    parser.add_argument('--api_version',
                        type=str,
                        default=None,
                        required=True)
    parser.add_argument('--azure_endpoint',
                        type=str,
                        default=None,
                        required=True)
    args = parser.parse_args()

    client = AsyncAzureOpenAI(api_version=args.api_version,
                              api_key=args.api_key,
                              azure_endpoint=args.azure_endpoint)
    proscript = dict()
    with open(args.proscript_test_path, 'r') as f:
        proscript['test'] = json.load(f)
    with open(args.proscript_dev_path, 'r') as f:
        proscript['dev'] = json.load(f)

    no_context_idx = {'dev': list(), 'test': list()}
    for key, val in proscript.items():
        for i, data in enumerate(val):
            if data['events'][str(len(data['events'])-2)] == 'NONE':
                no_context_idx[key].append(i)

    proscript_val_dic = dict()
    for seed in range(3):
        for key, val in proscript.items():
            proscript_val_dic[seed][key] = dict()
            random.seed(seed)
            sampled_idx = random.sample(no_context_idx[key], 100)

            iter_idx = 0
            new_test_responses = list()
            while iter_idx < 100:
                prompts = [gen_proscript_dep_prompt(val[sampled_idx[i]]) for i in range(iter_idx, iter_idx+20)]
                responses = await generate_from_openai_chat_completion(client,
                                                                       prompts,
                                                                       '',
                                                                       system_prompt='You are ChatGPT',
                                                                       n_choices=5)
                iter_idx += 20
                new_test_responses.extend(responses)
            proscript_val_dic[seed][key]['responses'] = new_test_responses
            recall, prec, f1 = eval_proscript(new_test_responses,
                                              proscript[key],
                                              sampled_idx)
            proscript_val_dic[seed][key]['recall'] = recall
            proscript_val_dic[seed][key]['prec'] = prec
            proscript_val_dic[seed][key]['f1'] = f1

    with open('proscript_val.json', 'w') as f:
        json.dump(proscript_val_dic, f)

if __name__ == '__main__':
    asyncio.run(main())