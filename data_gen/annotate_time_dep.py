'''
Annotate time and step dependencies in the wikihow data
'''
from openai import AsyncAzureOpenAI
import json
import re
import argparse
from copy import deepcopy
from collections import Counter
from utils.utils import *
import asyncio
import os


def veri_nec(contents):
    '''
    Verify necessity
        Parameters:
            task_prompt: task prompt
            nec_nshot_prompt: n-shot prompt
            model: model name
            n_choices: number of choices
        Returns:
            is_necessary: boolean indicating whether the plan is necessary
    '''
    try:
        no_count = 0
        for content in contents:
            all_words = set(re.findall(r"[\w']+", content.lower()))
            if 'no' in all_words:
                no_count += 1

        if no_count > len(contents)//2:
            return True
        else:
            return False
    except Exception:
        return False


def get_time_dic_per_response(response,
                              step_num):
    '''
    Parse time description into a dictionary of time steps
        Parameters:
            time_description: string
            step_num: number of steps
        Returns:
            stepwise_time_dic: dictionary
            valid: boolean indicating whether the time annotation is valid (i.e. numerical time for each step)
    '''
    try:
        time_dic = json.loads(re.findall(r'\{[\s\S]*.*\}', response)[0])
        if len(time_dic) != step_num:
            return time_dic, False
        for val in time_dic.values():
            if not re.findall(r'\d+\s*(?:min|minute|minutes|hr|hour|hours|sec|second|seconds|week|weeks|day|days|month|months|year|years)', val):
                return time_dic, False
        return time_dic, True
    except Exception:
        try:
            valid = True
            lines = response.split('\n')
            time_dic = dict()
            for line in lines:
                step_idx = re.findall(r'\d+"?\s*:', line)
                if not step_idx:
                    continue
                step_idx = line.split(':')[0]
                time_expression = line.split(':')[-1]
                time = re.findall(r'\d+\s*(?:min|minute|minutes|hr|hour|hours|sec|second|seconds|week|weeks|day|days|month|months|year|years)', time_expression)
                if not time:
                    valid = False
                time_dic[step_idx] = time_expression
            if len(time_dic) != step_num:
                return time_dic, False
            return time_dic, valid
        except Exception:
            return dict(), False


def get_time_dic(response_contents,
                 step_num):
    '''
    Get time annotation from prompt and return whether the dictionary contains all steps for numerical time
        Parameters:
            time_prompt: prompt for time annotation
            step_num: number of steps
            model: model name
            temperature: temperature
            n_choices: number of choices
        Returns:
            response_contents: list of response contents
            time_dics: list of time dictionaries
            valids: list of booleans indicating whether the time annotation is valid
    '''
    try:
        time_dics = list()
        valids = list()

        for response in response_contents:
            time_dic, valid = get_time_dic_per_response(response, step_num)
            time_dics.append(time_dic)
            valids.append(valid)

        return time_dics, valids
    except Exception:
        return [], []

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file',
                        type=str,
                        default=None,
                        required=True)
    parser.add_argument('--output_dir',
                        type=str,
                        default=None,
                        required=True)
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

    nec_nshot_prompt = '''###Example:
    To 'Make a Chicken Sandwich', here is a script in 'Making a Fried Chicken Sandwich'.
    step1: Done!; step2: Add oil to a large frying pan.; step3: Cut the chicken into thin strips and add toppings of your choice.; step4: Get the necessary ingredients.; step5: Mix the batter.; step6: Batter the chicken.; step7: Put each piece of chicken in the pan.
    Question: Is this script showing different alternatives to complete this task? Let's think step by step then provide final answer yes or no in double quotes.
    Answer: The steps as presented are not in a logical sequential order. However, they don't provide alternative methods to make a fried chicken sandwich but rather are parts of a single method that are out of order. To properly make a sandwich, these steps need to be rearranged into a sensible sequence (e.g., gathering ingredients, preparing the chicken and batter, frying the chicken, and assembling the sandwich).
    So, the final answer is: "No".

    Here is a script to 'Help Your Teen Learn to Control Their Asthma'.
    step1: Follow the action plan.; step2: Remove the trigger.; step3: Pay attention to the signs.
    Question: Is this script showing different alternatives to complete this task? Let's think step by step then provide final answer yes or no in double quotes.
    Answer: These steps seem to be complementary rather than alternative paths. Each step builds upon the previous one to create a comprehensive approach to asthma management. The first step establishes a plan, the second step aims to prevent triggers that can cause asthma attacks, and the third step is about monitoring and responding to asthma symptoms promptly.
    So, the final answer is: "No".

    To 'Be a Guest During Birth', here is a script in 'Discuss the Mother’s Preferences in Advance'.
    step1: Find out how she expects you to behave.; step2: Do not assume the invitation is extended to your family or significant other.; step3: Ask about the mother’s guest list.
    Question: Is this script showing different alternatives to complete this task? Let's think step by step then provide final answer yes or no in double quotes.
    Answer: The steps appear to be complementary aspects of a single approach: discussing and understanding the mother's preferences before the birth. Each step is part of a respectful and considerate process to ensure that your presence is supportive and appropriate. They aren't alternative ways to do the task but rather different components of the same task.
    The script provides a structured approach to ensuring you understand and respect the mother's wishes and boundaries as a guest during birth. Therefore, it is not showing different alternatives but rather detailing consecutive steps within a single method or strategy.
    So, the final answer is: "No".

    To 'Get OSHA Certified', here is a script in 'Satisfying OSHA Standards'.
    step1: Receive help as a small business.; step2: Hire a consultant as a large business.; step3: Identify your responsibilities as an employer.; step4: Learn about OSHA’s standards.; step5: Implement training standards.; step6: Seek SHARP recognition as a small business.
    Question: Is this script showing different alternatives to complete this task? Let's think step by step then provide final answer yes or no in double quotes.
    Answer: Some steps seem tailored to different sizes of businesses (steps 1 and 2 suggest different actions for small and large businesses, respectively). Different businesses only need to execute part of the plan based on their sizes. Therefore, the script does provide different alternatives for the same task.
    So, the final answer is: "Yes".

    To 'Avoid a Boring Life', here is a script in 'Exploring New places'.
    step1: Hike to challenge yourself.; step2: Visit local attractions.; step3: Travel with friends and alone.; step4: Walk, instead of driving.
    Question: Is this script showing different alternatives to complete this task? Let's think step by step then provide final answer yes or no in double quotes.
    Answer: The steps provided are not necessarily sequential; they don't need to be followed in the order presented to achieve the goal of avoiding a boring life through exploring new places. Instead, each step represents a different strategy or activity that can independently contribute to a more varied and engaging life. They are various approaches one might take to break the monotony and introduce new experiences.
    The script seems to be providing different alternatives or strategies for exploring new places, each of which can help avoid a boring life. The alternatives are not mutually exclusive and can be mixed and matched according to personal preference, time, and ability.
    So, the final answer is: "Yes".
    '''

    with open(args.input_file, 'r') as f:
        sampled_dic = json.load(f)

    # first check if all parts of different plans are necessary
    idx_to_loop = list()
    nec_prompts_parts = sampled_dic['nec_prompts_parts']
    start_idx = 0
    nec_parts_results = list()
    while start_idx < len(nec_prompts_parts):
        curr_prompts = nec_prompts_parts[start_idx:min(start_idx+80, len(nec_prompts_parts))]
        temp_res = await generate_from_openai_chat_completion(client,
                                                              curr_prompts,
                                                              nshot_prompt=nec_nshot_prompt,
                                                              model='gpt-35-turbo')
        nec_parts_results.extend(temp_res)
        start_idx += 80

    res_dic = {'nec_responses_parts': list(),
               'nec_bool_parts': list(),
               'nec_content_parts': list()}

    for i, nec_prompt_part in enumerate(nec_prompts_parts):
        if not nec_prompt_part or not check_safety(nec_parts_results[i]):
            res_dic['nec_responses_parts'].append([])
            res_dic['nec_bool_parts'].append([])
            res_dic['nec_content_parts'].append([])
            if not nec_prompt_part:
                idx_to_loop.append(i)
            continue

        res_dic['nec_responses_parts'].append(nec_parts_results[i])
        nec_content_parts = [choice['message']['content'] for choice in nec_parts_results[i]['choices']]
        res_dic['nec_content_parts'].append(nec_content_parts)
        nec_bool_parts = veri_nec(nec_content_parts)
        res_dic['nec_bool_parts'].append(nec_bool_parts)
        if nec_bool_parts:
            idx_to_loop.append(i)

    # then check if all steps in different plans are necessary
    nec_prompts_instances = list()
    nxt_idx_to_loop = list()
    for i in range(len(sampled_dic['nec_prompts_instance'])):
        if i in idx_to_loop:
            nec_prompts_instances.extend(sampled_dic['nec_prompts_instance'][i])
        else:
            # add empty prompt to keep list aligned
            nec_prompts_instances.extend(['' for i in range(len(sampled_dic['nec_prompts_instance'][i]))])
    # assert lists are aligned
    assert len(nec_prompts_instances) == len([kk for k in sampled_dic['nec_prompts_instance'] for kk in k])

    start_idx = 0
    nec_result_instances = list()
    while start_idx < len(nec_prompts_instances):
        temp_res = await generate_from_openai_chat_completion(client,
                                                              nec_prompts_instances[start_idx:start_idx+80],
                                                              nec_nshot_prompt,
                                                              model='gpt-35-turbo')
        nec_result_instances.extend(temp_res)
        start_idx += 80

    res_dic.update({'nec_responses_instances': list(),
                    'nec_bool_instances': list(),
                    'nec_content_instances': list()})

    curr_idx = 0
    for i in range(len(sampled_dic['nec_prompts_instance'])):
        if i not in idx_to_loop:
            res_dic['nec_responses_instances'].append([])
            res_dic['nec_bool_instances'].append([])
            res_dic['nec_content_instances'].append([])
            curr_idx = curr_idx+len(sampled_dic['nec_prompts_instance'][i])
            continue

        nec_responses_instances = list()
        nec_bool_instances = list()
        nec_content_instances = list()
        for j in range(len(sampled_dic['nec_prompts_instance'][i])):
            if not check_safety(nec_result_instances[curr_idx]):
                nec_responses_instances.append([])
                nec_bool_instances.append([])
                nec_content_instances.append([])
                curr_idx += 1
                continue
            nec_responses_instances.append(nec_result_instances[curr_idx])
            nec_content_instances.append([choice['message']['content'] for choice in nec_result_instances[curr_idx]['choices']])
            nec_bool_instances.append(veri_nec([choice['message']['content'] for choice in nec_result_instances[curr_idx]['choices']]))

        if sampled_dic['multi_parts'][i]:
            if False not in nec_bool_instances:
                nxt_idx_to_loop.append((i, 'all'))
            continue

        for k, nec_bool_instance in enumerate(nec_bool_instances):
            if nec_bool_instance:
                nxt_idx_to_loop.append((i, k))

    # annotate time
    time_prompts = list()
    nnxt_idx_to_loop = list()
    for i in range(len(sampled_dic['time_prompts'])):
        for j in range(len(sampled_dic['time_prompts'][i])):
            if (i, j) in nxt_idx_to_loop:
                time_prompts.append(sampled_dic['time_prompts'][i][j])
            elif (not j) and (i, 'all') in nxt_idx_to_loop:
                time_prompts.append(sampled_dic['time_prompts'][i][j])
            else:
                # add empty prompt to keep list aligned
                time_prompts.append('')

    assert len(time_prompts) == len([kk for k in sampled_dic['time_prompts'] for kk in k])
    start_idx = 0
    time_results = list()
    while start_idx < len(time_prompts):
        temp_res = await generate_from_openai_chat_completion(client,
                                                              time_prompts[start_idx:start_idx+80],
                                                              model='gpt-35-turbo')
        time_results.extend(temp_res)
        start_idx += 80
        if not start_idx % 800:
            print(start_idx)

    res_dic['time_responses'] = list()
    res_dic['time_dics'] = list()
    res_dic['time_valids'] = list()
    nnxt_idx_to_loop = list()

    curr_idx = 0
    for i in range(len(sampled_dic['time_prompts'])):
        temp_time_responses = list()
        temp_time_dics = list()
        temp_time_valids = list()
        for j in range(len(sampled_dic['time_prompts'][i])):
            if ((i, 'all') not in set(nxt_idx_to_loop) and (i, j) not in set(nxt_idx_to_loop)) or not check_safety(time_results[curr_idx]):
                temp_time_responses.append([])
                temp_time_dics.append([])
                temp_time_valids.append([])
            else:
                temp_time_responses.append(time_results[curr_idx])
                if sampled_dic['multi_parts'][i]:
                    time_dics, valids = get_time_dic([choice['message']['content'] for choice in time_results[curr_idx]['choices']],
                                                     sampled_dic['max_step_num'][i])
                else:
                    time_dics, valids = get_time_dic([choice['message']['content'] for choice in time_results[curr_idx]['choices']],
                                                     sampled_dic['step_num'][i][j])
                temp_time_dics.append(time_dics)
                temp_time_valids.append(valids)
                if True in valids:
                    if not sampled_dic['multi_parts'][i]:
                        nnxt_idx_to_loop.append((i, j))
                    else:
                        nnxt_idx_to_loop.append((i, 'all'))
            curr_idx += 1
        res_dic['time_responses'].append(temp_time_responses)
        res_dic['time_dics'].append(temp_time_dics)
        res_dic['time_valids'].append(temp_time_valids)

    # annotate dependencies for different parts
    dep_prompts_parts = list()
    for i in range(len(sampled_dic['dep_prompts_part'])):
        if (i, 'all') in nnxt_idx_to_loop:
            dep_prompts_parts.append(sampled_dic['dep_prompts_part'][i])
        else:
            # add empty prompt to keep list aligned
            dep_prompts_parts.append('')
    # assert lists are aligned
    assert len(dep_prompts_parts) == len(sampled_dic['dep_prompts_part'])

    dep_results_parts = list()
    start_idx = 0
    while start_idx < len(dep_prompts_parts):
        temp_res = await generate_from_openai_chat_completion(client,
                                                              dep_prompts_parts[start_idx:start_idx+20],
                                                              model='gpt-4',
                                                              system_prompt='You are ChatGPT.',
                                                              n_choices=5)
        dep_results_parts.extend(temp_res)
        start_idx += 20

    fin_idx_to_loop = set(nnxt_idx_to_loop)
    res_dic['dep_responses_parts'] = list()
    res_dic['dep_content_parts'] = list()
    res_dic['edge_list_parts'] = list()
    res_dic['consistent_parts'] = list()
    res_dic['most_freq_deps_parts'] = list()

    for i in range(len(dep_results_parts)):
        if ((i, 'all') not in nnxt_idx_to_loop) or not check_safety(dep_results_parts[i]):
            res_dic['dep_responses_parts'].append([])
            res_dic['dep_content_parts'].append([])
            res_dic['edge_list_parts'].append([])
            res_dic['consistent_parts'].append([])
            res_dic['most_freq_deps_parts'].append([])
            if (i, 'all') in nnxt_idx_to_loop:
                fin_idx_to_loop.remove((i, 'all'))
            continue
        dep_content_parts = [choice['message']['content'] for choice in dep_results_parts[i]['choices']]
        res_dic['dep_responses_parts'].append(dep_results_parts[i])
        res_dic['dep_content_parts'].append(dep_content_parts)
        dep_edge_list, most_freq_deps, count_freq = get_edge_list(dep_content_parts)
        res_dic['edge_list_parts'].append(dep_edge_list)
        res_dic['most_freq_deps_parts'].append(most_freq_deps)
        res_dic['consistent_parts'].append(count_freq)
        if count_freq < 4:
            fin_idx_to_loop.remove((i, 'all'))

    # find low-level plans with valid dependency
    dep_prompts_instance = list()
    res_idx = list()
    for i in range(len(sampled_dic['dep_prompts'])):
        for j in range(len(sampled_dic['dep_prompts'][i])):
            if (i, j) in fin_idx_to_loop or (i, 'all') in fin_idx_to_loop:
                dep_prompts_instance.append(sampled_dic['dep_prompts'][i][j])
                res_idx.append((i, j))
            else:
                # add empty prompt to keep list aligned
                dep_prompts_instance.append('')

    # assert lists are aligned
    assert len(dep_prompts_instance) == len([kk for k in sampled_dic['dep_prompts'] for kk in k])
    start_idx = 0
    dep_results_instances = list()
    while start_idx < len(dep_prompts_instance):
        temp_res = await generate_from_openai_chat_completion(client,
                                                              dep_prompts_instance[start_idx:start_idx+20],
                                                              model='gpt-4',
                                                              system_prompt='You are ChatGPT.',
                                                              n_choices=5)
        dep_results_instances.extend(temp_res)
        start_idx += 20

    res_dic['most_freq_deps_instance'] = list()
    res_dic['dep_responses_instance'] = list()
    res_dic['dep_content_instance'] = list()
    res_dic['edge_list_instance'] = list()
    res_dic['consistent_instance'] = list()

    curr_idx = 0
    for i in range(len(sampled_dic['dep_prompts'])):
        temp_dep_responses = list()
        temp_dep_content = list()
        temp_edge_list = list()
        temp_consistent = list()
        temp_most_freq_deps = list()
        for j in range(len(sampled_dic['dep_prompts'][i])):
            if (i, j) not in fin_idx_to_loop and (i, 'all') not in fin_idx_to_loop:
                temp_dep_responses.append([])
                temp_dep_content.append([])
                temp_edge_list.append([])
                temp_consistent.append([])
                temp_most_freq_deps.append([])
                curr_idx += 1
                continue
            else:
                temp_dep_responses.append(dep_results_instances[curr_idx])
                temp_dep_content.append([choice['message']['content'] for choice in dep_results_instances[curr_idx]['choices']])
                edge_list, most_freq_deps, count_freq = get_edge_list([choice['message']['content'] for choice in dep_results_instances[curr_idx]['choices']])
                temp_edge_list.append(edge_list)
                temp_consistent.append(count_freq)
                temp_most_freq_deps.append(most_freq_deps)
                curr_idx += 1
                continue
        res_dic['dep_responses_instance'].append(temp_dep_responses)
        res_dic['dep_content_instance'].append(temp_dep_content)
        res_dic['edge_list_instance'].append(temp_edge_list)
        res_dic['consistent_instance'].append(temp_consistent)
        res_dic['most_freq_deps_instance'].append(temp_most_freq_deps)

    os.mkdir(args.output_dir, exist_ok=True)
    with open(f'{args.output_dir}/annotated_wikihow.json', 'w') as f:
        json.dump(res_dic, f)


if __name__ == '__main__':
    asyncio.run(main())
