'''
Preprocess wikihow data
'''
import os
import json
import random
from os import listdir
from os.path import isfile, join
import re
import argparse


def replace_url(text):
    '''
    Replace urls in text with [URL]
        Parameters:
            text (str): text to be processed
        Returns:
            replaced_text (str): text with urls replaced with [URL]
    '''

    # Regular expression pattern for matching URLs
    url_pattern = r'\b(?:https?://)?(?:www\.)?[a-zA-Z0-9-]+(?:\.[a-zA-Z]{2,})+(?:/[^\s]*)?\b'

    # Replace URLs in the text with [URL]
    replaced_text = re.sub(url_pattern, '[URL]', text)

    return replaced_text


def read_file_to_dict(path,
                      helpful_percent=60,
                      load_by_method=True):
    '''
    Read wikihow data from json files to a dictionary
        Parameters:
            path (str): path to the folder containing wikihow data
            helpful_percent (float): only store scripts with scores >= this threshold
            load_by
        Returns:
            wikihow_dic (dict): a dictionary containing wikihow data
    '''
    wikihow_dic = {'titles': list(),  # list of titles for each plan
                   'steps': list(),  # list of steps for each plan
                   'multi_methods': list(),  # list of whether each plan has more than one method
                   'multi_parts': list(),  # list of whether each plan has more than one part
                   'step_num': list(),  # number of steps in each plan
                   'max_step_num': list(),  # maximum number of steps in all plans
                   'original_data': list(),  # store the original data for error analysis but we should remove it in final submission for anonymity
                   'helpful_percent': list(),  # helpfulness of each plan
                   'n_votes': list()  # number of votes for each plan
                   }
    wikihow_path = [f for f in listdir(path) if isfile(join(path, f))]
    # keep a lower-case set to check for duplicate plans
    non_dup_plans = set()

    for subpath in wikihow_path:
        with open(path+subpath, 'r') as f:
            task = json.load(f)

        # sanity check
        if (task.get('methods') and task.get('steps')) or (task.get('methods') and task.get('parts')) or (task.get('steps') and task.get('parts')):
            print(task)
            break
        if not task['rating']['helpful_percent'] or task['rating']['helpful_percent'] < helpful_percent:
            continue

        title = task['title']
        steps = list()
        step_num = list()
        # keep a lower-case set to check for duplicate steps
        lower_steps = set()
        if task.get('steps'):
            # when there is only one method data format is as below
            steps.append([title, list()])
            for step in task['steps']:
                headline = replace_url(step['headline']).strip()+': '+replace_url(step['description']).strip()
                # deduplicate
                if headline.lower() not in lower_steps:
                    steps[-1][-1].append(headline)
                    lower_steps.add(headline.lower())

            step_num.append(len(steps[-1][-1]))
            # make sure there is no duplicate steps
            assert len(steps[-1][-1]) == len(lower_steps)

        if task.get('parts'):
            step_num = [0]
            for part in task['parts']:
                steps.append([part['name'], list()])
                lower_steps = set()
                for step in part['steps']:
                    headline = replace_url(step['headline']).strip()+': '+replace_url(step['description']).strip()
                    if headline.lower() not in lower_steps:
                        steps[-1][-1].append(headline)
                        lower_steps.add(headline.lower())
                if not len(steps[-1]):
                    steps.pop()
                    continue
                if tuple(sorted(steps[-1][-1])) not in non_dup_plans:
                    non_dup_plans.add(tuple(sorted(steps[-1][-1])))
                    step_num[0] += len(steps[-1][-1])
                    assert len(steps[-1][-1]) == len(lower_steps)
                else:
                    steps.pop()

        if task.get('methods'):
            if load_by_method:
                for method in task['methods']:
                    lower_steps = set()
                    # when there are more than one method data format is as below
                    steps.append([method['name'], list()])
                    for step in method['steps']:
                        headline = replace_url(step['headline']).strip()+': '+replace_url(step['description']).strip()
                        if headline.lower() not in lower_steps:
                            steps[-1][-1].append(headline)
                            lower_steps.add(headline.lower())
                    if not len(steps[-1]):
                        steps.pop()
                        continue
                    if tuple(sorted(steps[-1][-1])) not in non_dup_plans:
                        non_dup_plans.add(tuple(sorted(steps[-1][-1])))
                        step_num.append(len(steps[-1][-1]))
                        assert len(steps[-1][-1])==len(lower_steps)
                    else:
                        steps.pop()
            else:
                steps.append([title, list()])
                for method in task['methods']:
                    for step in method['steps']:
                        headline = replace_url(step['headline']).strip()+': '+replace_url(step['description']).strip()
                        if headline.lower() not in lower_steps:
                            steps[-1][-1].append(headline)
                            lower_steps.add(headline.lower())

                step_num.append(len(steps[-1][-1]))
                # make sure there is no duplicate steps
                assert len(steps[-1][-1]) == len(lower_steps)

        if not steps:
            continue
        if tuple(sorted(steps[-1][-1])) not in non_dup_plans or (task.get('methods') and load_by_method) or task.get('parts'):
            non_dup_plans.add(tuple(sorted(steps[-1][-1])))
            wikihow_dic['titles'].append(title)
            wikihow_dic['steps'].append(steps)
            wikihow_dic['max_step_num'].append(max(step_num))
            wikihow_dic['helpful_percent'].append(task['rating']['helpful_percent'])
            wikihow_dic['n_votes'].append(task['rating']['n_votes'])
            wikihow_dic['step_num'].append(step_num)
            wikihow_dic['original_data'].append(task)

            if task.get('methods'):
                wikihow_dic['multi_methods'].append(1)
                # sort methods by steps
                s = wikihow_dic['step_num'][-1]
                sorted_idx = sorted(range(len(s)), key=lambda k: s[k], reverse=True)
                wikihow_dic['step_num'][-1] = [s[i] for i in sorted_idx]
                wikihow_dic['steps'][-1] = [wikihow_dic['steps'][-1][i] for i in sorted_idx]
            else:
                wikihow_dic['multi_methods'].append(0)

            if task.get('parts'):
                wikihow_dic['multi_parts'].append(1)
            else:
                wikihow_dic['multi_parts'].append(0)

    print('Number of plans: ', len(wikihow_dic['steps']))

    return wikihow_dic


def generate_wikihow_task_description(wikihow_data, idx, task):
    '''
    Generate task description from wikihow dataframe
        Parameters:
            wikihow_data: dictionary
            idx: index of the plan in the dictionary
            task: task name
        Returns:
            prompts: list of plan descriptions
    '''
    title = wikihow_data['titles'][idx].split('How to ')[1]
    part_prompt = ''
    # assert a plan does not have both multi_parts and multi_methods
    assert not (wikihow_data['multi_parts'][idx] and wikihow_data['multi_methods'][idx])

    if task == 'dependency':
        if wikihow_data['multi_parts'][idx]:
            prompts = list()
            part_prompt = f'Here are steps needed to \'{title}\'. '
            random.seed(0)
            part_steps = '\n'.join([f'step{ii+1}: {s}' for ii, s in enumerate(random.sample([s[0] for s in wikihow_data['steps'][idx]], len(wikihow_data['steps'][idx])))])
            part_idx = len(wikihow_data['steps'][idx])
            part_steps += f'\n{part_idx+1}: NONE.'
            part_steps += f'\n{part_idx+2}: {title}.'
            part_prompt += f'\n{part_steps}\n'
            part_prompt += 'Question: Assume steps should be parallelized where possible, identify steps which must be executed in a sequential order.\nAnswer:'
            for part in wikihow_data['steps'][idx]:
                part_name = part[0]
                random.seed(0)
                steps = '\n'.join([f'step{ii+1}: {s}' for ii, s in enumerate(random.sample(part[-1], len(part[-1])))])
                ind = len(part[-1])
                steps += f'\n{ind+1}: NONE.'
                steps += f'\n{ind+2}: {part_name}.'
                prompt = f'To \'{title}\', here are the steps needed in \'{part_name}\'.'
                prompt += f'\n{steps}\n'
                prompt += 'Question: Assume steps should be parallelized where possible, identify steps which must be executed in a sequential order.\nAnswer:'
                prompts.append(prompt)
        elif wikihow_data['multi_methods'][idx]:
            prompts = list()
            for method in wikihow_data['steps'][idx]:
                method_name = method[0]
                random.seed(0)
                steps = '\n'.join([f'step{ii+1}: {s}' for ii, s in enumerate(random.sample(method[-1], len(method[-1])))])
                ind = len(method[-1])
                steps += f'\n{ind+1}: NONE.'
                steps += f'\n{ind+2}: {method_name}.'
                prompt = f'To \'{title}\', here are the steps needed in \'{method_name}\'. '
                prompt += f'\n{steps}\n'
                prompt += 'Question: Assume steps should be parallelized where possible, identify steps which must be executed in a sequential order.\nAnswer:'
                prompts.append(prompt)
        else:
            prompt = f'Here are the steps needed to \'{title}\'. '
            random.seed(0)
            steps = '\n'.join([f'step{ii+1}: {s}' for ii, s in enumerate(random.sample(wikihow_data['steps'][idx][-1][-1], len(wikihow_data['steps'][idx][-1][-1])))])
            ind = len(wikihow_data['steps'][idx][-1][-1])
            steps += f'\n{ind+1}: NONE.'
            steps += f'\n{ind+2}: {title}.'
            prompt += f'\n{steps}\n'
            prompt += 'Question: Assume steps should be parallelized where possible, identify steps which must be executed in a sequential order.\nAnswer:'
            prompts = [prompt]

    elif task == 'time':
        if not wikihow_data['multi_methods'][idx]:
            prompt = f'Here are the steps needed to \'{title}\'. '
            step_data = list()
            for steps in wikihow_data['steps'][idx]:
                step_data.extend(steps[-1])
            steps = '\n'.join([f'step{ii+1}: {s}' for ii, s in enumerate(step_data)])
            prompt += f'\n{steps}\nQuestion: Let\'s think step by step. How long does it take to complete each step? Answer "ongoing" if any step is a long-time status or "none" if any step is not an action/does not take any time.\nAnswer:'
            prompts = [prompt]

        else:
            prompts = list()
            for method in wikihow_data['steps'][idx]:
                method_name = method[0]
                step_data = method[-1]
                prompt = f'To \'{title}\', here are the steps needed in \'{method_name}\'. '
                steps = '\n'.join([f'step{ii+1}: {s}' for ii, s in enumerate(step_data)])
                prompt += f'\n{steps}\nQuestion: Let\'s think step by step. How long does it take to complete each step? Answer "ongoing" if any step is a long-time status or "none" if any step is not an action/does not take any time.\nAnswer:'
                prompts.append(prompt)

    else:
        raise ValueError('task should be either "dependency" or "time"')

    return prompts, part_prompt


def contain_keywords(steps):
    '''
    Check if a plan contains keywords
        Parameters:
            steps (list): list of steps
        Returns:
            True if the plan contains keywords, False otherwise
    '''
    keywords = set(['this', 'above', 'below', 'keep', 'know', 'knowing', 'opt', 'if', 'when', 'while', 'become', 'be', 'stay', 'repeat', 'after', 'before'])
    words_in_steps = set(re.findall(r"[\w']+", ' '.join([step.lower().split(': ')[0] for step in steps])))

    return len(keywords.intersection(words_in_steps)) > 0


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--wikihow_path',
                        type=str,
                        default=None,
                        required=True,
                        help='Path to wikihow data')
    parser.add_argument('--out_path',
                        type=str,
                        default=None,
                        required=True,
                        help='Path to output data')
    args = parser.parse_args()

    # Load wikihow data
    wikihow_dic = read_file_to_dict(args.wikihow_path)
    filtered_wikihow_dic = {key: [] for key in wikihow_dic.keys()}

    for i, desc_steps in enumerate(wikihow_dic['steps']):
        if not wikihow_dic['multi_methods'][i]:
            steps = list()
            for desc, step in desc_steps:
                steps += step
                steps.append(desc)
            if not contain_keywords(steps):
                for key, val in filtered_wikihow_dic.items():
                    filtered_wikihow_dic[key].append(wikihow_dic[key][i])
            continue

        step_nums = list()
        filtered_steps = list()
        for desc, steps in desc_steps:
            has_keywords = contain_keywords(steps+[desc])
            if has_keywords:
                continue
            filtered_steps.append([desc, steps])
            step_nums.append(len(steps))

        if step_nums:
            for key, val in filtered_wikihow_dic.items():
                if key == 'step_num':
                    val.append(step_nums)
                elif key == 'steps':
                    val.append(filtered_steps)
                elif key == 'max_step_num':
                    val.append(max(step_nums))
                elif key == 'multi_methods':
                    if len(step_nums) > 1:
                        val.append(1)
                    else:
                        val.append(0)
                else:
                    val.append(wikihow_dic[key][i])

    os.mkdir(args.out_path, exist_ok=True)
    with open(f'./{args.out_path}/wikihow_filtered.json', 'w') as f:
        json.dump(filtered_wikihow_dic, f, indent=4)


if __name__ == '__main__':
    main()
