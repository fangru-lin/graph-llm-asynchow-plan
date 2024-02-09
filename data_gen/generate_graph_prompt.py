'''
This file generates the graph, prompt, and answer for the asynchow task.
You have to draft up n-shot prompt and instructions on your own,
as we expect the model to be non-deterministic in producing valid examples.
'''
from copy import deepcopy
from datetime import timedelta
import re
import argparse
import json
from utils.utils import *
import pickle
import random


def edge_list_to_adjacency_structures(edge_list):
    '''
    Convert edge list to adjacency matrix, list, and CSR format
        Parameters:
            edge_list: list of edges
        Returns:
            adj_list: adjacency list
            adj_matrix: adjacency matrix
            csr: CSR format
    '''
    # Convert edge list to unique nodes
    nodes = set()
    for edge in edge_list:
        nodes.update(edge)
    nodes = sorted(list(nodes))  # Sort to maintain order

    # Create mapping of nodes to integers for easier adjacency matrix and CSR representation
    node_to_index = {node: i for i, node in enumerate(nodes)}

    # Initialize adjacency matrix with zeros
    size = len(nodes)
    adj_matrix = [[0] * size for _ in range(size)]

    # Initialize adjacency list as a dictionary
    adj_list = {node: [] for node in nodes}

    # Initialize variables for Compressed Sparse Row (CSR) representation
    values = []  # Non-zero values (1s in this case as it's unweighted)
    col_indices = []  # Column indices of the non-zero values
    row_indices = [0] * (size + 1)  # Row pointer

    # Populate the adjacency matrix, list, and CSR format
    for u, v in edge_list:
        adj_list[u].append(v)

        # For adjacency matrix
        row, col = node_to_index[u], node_to_index[v]
        adj_matrix[row][col] = 1

        values.append(1)
        col_indices.append(col)
        row_indices[row + 1] += 1

    for i in range(1, len(row_indices)):
        row_indices[i] += row_indices[i-1]

    return adj_list, adj_matrix, (values, row_indices, col_indices)


def generate_edge_list(edge_list: tuple,
                       step_num: int):
    '''
    Generate edge list from dependency responses
        Parameters:
            edge_list: list of edge lists
            step_num: number of steps
        Returns:
            res_edge_list: list of edge lists
    '''
    temp_edge_list = list(edge_list)
    # print(edge_list)
    # key precedes value, essentially adjacency list
    precede_graph = {str(n): [] for n in range(1, step_num+1)}
    # key follows value
    follow_graph = {str(n): [] for n in range(1, step_num+1)}
    for ele in edge_list:
        left_number, right_number = ele
        if left_number == '0' or right_number == '0' or int(left_number) > step_num or int(right_number) > step_num:
            continue
        if right_number not in precede_graph[left_number]:
            precede_graph[left_number].append(right_number)
        if left_number not in follow_graph[right_number]:
            follow_graph[right_number].append(left_number)
    res_edge_list = deepcopy(temp_edge_list)
    for key, value in follow_graph.items():
        if not value:
            res_edge_list.append(('START', key))
    for key, val in precede_graph.items():
        if not val:
            res_edge_list.append((key, 'END'))

    return res_edge_list


def get_all_paths(adjacency_list, start, end):
    '''
    Get all paths from start to end node
        Parameters:
            adjacency_list: adjacency list
            start: start node
            end: end node
        Returns:
            all_paths: list of all paths
            shortest_path: shortest path
            longest_path: longest path
            path_count: number of paths
    '''
    # Variables to store paths and shortest path length
    all_paths = []

    # Modified DFS to track all paths
    def dfs(node, path):
        if node == end:
            all_paths.append(path)
            return
        for neighbour in adjacency_list[node]:
            if neighbour not in path:  # Avoid cycles
                dfs(neighbour, path + [neighbour])

    # Start DFS
    dfs(start, [start])

    shortest_path = min(all_paths, key=len)
    longest_path = max(all_paths, key=len)
    path_count = len(all_paths)
    return all_paths, shortest_path, longest_path, path_count


def match_indexes(list1, list2):
    """
    This function returns a dictionary where keys are elements of the first list (list1)
    and values are the indexes in the second list (list2) where these elements can be found.
        Parameters:
            list1: list of elements
            list2: list of elements
        Returns:
            index_match: dictionary of elements from list1 and their matching indexes in list2
    """
    # Create a dictionary to store the element from list1 and its matching index in list2
    index_match = {}

    # Create a dictionary for faster lookup of indexes in list2
    list2_indexes = {element: i for i, element in enumerate(list2)}

    # Iterate through list1 and find matching indexes from list2
    for iter, element in enumerate(list1):
        try:  # Check if the element is in list2
            index_match[iter] = list2_indexes[element]
        except KeyError:  # If element is not in list2, assign None
            raise Exception('Element not found in list2')

    return index_match


def match_shuffled_edges(shuffled_deps,
                         shuffled_to_orig):
    if not shuffled_deps:
        return []
    res_edges = list()
    for edge in shuffled_deps:
        prev, foll = edge
        if prev == '0' or foll == '0':
            continue
        orig_prev = str(shuffled_to_orig[int(prev)-1]+1)
        orig_foll = str(shuffled_to_orig[int(foll)-1]+1)
        res_edges.append((orig_prev, orig_foll))

    return res_edges


def is_line_graph(edge_list):
    # Constructing graph representation using a dictionary
    graph = {}
    indegree = {}  # To keep track of incoming edges
    outdegree = {}  # To keep track of outgoing edges

    for u, v in edge_list:
        if u not in graph:
            graph[u] = []
            indegree[u] = 0
            outdegree[u] = 0
        if v not in graph:
            graph[v] = []
            indegree[v] = 0
            outdegree[v] = 0

        graph[u].append(v)
        outdegree[u] += 1
        indegree[v] += 1

    # Checking linearity: One node with indegree 0 (start), one with outdegree 0 (end), rest should have 1 in and 1 out
    start_nodes = 0
    end_nodes = 0
    for node in graph:
        if indegree[node] == 0:
            start_nodes += 1
        elif outdegree[node] == 0:
            end_nodes += 1
        elif not (indegree[node] == 1 and outdegree[node] == 1):
            return False  # Not linear if more than one in/out edges for middle nodes

    # There must be exactly one start node and one end node for it to be a line
    return start_nodes == 1 and end_nodes == 1


def get_timedelta_interval(time_dic):
    '''
    Parse time description into a dictionary of time steps
        Parameters:
            time_dic: dictionary of time steps
        Returns:
            timedelta_interval_dic: dictionary of time delta for each step
            stepwise_time_dic: dictionary of time for each step
    '''
    timedelta_interval_dic = dict()
    stepwise_time_dic = dict()
    # stepwise_timedelta_dic, stepwise_time_dic = dict(), dict()
    # time_spans = time_description.replace('.','').strip().split('; ')
    for step, time_exp in time_dic.items():
        # time_exp = time_span.split(': ')[-1]
        try:
            time_span = re.findall(r'\d+\s*(?:min|minute|minutes|hr|hour|hours|sec|second|seconds|week|weeks|day|days|month|months|year|years)', time_exp)[-1]
            time = re.findall(r'\d+', time_span)[-1]
            unit = re.findall(r'\b[a-z]+', time_span)[-1].strip()
            if unit in ['year', 'years', 'y']:
                delta = [timedelta(days=int(time)*365), timedelta(days=int(time)*366)]
            elif unit in ['month', 'months', 'm']:
                # define a loose range for month
                # match other units in same format
                delta = [timedelta(days=int(time)*28), timedelta(days=int(time)*31)]
            elif unit in ['week', 'weeks', 'w']:
                delta = [timedelta(weeks=int(time)), timedelta(weeks=int(time))]
            elif unit in ['day', 'days', 'd']:
                delta = [timedelta(days=int(time)), timedelta(days=int(time))]
            elif unit in ['hour', 'hours', 'h']:
                delta = [timedelta(hours=int(time)), timedelta(hours=int(time))]
            elif unit in ['minute', 'min', 'minutes', 'mins']:
                delta = [timedelta(minutes=int(time)), timedelta(minutes=int(time))]
            elif unit in ['second', 'sec', 'seconds', 'secs']:
                delta = [timedelta(seconds=int(time)), timedelta(seconds=int(time))]
            else:
                raise ValueError(f'unit not found: {time_span}')
            stepwise_time = f'{time} {unit}'
        except Exception:
            delta = None
            stepwise_time = ''
        try:
            timedelta_interval_dic[re.findall(r'\d+', step)[0]] = delta
        except Exception:
            continue
        stepwise_time_dic[re.findall(r'\d+', step)[0]] = stepwise_time

    return timedelta_interval_dic, stepwise_time_dic


def get_fin_time_dics(time_dics):
    '''
    Annotate time and time delta for each step
        Parameters:
            time_dics: list of time dictionaries
        Returns:
            res_timedelta_dic: dictionary of time delta for each step
            res_time_dic: dictionary of time for each step
    '''
    res_time_dic = dict()
    res_timedelta_dic = dict()
    temp_timedelta_dic = dict()
    temp_time_dic = dict()
    for i in range(len(time_dics)):
        iter_timedelta_dic, stepwise_time_dic = get_timedelta_interval(time_dics[i])
        for key, val in iter_timedelta_dic.items():
            if key in temp_timedelta_dic and val:
                temp_time_dic[key].append(stepwise_time_dic[key])
                temp_timedelta_dic[key].append((val, len(temp_timedelta_dic.get(key, []))))
            elif val:
                temp_time_dic[key] = [stepwise_time_dic[key]]
                temp_timedelta_dic[key] = [(val, 0)]

    for key, val in temp_timedelta_dic.items():
        # pick the longest time
        # if there is only one time, pick the only time
        longest = sorted(val, key=lambda x: x[0][1])[-1]
        res_timedelta_dic[key] = longest[0]
        res_time_dic[key] = temp_time_dic[key][longest[1]]
        assert res_time_dic[key] != ''
    assert len(res_timedelta_dic) == len(res_time_dic)
    return res_timedelta_dic, res_time_dic


def match_shuffled_time(timedelta_dic,
                        time_dic,
                        orig_to_shuffled):
    '''
    Match the shuffled time to the original time
        Parameters:
            timedelta_dic: dictionary of time delta for each step
            time_dic: dictionary of time for each step
            orig_to_shuffled: mapping from original to shuffled
        Returns:
            res_timedelta_dic: dictionary of time delta for each step
            res_time_dic: dictionary of time for each step
    '''
    res_timedelta_dic = dict()
    res_time_dic = dict()
    for key, val in timedelta_dic.items():
        shuffled_key = str(orig_to_shuffled[int(key)-1]+1)
        res_timedelta_dic[shuffled_key] = val
        res_time_dic[shuffled_key] = time_dic[key]

    return res_timedelta_dic, res_time_dic


def find_plan_type(edge_list, step_num):
    '''
    Find plan type from edge list
        Parameters:
            edge_list: list of edge lists
            step_num: number of steps
        Returns:
            plan_type: plan type
    '''
    if not edge_list:
        return 'parallel'
    if is_line_graph(edge_list):
        nodes_in_edge_list = set([int(node) for edge in edge_list for node in edge])
        if nodes_in_edge_list == set([i+1 for i in range(step_num)]):
            return 'sequential'
    return 'async'


def series_composition(edge_lists,
                       nodes):
    '''
    Composite two graphs in series
        Parameters:
            edge_lists: list of edge lists
            nodes: list of nodes
        Returns:
            fin_edge_list: list of edges
            common_nodes: list of common nodes
    '''
    if len(edge_lists) != 2 or len(nodes) != 2:
        raise ValueError('series composition requires two edge lists')

    temp_edge_list_0 = list(edge_lists[0])
    temp_edge_list_1 = list(edge_lists[1])
    # print(edge_list)
    # key precedes value, essentially adjacency list
    precede_graph_0 = {str(n): [] for n in nodes[0]}
    # key follows value
    follow_graph_0 = {str(n): [] for n in nodes[0]}
    # key precedes value, essentially adjacency list
    precede_graph_1 = {str(n): [] for n in nodes[1]}
    # key follows value
    follow_graph_1 = {str(n): [] for n in nodes[1]}

    for ele in edge_lists[0]:
        left_number, right_number = ele
        if left_number == '0' or right_number == '0':
            continue
        if right_number not in precede_graph_0[left_number]:
            precede_graph_0[left_number].append(right_number)
        if left_number not in follow_graph_0[right_number]:
            follow_graph_0[right_number].append(left_number)

    for ele in edge_lists[1]:
        left_number, right_number = ele
        if left_number == '0' or right_number == '0':
            continue
        if right_number not in precede_graph_1[left_number]:
            precede_graph_1[left_number].append(right_number)
        if left_number not in follow_graph_1[right_number]:
            follow_graph_1[right_number].append(left_number)

    fin_edge_list = temp_edge_list_0+temp_edge_list_1
    for key, value in precede_graph_0.items():
        if not value:
            for key_1, val_1 in follow_graph_1.items():
                if not val_1:
                    fin_edge_list.append((key, key_1))
    common_nodes = list(set(nodes[0]).union(set(nodes[1])))
    return fin_edge_list, common_nodes


def merge_all_graphs(edge_list_instances,
                     edge_list_part,
                     step_nums_instances):
    '''
    Merge all graphs into one graph
        Parameters:
            edge_list_instances: list of edge lists
            edge_list_part: list of edges
            step_nums_instances: list of step numbers
        Returns:
            fin_edge_list: list of edges
            fin_edge_list_instances: list of edge lists
    '''
    # print(edge_list_instances, edge_list_part, step_nums_instances)
    temp_edge_list_instances = deepcopy(edge_list_instances)
    start_idxs = [1]
    for i, edge_list_instance in enumerate(temp_edge_list_instances):
        if not i:
            continue
        plus_step = sum(step_nums_instances[:i])
        start_idxs.append(plus_step+1)
        if not edge_list_instance:
            continue
        for j, edge in enumerate(edge_list_instance):
            temp_edge_list_instances[i][j] = (str(int(edge[0])+plus_step), str(int(edge[1])+plus_step))
    edge_list_dic = {str(i+1): temp_edge_list_instances[i] for i in range(len(temp_edge_list_instances))}
    nodes = [[str(n) for n in range(start_idxs[i], start_idxs[i]+step_nums_instances[i])] for i in range(len(step_nums_instances))]
    for edge in edge_list_part:
        edges = [edge_list_dic[edge[0]], edge_list_dic[edge[1]]]
        nodess = [nodes[int(edge[0])-1], nodes[int(edge[1])-1]]
        composition, updated_nodes = series_composition(edges,
                                                        nodess)
        edge_list_dic[edge[0]] = composition
        edge_list_dic[edge[1]] = composition
        nodes[int(edge[0])-1] = updated_nodes
        nodes[int(edge[1])-1] = updated_nodes

    fin_edge_list = set()
    for edge_list in edge_list_dic.values():
        fin_edge_list.update(edge_list)

    total_step_num = sum(step_nums_instances)
    return fin_edge_list, generate_edge_list(tuple(fin_edge_list), total_step_num)


def generate_task_time(all_paths,
                       timedelta_dic,):
    '''
    Generate the time for each task in the plan
        Parameters:
            all_paths: list of all paths
            timedelta_dic: dictionary of time delta for each step
        Returns:
            all_corr_paths: list of all paths which can give the same correct answer
            time: shortest time for executing the plan
    '''
    max_time = [timedelta(seconds=0), timedelta(seconds=0)]
    all_corr_paths = list()
    for path in all_paths:
        time = [timedelta(0), timedelta(0)]
        for node in path:
            if node == 'START' or node == 'END':
                continue
            try:
                time = [time[i]+timedelta_dic[node][i] for i in range(2)]
            except Exception:
                print(node, all_paths)
                raise ValueError('node not found in time_dic')
        if time[-1] > max_time[-1]:
            max_time = time
            all_corr_paths = [path]
        elif time[-1] >= max_time[0] and path not in all_corr_paths:
            all_corr_paths.append(path)
            max_time[0] = time[0]

    return all_corr_paths, max_time


def get_all_dependencies_proscript(text, aux_idx):
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
        if prec_node == '0' or int(foll_node) == aux_idx:
            continue
        if int(prec_node) > aux_idx:
            prec_node = str(int(prec_node)-1)
        if int(foll_node) > aux_idx:
            foll_node = str(int(foll_node)-1)
        all_deps.add(tuple([prec_node, foll_node]))

    res_deps = deepcopy(all_deps)
    all_deps = list(all_deps)
    for i, dep in enumerate(all_deps):
        if can_reach(all_deps[:i]+all_deps[i+1:], dep[0], dep[1]):
            res_deps.remove(dep)

    return res_deps


def get_time_per_step_proscript(time_dic):
    '''
    Get time per step by rounding up time to reasonable units
        Parameters:
            script: dict
        Returns:
            time_per_step: dict
    '''
    time_per_step = dict()
    for step, desc in time_dic.items():
        time_desc_dic = {'days': 0, 'hours': 0, 'minutes': 0, 'seconds': 0}
        time_desc_dic['sec'] = int((desc % 1)*60)
        time_desc_dic['days'] = int(desc//1440)
        remaining = desc % 1440-desc % 1
        time_desc_dic['hours'] = int(remaining // 60)
        time_desc_dic['minutes'] = int(remaining % 60)
        time_desc = ''
        if time_desc_dic['days']:
            time_desc += str(time_desc_dic['days'])+' days '
        if time_desc_dic['hours']:
            time_desc += str(time_desc_dic['hours'])+' hours '
        if time_desc_dic['minutes']:
            time_desc += str(time_desc_dic['minutes'])+' minutes '
        if time_desc_dic['sec']:
            time_desc += str(time_desc_dic['sec'])+' seconds '
        time_per_step[step] = time_desc.strip()
    return time_per_step


def sample_test_instances(full_benchmark,
                          from_step,
                          to_step,
                          bin_width,
                          plan_type):
    sampled_idxs = {key: list() for key in range(from_step, to_step+1)}
    sampled_benchmark = {key: list() for key in full_benchmark.keys() if key!='prompts'}
    sampled_benchmark['prompts'] = dict()
    for i in range(len(full_benchmark['titles'])):
        if full_benchmark['plan_types'][i] == plan_type:
            step_num = full_benchmark['step_nums'][i]
            if from_step <= step_num <= to_step:
                sampled_idxs[step_num].append(i)

    for key in sampled_idxs.keys():
        random.seed(2024)
        curr_idx = random.sample(sampled_idxs[key], min(bin_width, len(sampled_idxs[key])))
        for idx in curr_idx:
            for key in sampled_benchmark.keys():
                if key != 'prompts':
                    sampled_benchmark[key].append(full_benchmark[key][idx])
                else:
                    for template, data in full_benchmark[key].items():
                        if template not in sampled_benchmark[key]:
                            sampled_benchmark[key][template] = dict()
                        for key2, val in data.items():
                            if key2 not in sampled_benchmark[key][template]:
                                sampled_benchmark[key][template][key2] = list()
                            sampled_benchmark[key][template][key2].append(val[idx])

    return sampled_benchmark


def generate_prompt(title,
                    steps,
                    time_dic,
                    step_deps,
                    graph=None,
                    graph_type=None,
                    cot=False,
                    bag=False,
                    adjacency_list=None,
                    ordering_template='[preceding step] must precede [following step].'):
    '''
    Generate prompts
        Parameters:
            title: string
            steps: list of strings
            time_dic: dictionary of time for each step
            step_deps: list of edges
            graph: graph representation
            graph_type: graph type
        Returns:
            prompt: string
    '''
    task = title.replace('How to ', '')
    template = '''To [TASK], here are the steps and the times needed for each step.
    [STEPS AND TIME]

    These ordering constraints need to be obeyed when executing above steps:
    [ORDERING CONSTRAINTS]

    [GRAPH]

    Question: Assume that you need to execute all the steps to complete the task and that infinite resources are available. What is the shortest possible time to [TASK]? Answer the time in double quotes.\nAnswer:'''

    prompt = template.replace('[TASK]', task)
    step_and_time = ''
    for i, step in enumerate(steps):
        step_and_time += f'Step {i+1}. {step} ({time_dic[str(i+1)]})\n'
    prompt = prompt.replace('[STEPS AND TIME]', step_and_time)
    if not step_deps:
        prompt = prompt.replace('\nThese ordering constraints need to be obeyed when executing above steps:\n[ORDERING CONSTRAINTS]', 'All steps can be executed in parallel.')
    else:
        ordering_constraints = list()
        if not adjacency_list:
            for edge in step_deps:
                temp_constraint = ordering_template.replace('[preceding step]', f'step {edge[0]}')
                temp_constraint = temp_constraint.replace('[following step]', f'step {edge[1]}')
                ordering_constraints.append(f'{temp_constraint}\n'.capitalize())
        else:
            for prec, foll in adjacency_list.items():
                if prec == 'START' or not foll or foll == ['END']:
                    continue
                meaningful_folls = sorted([ele for ele in foll if ele != 'END'])
                foll_steps = ', '.join(meaningful_folls[:-1])+' and '+meaningful_folls[-1] if len(meaningful_folls)>1 else meaningful_folls[0]
                temp_constraint = ordering_template.replace('[preceding step]', f'step {prec}')
                temp_constraint = temp_constraint.replace('[following step]', f'step {foll_steps}')
                ordering_constraints.append(f'{temp_constraint}\n'.capitalize())
        prompt = prompt.replace('[ORDERING CONSTRAINTS]', ''.join(sorted(ordering_constraints)))

    if graph and graph_type:
        if graph_type == 'adjacency_list':
            prompt = prompt.replace('[GRAPH]', f'Here is the adjacency list representation of the step ordering constraints:\n{graph}\nTime for each step can be represented as a dictionary:\n{time_dic}')
        elif graph_type == 'adjacency_matrix':
            prompt = prompt.replace('[GRAPH]', f'Here is the adjacency matrix representation of the step ordering constraints (assuming the node order to be 1 to {len(steps)}, \'END\', \'START\'):\n{graph}\nTime for each step can be represented as a dictionary:\n{time_dic}')
        elif graph_type == 'csr':
            values, row_indices, col_indices = graph
            prompt = prompt.replace('[GRAPH]', f'Here is the compressed sparse row representation of the step ordering constraints (assuming the node order to be 1 to {len(steps)}, \'END\', \'START\'):\nValues: {values}\nRow indices: {row_indices}\nColumn indices: {col_indices}\nTime for each step can be represented as a dictionary:\n{time_dic}')
        elif graph_type == 'edge_list':
            prompt = prompt.replace('[GRAPH]', f'Here is the edge list representation of the step ordering constraints:\n{graph}\nTime for each step can be represented as a dictionary:\n{time_dic}')
        else:
            raise ValueError('graph type not found')
    else:
        prompt = prompt.replace('\n[GRAPH]\n', '')

    if bag:
        assert not graph_type and not graph
        prompt = prompt.replace('Answer the time in double quotes.', 'Let\'s construct a graph with the nodes and edges first to represent step ordering constraints, and also construct a dictionary to represent time needed for each step. Use the graph and dictionary to calculate the shortest possible time needed for the task. Answer the time in double quotes.')

    if cot:
        prompt = prompt.replace('Answer the time in double quotes.', 'Let\'s think step by step and then answer the time in double quotes.')

    return prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocessed_wikihow_path',
                        type=str,
                        default=None,
                        required=True,
                        help='Path to preprocessed wikihow')
    parser.add_argument('--annotated_wikihow_path',
                        type=str,
                        default=None,
                        required=True,
                        help='Path to annotated wikihow')
    parser.add_argument('--proscript_dir',
                        type=str,
                        default=None,
                        required=True,
                        help='Path to proscript directory')
    parser.add_argument('--output_dir',
                        type=str,
                        default=None,
                        required=True,
                        help='Path to output directory')

    args = parser.parse_args()

    with open(args.preprocessed_wikihow_path, 'r') as f:
        sampled_dic = json.load(f)
    with open(args.annotated_wikihow_path, 'r') as f:
        res_dic = json.load(f)
    proscript = list()
    for f_name in ['train', 'dev', 'test']:
        with open(f'{args.proscript_dir}/{f_name}.json', 'r') as f:
            proscript.extend([json.loads(i) for i in f.readlines()])

    nl_prompt_templates = ["Before starting [following step], complete [preceding step].",
                           "[preceding step] must precede [following step].",
                           "Upon completing [preceding step], proceed to [following step].",
                           "After [preceding step], [following step] should commence.",
                           "Prioritize [preceding step] before advancing to [following step].",
                           "[preceding step] is a prerequisite for [following step].",
                           "Initiate [following step] subsequent to [preceding step].",
                           "Ensure [preceding step] is done before [following step].",
                           "Sequence the tasks: firstly [preceding step], then [following step].",
                           "[following step] follows the completion of [preceding step]."]

    fin_res_dic = {'titles': list(),
                   'multi_methods': list(),
                   'multi_parts': list(),
                   'steps': list(),
                   'step_nums': list(),
                   'sub_titles': list(),
                   'orig_idx': list(),
                   'step_deps': list(),
                   'edge_list': list(),
                   'adjacency_matrix': list(),
                   'csr': list(),
                   'adjacency_list': list(),
                   'time_dic': list(),
                   'timedelta_dic': list(),
                   'longest_path_len': list(),
                   'shortest_path_len': list(),
                   'path_count': list(),
                   'all_paths': list(),
                   've_count': list(),
                   'plan_types': list(),
                   'all_corr_paths': list(),
                   'task_time': list()}

    for i in range(len(sampled_dic['titles'])):
        if not sampled_dic['multi_parts'][i] or len(res_dic['consistent_instance'][i])==1:
            step_nums = sampled_dic['step_num'][i]
            consistent_instances = res_dic['consistent_instance'][i]
            for ii, cons in enumerate(consistent_instances):
                if cons and cons >= 4:
                    shuffled_to_orig = match_indexes(sampled_dic['shuffled_instances_steps'][i][ii],
                                                     sampled_dic['steps'][i][ii][-1])
                    # print(cons, shuffled_to_orig, res_dic['dep_results_instances'][i][ii])
                    orig_edge_list = match_shuffled_edges(res_dic['most_freq_deps_instance'][i][ii],
                                                          shuffled_to_orig)
                    complete_edge_list = generate_edge_list(tuple(orig_edge_list), step_nums[ii])
                    plan_type = find_plan_type(orig_edge_list, step_nums[ii])
                    ve = len(complete_edge_list)+step_nums[ii]+2
                    adjacency_list, adjacency_matrix, (values, row_indices, col_indices) = edge_list_to_adjacency_structures(complete_edge_list)
                    fin_timedelta_dic, fin_time_dic = get_fin_time_dics(res_dic['time_dics'][i][ii])
                    all_paths, shortest_path_len, longest_path_len, path_count = get_all_paths(adjacency_list,
                                                                                               'START',
                                                                                               'END')
                    all_corr_paths, task_time = generate_task_time(all_paths, fin_timedelta_dic)
                    fin_res_dic['titles'].append(sampled_dic['titles'][i])
                    fin_res_dic['multi_methods'].append(sampled_dic['multi_methods'][i])
                    fin_res_dic['multi_parts'].append(sampled_dic['multi_parts'][i])
                    fin_res_dic['steps'].append(sampled_dic['steps'][i][ii][-1])
                    fin_res_dic['step_nums'].append(step_nums[ii])
                    fin_res_dic['sub_titles'].append(sampled_dic['steps'][i][ii][0])
                    fin_res_dic['orig_idx'].append(sampled_dic['orig_idx'][i])
                    fin_res_dic['edge_list'].append(complete_edge_list)
                    fin_res_dic['step_deps'].append(orig_edge_list)
                    fin_res_dic['adjacency_matrix'].append(adjacency_matrix)
                    fin_res_dic['csr'].append((values, row_indices, col_indices))
                    fin_res_dic['adjacency_list'].append(adjacency_list)
                    fin_res_dic['time_dic'].append(fin_time_dic)
                    fin_res_dic['timedelta_dic'].append(fin_timedelta_dic)
                    fin_res_dic['longest_path_len'].append(longest_path_len)
                    fin_res_dic['shortest_path_len'].append(shortest_path_len)
                    fin_res_dic['path_count'].append(path_count)
                    fin_res_dic['all_paths'].append(all_paths)
                    fin_res_dic['ve_count'].append(ve)
                    fin_res_dic['plan_types'].append(plan_type)
                    fin_res_dic['all_corr_paths'].append(all_corr_paths)
                    fin_res_dic['task_time'].append(task_time)
        else:
            instance_step_nums = sampled_dic['step_num'][i]
            consistent_instances = res_dic['consistent_instance'][i]
            consistent_parts = res_dic['consistent_parts'][i]
            edge_list_instances = list()

            if (4 not in consistent_instances) and (5 not in consistent_instances):
                continue
            for ii, cons in enumerate(consistent_instances):
                if cons and cons >= 4:
                    edge_list_instances.append(res_dic['most_freq_deps_instance'][i][ii])
                else:
                    edge_list_instances.append(None)

            edge_list_part = res_dic['most_freq_deps_parts'][i]

            if not consistent_parts or consistent_parts < 4 or ([] in consistent_instances) or (1 in consistent_instances) or (2 in consistent_instances) or (3 in consistent_instances):
                for ii, edge_list_instance in enumerate(edge_list_instances):
                    if edge_list_instance is None:
                        continue
                    if not consistent_instances[ii] or consistent_instances[ii] < 4:
                        continue
                    shuffled_to_orig = match_indexes(sampled_dic['shuffled_instances_steps'][i][ii], sampled_dic['steps'][i][ii][-1])
                    orig_edge_list = match_shuffled_edges(edge_list_instance, shuffled_to_orig)
                    complete_edge_list = generate_edge_list(tuple(orig_edge_list), instance_step_nums[ii])
                    plan_type = find_plan_type(orig_edge_list, instance_step_nums[ii])
                    ve = len(complete_edge_list)+instance_step_nums[ii]+2
                    adjacency_list, adjacency_matrix, (values, row_indices, col_indices) = edge_list_to_adjacency_structures(complete_edge_list)
                    fin_timedelta_dic, fin_time_dic = get_fin_time_dics(res_dic['time_dics'][i][0])
                    try:
                        all_paths, shortest_path_len, longest_path_len, path_count = get_all_paths(adjacency_list, 'START', 'END')
                    except Exception:
                        # exclude cycles
                        continue
                    all_corr_paths, task_time = generate_task_time(all_paths, fin_timedelta_dic)
                    fin_res_dic['titles'].append(sampled_dic['titles'][i])
                    fin_res_dic['multi_methods'].append(sampled_dic['multi_methods'][i])
                    fin_res_dic['multi_parts'].append(sampled_dic['multi_parts'][i])
                    fin_res_dic['steps'].append(sampled_dic['steps'][i][ii][-1])
                    fin_res_dic['step_nums'].append(instance_step_nums[ii])
                    fin_res_dic['sub_titles'].append(sampled_dic['steps'][i][ii][0])
                    fin_res_dic['orig_idx'].append(sampled_dic['orig_idx'][i])
                    fin_res_dic['edge_list'].append(complete_edge_list)
                    fin_res_dic['step_deps'].append(orig_edge_list)
                    fin_res_dic['adjacency_matrix'].append(adjacency_matrix)
                    fin_res_dic['csr'].append((values, row_indices, col_indices))
                    fin_res_dic['adjacency_list'].append(adjacency_list)
                    fin_res_dic['time_dic'].append(fin_time_dic)
                    fin_res_dic['timedelta_dic'].append(fin_timedelta_dic)
                    fin_res_dic['longest_path_len'].append(longest_path_len)
                    fin_res_dic['shortest_path_len'].append(shortest_path_len)
                    fin_res_dic['path_count'].append(path_count)
                    fin_res_dic['all_paths'].append(all_paths)
                    fin_res_dic['ve_count'].append(ve)
                    fin_res_dic['plan_types'].append(plan_type)
                    fin_res_dic['all_corr_paths'].append(all_corr_paths)
                    fin_res_dic['task_time'].append(task_time) 
            else:
                shuffled_to_orig_part = match_indexes(sampled_dic['shuffled_part_steps'][i], [step[0] for step in sampled_dic['steps'][i]])
                orig_edge_list_part = match_shuffled_edges(edge_list_part, shuffled_to_orig_part)
                shuffled_to_orig_instances = [match_indexes(sampled_dic['shuffled_instances_steps'][i][ii], sampled_dic['steps'][i][ii][-1]) for ii in range(len(sampled_dic['shuffled_instances_steps'][i]))]
                orig_edge_list_instances = [match_shuffled_edges(edge_list_instances[ii], shuffled_to_orig_instances[ii]) for ii in range(len(edge_list_instances))]
                all_edge_list, complete_edge_list = merge_all_graphs(orig_edge_list_instances,
                                                                     orig_edge_list_part,
                                                                     instance_step_nums)
                plan_type = find_plan_type(all_edge_list, sum(instance_step_nums))
                ve = len(complete_edge_list)+sum(instance_step_nums)+2
                adjacency_list, adjacency_matrix, (values, row_indices, col_indices) = edge_list_to_adjacency_structures(complete_edge_list)
                fin_timedelta_dic, fin_time_dic = get_fin_time_dics(res_dic['time_dics'][i][0])
                all_paths, shortest_path_len, longest_path_len, path_count = get_all_paths(adjacency_list, 'START', 'END')
                all_corr_paths, task_time = generate_task_time(all_paths, fin_timedelta_dic)
                fin_res_dic['titles'].append(sampled_dic['titles'][i])
                fin_res_dic['multi_methods'].append(sampled_dic['multi_methods'][i])
                fin_res_dic['multi_parts'].append(sampled_dic['multi_parts'][i])
                fin_res_dic['steps'].append([step for steps in sampled_dic['steps'][i] for step in steps[-1]])
                fin_res_dic['step_nums'].append(sum(instance_step_nums))
                fin_res_dic['sub_titles'].append([steps[0] for steps in sampled_dic['steps'][i]])
                fin_res_dic['orig_idx'].append(sampled_dic['orig_idx'][i])
                fin_res_dic['edge_list'].append(complete_edge_list)
                fin_res_dic['step_deps'].append(all_edge_list)
                fin_res_dic['adjacency_matrix'].append(adjacency_matrix)
                fin_res_dic['csr'].append((values, row_indices, col_indices))
                fin_res_dic['adjacency_list'].append(adjacency_list)
                fin_res_dic['time_dic'].append(fin_time_dic)
                fin_res_dic['timedelta_dic'].append(fin_timedelta_dic)
                fin_res_dic['longest_path_len'].append(longest_path_len)
                fin_res_dic['shortest_path_len'].append(shortest_path_len)
                fin_res_dic['path_count'].append(path_count)
                fin_res_dic['all_paths'].append(all_paths)
                fin_res_dic['ve_count'].append(ve)
                fin_res_dic['plan_types'].append(plan_type)
                fin_res_dic['all_corr_paths'].append(all_corr_paths)
                fin_res_dic['task_time'].append(task_time)

    formatted_proscript = {'titles': list(),
                           'steps': list(),
                           'step_nums': list(),
                           'orig_idx': list(),
                           'edge_list': list(),
                           'step_deps': list(),
                           'adjacency_matrix': list(),
                           'csr': list(),
                           'adjacency_list': list(),
                           'time_dic': list(),
                           'timedelta_dic': list(),
                           've_count': list(),
                           'longest_path_len': list(),
                           'shortest_path_len': list(),
                           'path_count': list(),
                           'all_paths': list(),
                           'plan_types': list(),
                           'all_corr_paths': list(),
                           'task_time': list()}

    for i in range(len(proscript)):
        step_num = len(proscript[i]['events_minutes'])
        if proscript[i]['events'][str(step_num)] != 'NONE':
            print(i)
            continue
        scenario = proscript[i]['scenario']
        temp_steps = [step.split(': ')[1] for step in proscript[i]['flatten_output_for_script_generation'].split(';')[:step_num+2]]
        temp_steps[0] = 'NONE'
        try:
            shuffled_to_orig = match_indexes([proscript[i]['events'][str(s)] for s in range(len(proscript[i]['events']))], temp_steps)
        except Exception:
            print('index match error', i)
            continue
        deps = '; '.join(proscript[i]['flatten_output_for_script_generation'].split(';')[step_num+2:])

        temp_time_dic = {str(shuffled_to_orig[int(key)]): val for key, val in proscript[i]['events_minutes'].items()}
        aux_idx = temp_steps.index(scenario)
        orig_time_dic = dict()
        for k, v in temp_time_dic.items():
            if int(k) > aux_idx:
                orig_time_dic[str(int(k)-1)] = temp_time_dic[k]
            else:
                orig_time_dic[k] = temp_time_dic[k]

        all_deps = get_all_dependencies_proscript(deps, aux_idx)
        edge_list = generate_edge_list(tuple(all_deps), step_num)
        adjacency_list, adjacency_matrix, (values, row_indices, col_indices) = edge_list_to_adjacency_structures(edge_list)
        try:
            all_paths, shortest_path_len, longest_path_len, path_count = get_all_paths(adjacency_list, 'START', 'END')
        except Exception:
            print('dep error', i)
            print(deps, all_deps)
            continue
        formatted_proscript['titles'].append(f'How to {scenario}')
        formatted_proscript['steps'].append(temp_steps[1:aux_idx]+temp_steps[aux_idx+1:])
        formatted_proscript['step_nums'].append(step_num)
        formatted_proscript['orig_idx'].append(i)
        formatted_proscript['step_deps'].append(all_deps)
        formatted_proscript['edge_list'].append(edge_list)
        formatted_proscript['adjacency_matrix'].append(adjacency_matrix)
        formatted_proscript['csr'].append((values, row_indices, col_indices))
        formatted_proscript['adjacency_list'].append(adjacency_list)
        time_dic = get_time_per_step_proscript(orig_time_dic)
        timedelta_dic = get_timedelta_interval(time_dic)[0]
        formatted_proscript['time_dic'].append(time_dic)
        formatted_proscript['timedelta_dic'].append(timedelta_dic)
        ve = len(edge_list)+step_num+2
        formatted_proscript['ve_count'].append(ve)
        formatted_proscript['longest_path_len'].append(longest_path_len)
        formatted_proscript['shortest_path_len'].append(shortest_path_len)
        formatted_proscript['path_count'].append(path_count)
        formatted_proscript['all_paths'].append(all_paths)
        plan_type = find_plan_type(all_deps, step_num)
        formatted_proscript['plan_types'].append(plan_type)
        all_corr_paths, task_time = generate_task_time(all_paths, timedelta_dic)
        formatted_proscript['all_corr_paths'].append(all_corr_paths)
        formatted_proscript['task_time'].append(task_time)

    for dic in [fin_res_dic, formatted_proscript]:
        dic['prompts'] = dict()

        for ordering_template in nl_prompt_templates:
            dic['prompts'][ordering_template] = {'vanilla_prompts': list(),
                                                                'vanilla_combined_prompts': list(),
                                                                'cot_prompts': list(),
                                                                'cot_combined_prompts': list(),
                                                                'edge_list_prompts': list(),
                                                                'adjacency_list_prompts': list(),
                                                                'adjacency_matrix_prompts': list(),
                                                                'csr_prompts': list(),
                                                                'edge_list_cot_prompts': list(),
                                                                'adjacency_list_cot_prompts': list(),
                                                                'adjacency_matrix_cot_prompts': list(),
                                                                'csr_cot_prompts': list(),
                                                                'bag_prompts': list(),
                                                                'bag_cot_prompts': list(),
                                                                'edge_list_combined_prompts': list(),
                                                                'adjacency_list_combined_prompts': list(),
                                                                'adjacency_matrix_combined_prompts': list(),
                                                                'csr_combined_prompts': list(),
                                                                'edge_list_cot_combined_prompts': list(),
                                                                'adjacency_list_cot_combined_prompts': list(),
                                                                'adjacency_matrix_cot_combined_prompts': list(),
                                                                'csr_cot_combined_prompts': list(),
                                                                'bag_combined_prompts': list(),
                                                                'bag_cot_combined_prompts': list(),}
            for i in range(len(dic['titles'])):
                for combined in ['', '_combined']:
                    if combined:
                        adjacency_list = dic['adjacency_list'][i]
                    else:
                        adjacency_list = None
                    dic['prompts'][ordering_template][f'vanilla{combined}_prompts'].append(generate_prompt(dic['titles'][i],
                                                                                                            dic['steps'][i],
                                                                                                            dic['time_dic'][i],
                                                                                                            dic['step_deps'][i],
                                                                                                            adjacency_list=adjacency_list,
                                                                                                            ordering_template=ordering_template))
                    dic['prompts'][ordering_template][f'cot{combined}_prompts'].append(generate_prompt(dic['titles'][i],
                                                                                                            dic['steps'][i],
                                                                                                            dic['time_dic'][i],
                                                                                                            dic['step_deps'][i],
                                                                                                            cot=True,
                                                                                                            adjacency_list=adjacency_list,
                                                                                                            ordering_template=ordering_template))
                    dic['prompts'][ordering_template][f'bag{combined}_prompts'].append(generate_prompt(dic['titles'][i],
                                                                                                        dic['steps'][i],
                                                                                                        dic['time_dic'][i],
                                                                                                        dic['step_deps'][i],
                                                                                                        bag=True,
                                                                                                        adjacency_list=adjacency_list,
                                                                                                        ordering_template=ordering_template))
                    dic['prompts'][ordering_template][f'bag_cot{combined}_prompts'].append(generate_prompt(dic['titles'][i],
                                                                                                            dic['steps'][i],
                                                                                                            dic['time_dic'][i],
                                                                                                            dic['step_deps'][i],
                                                                                                            bag=True,
                                                                                                            cot=True,
                                                                                                            adjacency_list=adjacency_list,
                                                                                                            ordering_template=ordering_template))
                    for graph_type in ['edge_list', 'adjacency_list', 'adjacency_matrix', 'csr']:
                        dic['prompts'][ordering_template][f'{graph_type}{combined}_prompts'].append(generate_prompt(dic['titles'][i],
                                                                                                                    dic['steps'][i],
                                                                                                                    dic['time_dic'][i],
                                                                                                                    dic['step_deps'][i],
                                                                                                                    dic[graph_type][i],
                                                                                                                    graph_type,
                                                                                                                    adjacency_list=adjacency_list,
                                                                                                                    ordering_template=ordering_template))
                        dic['prompts'][ordering_template][f'{graph_type}_cot{combined}_prompts'].append(generate_prompt(dic['titles'][i],
                                                                                                                        dic['steps'][i],
                                                                                                                        dic['time_dic'][i],
                                                                                                                        dic['step_deps'][i],
                                                                                                                        dic[graph_type][i],
                                                                                                                        graph_type,
                                                                                                                        cot=True,
                                                                                                                        adjacency_list=adjacency_list,
                                                                                                                        ordering_template=ordering_template))
    aligned_res_dic = {key: list() for key in fin_res_dic.keys() if key != 'prompts'}

    for i in range(len(fin_res_dic['titles'])):
        if len(fin_res_dic['time_dic'][i])!=len(fin_res_dic['steps'][i]):
            continue
        for key, val in aligned_res_dic.items():
            aligned_res_dic[key].append(fin_res_dic[key][i])

    aligned_res_dic['prompts'] = dict()

    for i in range(len(fin_res_dic['titles'])):
        if len(fin_res_dic['time_dic'][i]) != len(fin_res_dic['steps'][i]):
            continue
        for ordering_template in fin_res_dic['prompts'].keys():
            if ordering_template not in aligned_res_dic['prompts']:
                aligned_res_dic['prompts'][ordering_template] = dict()
            for key, val in fin_res_dic['prompts'][ordering_template].items():
                if key not in aligned_res_dic['prompts'][ordering_template]:
                    aligned_res_dic['prompts'][ordering_template][key] = list()
                aligned_res_dic['prompts'][ordering_template][key].append(val[i])

    benchmark = {key: list() for key in formatted_proscript.keys() if key != 'prompts'}
    benchmark['prompts'] = dict()
    for key, val in formatted_proscript.items():
        if key != 'prompts':
            benchmark[key] = val+aligned_res_dic[key]
        else:
            for template, data in formatted_proscript[key].items():
                if template not in benchmark[key]:
                    benchmark[key][template] = dict()
                for key2, val in data.items():
                    if key2 not in benchmark[key][template]:
                        benchmark[key][template][key2] = val+aligned_res_dic[key][template][key2]

    async_benchmark = {key: list() for key in formatted_proscript.keys() if key != 'prompts'}
    async_benchmark['prompts'] = dict()
    for i in range(len(formatted_proscript['titles'])):
        if formatted_proscript['plan_types'][i] == 'async':
            for key in async_benchmark.keys():
                if key != 'prompts':
                    async_benchmark[key].append(formatted_proscript[key][i])
                else:
                    for template, data in formatted_proscript[key].items():
                        if template not in async_benchmark[key]:
                            async_benchmark[key][template] = dict()
                        for key2, val in data.items():
                            if key2 not in async_benchmark[key][template]:
                                async_benchmark[key][template][key2] = list()
                            async_benchmark[key][template][key2].append(val[i])

    for i in range(len(aligned_res_dic['titles'])):
        if aligned_res_dic['plan_types'][i] == 'async':
            for key in async_benchmark.keys():
                if key != 'prompts':
                    async_benchmark[key].append(aligned_res_dic[key][i])
                else:
                    for template, data in aligned_res_dic[key].items():
                        if template not in async_benchmark[key]:
                            async_benchmark[key][template] = dict()
                        for key2, val in data.items():
                            if key2 not in async_benchmark[key][template]:
                                async_benchmark[key][template][key2] = list()
                            async_benchmark[key][template][key2].append(val[i])

    with open(f'{args.output_dir}/async_benchmark.pkl', 'wb') as f:
        pickle.dump(async_benchmark, f)

    seq_benchmark = sample_test_instances(benchmark,
                                          3,
                                          7,
                                          40,
                                          'sequential')
    para_benchmark = sample_test_instances(benchmark,
                                           3,
                                           7,
                                           40,
                                           'parallel')
    with open(f'{args.output_dir}/seq_benchmark.pkl', 'wb') as f:
        pickle.dump(seq_benchmark, f)
    with open(f'{args.output_dir}/para_benchmark.pkl', 'wb') as f:
        pickle.dump(para_benchmark, f)


if __name__ == '__main__':
    main()
