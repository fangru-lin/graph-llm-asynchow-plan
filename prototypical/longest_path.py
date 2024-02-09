import networkx as nx
import numpy as np
import random
from operator import itemgetter
import re
from datetime import timedelta
from time import sleep
from utils import queryLLM
import copy as cp 

def read_prompts(file_, tag):
    with open(file_, 'r') as f:
        s = f.read()
        start = f"<{tag}>"
        end = f"</{tag}>"
        prompt = s[s.index(start) + len(start): s.index(end)]
    assert len(prompt) > 0
    return prompt

def generate_prompt(graph, f, ftype='cot-edges'):
    query = read_prompts(f, ftype)
    prompt = cp.deepcopy(query)
    prompt = prompt.replace('@nodes@', str(graph['vertices']))
    prompt = prompt.replace('@i@', str(graph['vertices'][0]))
    prompt = prompt.replace('@j@', str(graph['vertices'][-1]))

    if ftype == 'cot-edges':
        prompt = prompt.replace('@edges@', str(graph['edges']))
    elif ftype == 'cot-adj':
        prompt = prompt.replace('@adj@', str(graph['adjacency']))
    elif ftype == 'cot-adj-list':
        prompt = prompt.replace('@adj@', str(graph['adjacency_list'])) 
    return prompt

def createGraph(num_flows=3, flow_max_length=5, graph_type='async', distribution=None):
    """
    num_flows:int, number of flows in the planning task
    flow_max_length:int, max length of each flow (min is 1)
    graph_type:str, 'async', 'parallel': if 'parallel', each flow is attached
     to the first node in the current flow, otherwise to a random node.
    distribution:list where to sample the time duration. If empty, sample from uniform
     between 1 and 10.
    """
    assert num_flows >= 0
    assert flow_max_length > 0
    edges = []
    vertices = []
    # Create the flows
    ctr = 1
    flows, terminals = [], [0]
    for _ in range(num_flows):
        f_length = random.randint(1, flow_max_length)
        f = [i for i in range(ctr, f_length+ctr)]
        flows.append(f)
        terminals.append(f[-1])  # last node is the one that we connect to the dst
        ctr += f_length
    vertices = [i for i in range(ctr+1)]
    # Instantiate the flows
    edges = []
    for f in flows:
        for e,e_next in zip(f, f[1:]):
            edges.append((e, e_next, -1))
    src, dst = [0], ctr
    # Compose the graph by random composition while keeping track of each terminal
    current_flow = src
    for f in flows:
        # print(f'current flow {current_flow}')
        # print(f'next flow {f}')
        if graph_type == 'async':
            v = random.choice(current_flow)   
        elif graph_type == 'parallel':
            v = current_flow[0]
        else:
            raise Exception(f'{graph_type} is not a supported value for graph_type.')     
        # print(f'Attaching {v} to node {f[0]} in the current flow.')
        edges.append((v, f[0], -1))
        if v in terminals:
            # print(f'Removing {v} from terminals ({terminals}) (if present).')
            terminals.remove(v)
            # print(f'Terminals after removal: {terminals}')
        current_flow = current_flow + f
    # Connect all the terminals to the dst node
    for t in terminals:
        edges.append((t, dst, -1))
    # Assign weights to each edge. Nodes with shared parent get the same weight
    edges = sorted(edges, key=itemgetter(0))
    edges = [list(e) for e in edges]
    i = 0
    while i<len(edges):
        if distribution is not None:
            if edges[i][0]==0:
                # print(f"Setting {edges[i]} weight to 1")
                w = 1
            else:
                w = random.choice(distribution)
        else:
            w = random.randint(1, 10)
        # print("Current edge:", edges[i])
        edges[i][2] = w
        j = i+1
        while True:
            if j<len(edges) and edges[i][0]==edges[j][0]:
                edges[j][2] = w
                j += 1
            else:
                break
        i = j
    return vertices, edges

def parse_time_per_step(time_span):
    time = re.findall(r'\d+', time_span)[0]
    unit = re.findall(r'\b[a-z]+\b', time_span)[-1].strip()
    if unit in ['year', 'years', 'y']:
        delta = [timedelta(days = int(time)*364), timedelta(days = int(time)*366)]
    elif unit in ['month', 'months', 'm']:
        # define a loose range for month
        # match other units in same format
        delta = [timedelta(days = int(time)*28), timedelta(days = int(time)*31)]
    elif unit in ['week', 'weeks', 'w']:
        delta = [timedelta(weeks = int(time)), timedelta(weeks = int(time))]
    elif unit in ['day', 'days', 'd']:
        delta = [timedelta(days = int(time)), timedelta(days = int(time))]
    elif unit in ['hour', 'hours', 'h']:
        delta = [timedelta(hours = int(time)), timedelta(hours = int(time))]
    elif unit in ['minute', 'min', 'minutes', 'mins']:
        delta = [timedelta(minutes = int(time)), timedelta(minutes = int(time))]
    elif unit in ['second', 'sec', 'seconds', 'secs']:
        delta = [timedelta(seconds = int(time)), timedelta(seconds = int(time))]
    else:
        raise ValueError(f'unit not found: {time_span}')
    return delta

def generate_adjlist_with_all_edges(G, delimiter=' '):
     for s, nbrs in G.adjacency():
        line = str(s) + delimiter
        for t, _ in nbrs.items():
                line += str(t) + delimiter
        yield line[: -len(delimiter)]

def longest_simple_paths(graph, source, target):
    longest_path = []
    longest_path_length = 0
    adjacency = nx.adjacency_matrix(G).todense()
    length = lambda x: sum([adjacency[i][j] for i,j in zip(x[:-1], x[1:])])
    all_simple_paths = nx.all_simple_paths(graph, source=source, target=target)
    # print([p for p in all_simple_paths])
    for path in all_simple_paths:
        l = length(path)
        if l > longest_path_length:
            longest_path_length = l
            longest_path = path
    return longest_path, longest_path_length

# Init vars
model_name = 'gpt35'
config_file = ('./config-gpt4.json' if model_name=='gpt4' else './config.json')
prompt_method = 'cot-adj'
sleep_time = 15
max_graphs_per_bin = 50
min_complexity, max_complexity = 10, 40

# List of durations
data = np.load('./data/ourbenchmark.pkl', allow_pickle=True)
time_distribution = []
for td in data['time_dic']:
    time_distribution.extend([t for t in td.values()])

ptime = lambda x: max(parse_time_per_step(x)[0].seconds//60, 1)
time_distribution = [ptime(t) for t in time_distribution]

graphs = {}
for n in range(20000):
    num_flows = random.randint(1, 8)
    flow_max_length = random.randint(1, 8)
    graph_type = ('async' if flow_max_length>1 else 'parallel')
    vertices, edges = createGraph(num_flows=num_flows, flow_max_length=flow_max_length,\
                                   graph_type=graph_type, distribution=time_distribution)
    src, dst = 0, len(vertices)-1
    G = nx.DiGraph()
    G.add_nodes_from(np.array(vertices))
    G.add_weighted_edges_from(edges)
    labels = nx.get_edge_attributes(G, 'weight')
    # Count |V| + |E| and assign it to the correct 'bucket'
    complexity = len(vertices) + len(edges)
    if complexity not in graphs.keys():
        graphs[complexity] = []

    if len(graphs[complexity]) < max_graphs_per_bin and (min_complexity <= complexity <= max_complexity):
        adjacency_matrix = nx.adjacency_matrix(G).todense()
        adjacency_list = [[int(j) for j in i.split(' ')] for i in generate_adjlist_with_all_edges(G)]
        adjacency_dict = [{adj[0]: ([(a,adjacency_matrix[adj[0]][a]) for a in adj[1:]])} for adj in adjacency_list]

        longest_path, longest_path_length = longest_simple_paths(G, src, dst)

        # Create the graph dictionary
        g = {}
        g['graph'] = G 
        g['vertices'] = vertices
        g['edges'] = edges
        g['longest_path'] = longest_path
        g['longest_path_len'] = longest_path_length
        g['adjacency'] = nx.adjacency_matrix(G).todense()
        g['adjacency_list'] = adjacency_dict
        g['csr'] = nx.to_scipy_sparse_array(G).todense()
        graphs[complexity].extend([g])
        # print(G.edges.data())
        # print(g['adjacency'])
        # print(src, dst)
        # print(g['longest_path'])
        # print(g['longest_path_len'])
        # print()

filtered_graphs = {}
for k in sorted(graphs.keys()):
    if len(graphs[k]) > 0:
        filtered_graphs[k] = graphs[k]
        # print(k, len(graphs[k]))

graphs = filtered_graphs
for k in graphs.keys():
    next_batch = graphs[k]
    correct, total = 0, 0
    for graph in next_batch:
        prompt = generate_prompt(graph, './prompt.txt', prompt_method)
        response = queryLLM(prompt, config_file, model_name)
        print(f"<prompt> {prompt} </prompt>")
        print(f"<response> {response} </response>")

        # Parse result
        if response is not None:
            y_hat = re.search(r'<result>(.*)</result>', response)
            try:    
                y_hat = y_hat.group(1)
                if int(y_hat.strip()) == graph['longest_path_len']:
                    correct += 1
            except:
                pass
            total += 1
        else:
            sleep(sleep_time + 30)  # wait as the API are not responding

        sleep(sleep_time)  # normal wait time
        # Logs
        with open(f'results-{model_name}-{prompt_method}.txt', 'a+') as file_:
            file_.write(f"\n<prompt>\n{prompt}\n</prompt>\n")
            file_.write(f"<response>\n{response}\n</response>\n")
            file_.write(f"<ground-truth>{graph['longest_path_len']}</ground-truth>\n")
        
    with open(f'results-{model_name}-{prompt_method}-accuracy.txt', 'a+') as file_:
        file_.write(f"\nAccuracy onO(|V|+|E|)=<complexity>{k}</complexity> on {len(next_batch)} instances: <accuracy>{correct/total}</accuracy>\n")
