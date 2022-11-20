from unicodedata import category
from data.graph import Graph
from utils.data import load_edgelist
from utils.random_walk import *
import yaml
import pickle
import numpy as np
import time

if __name__ == "__main__":

    start = time.time()

    config = yaml.safe_load(open("config.yaml"))
    seed = config['seed']
    data_config = config['data']
    rng = np.random.default_rng(seed)

    graph = Graph(is_undirected = data_config['is_undirected'])
    # graph = load_edgelist(graph, data_config['all_category_names'], f'{data_config["dataset_dir_test"]}/set1_D_D.csv', 'disease', 'disease', verbose = True)
    # graph = load_edgelist(graph, data_config['all_category_names'], f'{data_config["dataset_dir_test"]}/set1_G_D.csv', 'gene', 'disease',  verbose = True)
    # graph = load_edgelist(graph, data_config['all_category_names'], f'{data_config["dataset_dir_test"]}/set1_G_G.csv', 'gene', 'gene', verbose = True)
    
    graph = load_edgelist(graph, data_config['all_category_names'], f'{data_config["dataset_dir_main"]}/disease_disease.csv', 'disease', 'disease', verbose = True)
    graph = load_edgelist(graph, data_config['all_category_names'], f'{data_config["dataset_dir_main"]}/disease_gene.csv', 'disease', 'gene',  verbose = True)
    graph = load_edgelist(graph, data_config['all_category_names'], f'{data_config["dataset_dir_main"]}/final_STRING.csv', 'gene', 'gene', verbose = True)

    graph.preprocess_graph(verbose = True)

    # Displays graph details
    graph.get_graph_details()

    candidate_metapaths = [['gene', 'gene', 'disease', 'disease'],
                       ['gene', 'gene', 'disease'], 
                       ['gene', 'disease', 'disease'],
                       ['gene', 'disease', 'gene', 'disease']
                      ]

    total_repeats = data_config['total_repeats']
    selected_metapath = select_metapath_from_candidates(graph, total_repeats, candidate_metapaths, rng)
    print(f'Selected Metapath: {selected_metapath}')
    print()


    node_list_per_type = defaultdict(set) 

    print('Writing random walks to the file...')
    f = open("corpus.txt", "w")
    node_indices = list(graph.nodes())
    for iteration in range(total_repeats):
        rng.shuffle(node_indices)
        for node_idx in node_indices:
            if graph.index_to_node[node_idx].category == selected_metapath[0]:
                path, _ = graph.random_walk(selected_metapath, visit_cnt = None, rng = rng, start = node_idx)
                
                for i, v in enumerate(path):
                    node_name, node_category = graph.index_to_node[v].name, graph.index_to_node[v].category
                    path[i] = node_name
                    node_list_per_type[node_category].add(node_name)
                
                line = ' '.join(path)
                f.write(f'{line}\n')

    f.close()
    print('Corpus generation done.')

    node_list_per_type = dict(node_list_per_type)
    for key in node_list_per_type.keys():
        node_list_per_type[key] = list(node_list_per_type[key])

    # print(node_list_per_type)

    with open('node_list_per_type.pkl', 'wb') as f:
        pickle.dump(node_list_per_type, f)   

    end = time.time()
    print(f"Total time elapsed: {end-start}")