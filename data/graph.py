from collections import defaultdict
import random
import numpy as np

class Graph:  
  def __init__(self, is_undirected=True):
    self.adj_info = defaultdict(dict)
    self.index_to_node = dict()
    self.name_to_index = dict()
    self.is_undirected = is_undirected

  def nodes(self):
    '''
    Returns the number of nodes in the graph.
    '''
    return self.adj_info.keys()

  def preprocess_graph(self, verbose = False):

    '''
    Preprocess the graph in such a way such that we can effieciently retrieve each and every information from the group.
    
    Initial formation of the graph: 
    
    graph = {
      node_id: {
        'gene' : [(adj_node_id, edge_weight), (adj_node_id, edge_weight), .... ],
        'disease: [(adj_node_id, edge_weight), (adj_node_id, edge_weight), .... ]
      },

      node_id: {
        'gene' : [(adj_node_id, edge_weight), (adj_node_id, edge_weight), .... ],
        'disease: [(adj_node_id, edge_weight), (adj_node_id, edge_weight), .... ]
      },

      ....
    } 

    After preprocessing:

    N.B. : Here prob is the probalility of choosing a adjacent node from current node following the selected metapath.
           prob is the normalized edge weights such that the summation of the probability of adjacent nodes for a give node equal to 1.  
    
    graph = {
      node_id: {
        'gene' : {
            'node': [adj_node_id, adj_node_id, ... ],
            'weight': [ edge_weight, edge_weight, ... ],
            'prob': [edge_prob, edge_prob, ...]
         },
        'disease: {
            'node': [adj_node_id, adj_node_id, ... ],
            'weight': [ edge_weight, edge_weight, ... ],
            'prob': [edge_prob, edge_prob, ...]
        }
      },

      node_id: {
        ....
      },

      ....
    }       


    '''

    print("Preprocesing begins...")

    edge_counts = defaultdict(int)
    duplicate_edge_counts = defaultdict(int)

    for node_idx in self.nodes():
      node_category = self.index_to_node[node_idx].category
      adj_dict = self.adj_info[node_idx]
      adj_category_names = adj_dict.keys()

      for adj_node_category in adj_category_names:
        adj_details = {
            'node': [],
            'weight': [],
            'prob': []
        }  

        adj_list = list(set(adj_dict[adj_node_category])) ## wrapping inside a set to remove duplicates
        
        ## Counting the number of each type of edges in the graph after removing duplicate edges
        edge_name = f'{node_category} - {adj_node_category}' if not self.is_undirected else f'{min(node_category, adj_node_category)} - {max(node_category, adj_node_category)}'
        edge_counts[edge_name] += len(adj_list)
        
        ## Counting the number of removed duplicate edges in the graph
        duplicate_edge_counts[edge_name] += len(adj_dict[adj_node_category]) - len(adj_list)

        sum_of_weights = 0.0
        for adj_node, weight in adj_list:
          adj_details['node'].append(adj_node)
          adj_details['weight'].append(weight)
          adj_details['prob'].append(weight)
          sum_of_weights += weight
      
        adj_details['prob'] = list(np.array(adj_details['prob'])/sum_of_weights)
        self.adj_info[node_idx][adj_node_category] = adj_details # updating adjaceny list of tuples using a dictionary

        del adj_details, adj_list, sum_of_weights

      del adj_dict, adj_category_names

    if verbose:
      print('Removed Duplicate Edge Count Information: ')
      for category in duplicate_edge_counts.keys():
        print(f'{category} : {int(duplicate_edge_counts[category]/2) if self.is_undirected else duplicate_edge_counts[category]}')

      print('\nEdge Count Information (after removing duplicates): ')
      for category in edge_counts.keys():
        print(f'{category} : {int(edge_counts[category]/2) if self.is_undirected else edge_counts[category]}')  

  def get_graph_details(self):
    '''
    This function gives overall details of the constructed graph such as-
      - Count of nodes for each type of entities.
      - Count of each type of edges in the graph 
    '''

    node_indices = list(self.nodes())
    node_counts = defaultdict(int)
    edge_counts = defaultdict(int)

    for node_idx in node_indices:
      node_category = self.index_to_node[node_idx].category
      node_counts[node_category] += 1

      for adj_node_category in self.adj_info[node_idx].keys():
        # if undirected then edges gene-disease and disease-gene are same. 
        # So, count of both gene-disease and disease-gene edges are added to same edge_counts[disease-gene] variable.
        edge_name = f'{node_category} - {adj_node_category}' if not self.is_undirected else f'{min(node_category, adj_node_category)} - {max(node_category, adj_node_category)}'
        edge_counts[edge_name] += len(self.adj_info[node_idx][adj_node_category]['node'])


    print('Node Count Information: ')
    for category in node_counts.keys():
      print(f'{category} : {node_counts[category]}')

    print('\nEdge Count Information: ')
    for category in edge_counts.keys():
      print(f'{category} : {int(edge_counts[category]/2) if self.is_undirected else edge_counts[category]}')
  

  def random_walk(self, meta_path, visit_cnt, rng=np.random.default_rng(1234), start=None):
    """ 
    Returns a truncated random walk.

    meta_path: Path to guide random walk between different entities.
    start: the start node of the random walk.
    """

    graph = self
    if start:
      path = [start]
    else:
      # Sampling is uniform w.r.t V, and not w.r.t E
      path = [rng.choice(list(graph.nodes()))]

    if visit_cnt is not None:
      visit_cnt[path[0]] += 1 

    while len(path) < len(meta_path):
      cur_node = path[-1]

      if len(graph.adj_info[cur_node][meta_path[len(path)]]['node']) > 0:
        ## rng.choice() returns a single value if size is None.
        new_node = rng.choice(graph.adj_info[cur_node][meta_path[len(path)]]['node'], size=None, p = graph.adj_info[cur_node][meta_path[len(path)]]['prob'])
        path.append(new_node)

        if visit_cnt is not None:
          visit_cnt[new_node] += 1
          
      else:
        break
      
    return path, visit_cnt