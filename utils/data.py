from data.node import Node

def load_edgelist(graph, all_category_names, file_name, category1, category2, verbose = False):
  '''
   Expects a csv file in the format catergory1,category2 or catergory1,category2,weight
    if weight is not given, then default 1.0 is assigned as edge weight.

   Load a given edgelist file and insert the edges in the constructed graph
  '''

  edge_cnt = 0
  self_loop_cnt = 0

  with open(file_name) as f:
    # skip header
    line = next(f)
    print("Skipping this Header: ", line.strip())

    for line in f:
      edge = line.strip().split(',')
      if len(edge) == 2:
        x, y = edge[:2]
        edge_weight = 1.0
      elif len(edge) == 3:
        x, y, edge_weight = edge[:3]
      else:
        continue

      # Removing self loop
      if x == y:
        self_loop_cnt += 1
        continue

      if x not in graph.name_to_index:
        graph.name_to_index[x] = len(graph.name_to_index)
        graph.index_to_node[graph.name_to_index[x]] = Node(x, category1)
        graph.adj_info[graph.name_to_index[x]] = {
            category : [] for category in all_category_names
        }

      if y not in graph.name_to_index:
        graph.name_to_index[y] = len(graph.name_to_index)
        graph.index_to_node[graph.name_to_index[y]] = Node(y, category2)
        graph.adj_info[graph.name_to_index[y]] = {
            category : [] for category in all_category_names
        }
      
      graph.adj_info[graph.name_to_index[x]][category2].append((graph.name_to_index[y], edge_weight))
      if graph.is_undirected:
        graph.adj_info[graph.name_to_index[y]][category1].append((graph.name_to_index[x], edge_weight))

      edge_cnt += 1

  if verbose:
    print(f"Number of {category1}-{category2} edges(excluding self loops): {edge_cnt}")
    print(f"Number of removed {category1}-{category2} self loops: {self_loop_cnt}\n")
  
  return graph
