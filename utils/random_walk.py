from collections import defaultdict
import numpy as np

def calculate_random_walk_based_measure(graph, total_repeats, meta_path, rng=np.random.default_rng(1234)):
  '''
    Calculates the random walk based measure for choosing the best metapath.
    This function returns the count of isolated walking nodes for a given metapath.
  '''

  node_indices = list(graph.nodes())
  visit_cnt = defaultdict(int)
  print("\nRandom walks following this metapath {}:\n".format(meta_path))

  for iteration in range(total_repeats):
    rng.shuffle(node_indices)
    for node_idx in node_indices:
      if graph.index_to_node[node_idx].category == meta_path[0]:
        path, visit_cnt = graph.random_walk(meta_path, visit_cnt, rng, start = node_idx)
        path = [graph.index_to_node[v].name for v in path]
        # print(path)

    # print()

  isolated_walking_node_count = 0
  for node_idx in node_indices:
    if visit_cnt[node_idx] <= total_repeats:  
      isolated_walking_node_count += 1
  
  return isolated_walking_node_count

def select_metapath_from_candidates(graph, total_repeats, candidate_metapaths, rng=np.random.default_rng(1234)):
  '''
    This functions return the best metapath from given candidates.
    The metapath with the least isolated walking nodes is selected for guiding the random walk.
  '''
  
  best_meta_path_idx = 0
  min_score = calculate_random_walk_based_measure(graph, total_repeats, candidate_metapaths[0], rng)
  print('Meta path:', candidate_metapaths[0], ' -> Score:', min_score)

  for i in range(1, len(candidate_metapaths)):
    score = calculate_random_walk_based_measure(graph, total_repeats, candidate_metapaths[i], rng)
    print('Meta path:', candidate_metapaths[i], ' -> Score:', score)
    if score < min_score:
      min_score = score
      best_meta_path_idx = i

  return candidate_metapaths[best_meta_path_idx]    