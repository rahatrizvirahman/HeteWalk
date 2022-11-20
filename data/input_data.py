from operator import mod
import numpy as np
from collections import deque
import pickle

class InputData:
    """Store data for word2vec, such as node map, sampling table and so on.

    Attributes:
        node_frequency: Count of each node and sampling table
        random_walk_count: Random walk count in files.
        node_count: node count in files.
    """

    def __init__(self, input_file_path, use_heuristic = False):
        self.input_file_path = input_file_path
        self.total_nodes = 0 # Number of total nodes in the whole file
        self.random_walk_count = 0
        self.node_pair_cache = deque()

        f = open('./node_list_per_type.pkl', 'rb')
        self.node_list_per_type = pickle.load(f)
        f.close()
        
        self.node_to_index = dict()
        self.index_to_node = dict()
        node_frequency = dict()
        node_count = 0

        input_file = open(self.input_file_path)
        for line in input_file:
            self.random_walk_count += 1
            line = line.strip().split(' ')
            self.total_nodes += len(line)
            for node in line:
                if node not in node_frequency:
                    self.node_to_index[node] = node_count
                    self.index_to_node[node_count] = node
                    
                    node_frequency[node] = 1
                    node_count += 1
                else:
                    node_frequency[node] += 1
        
        input_file.close()
        assert node_count == len(node_frequency), "node_frequency length must be equal of the node_count"

        self.node_frequency = node_frequency        
        self.node_count = node_count
        del node_frequency, node_count

        # Input file is opened again for generating batched positive samples to train the model.
        self.input_file = open(self.input_file_path)
        self.cur_line = 0

        # create a dictionary which maps node_index to its category
        node_index_to_category = dict()
        for category in self.node_list_per_type.keys():
            node_list = self.node_list_per_type[category]
            for node in node_list:
                node_index_to_category[self.node_to_index[node]] = category

        self.node_index_to_category = node_index_to_category
        del node_index_to_category

        self.neg_sampling_prob = None
        if use_heuristic is True:
            self.neg_sampling_prob = self.calculate_neg_sampling_prob()

        print('Distinct Node Count: %d' % self.node_count)
        print('Number of total nodes in the file: %d' % (self.total_nodes))

    def calculate_neg_sampling_prob(self):
        neg_sampling_prob = {}

        for category in self.node_list_per_type.keys():
            node_list = self.node_list_per_type[category]
            node_pow_frequencies = np.array([self.node_frequency[node] ** 0.75 for node in node_list])
            neg_sampling_prob[category] = node_pow_frequencies/np.sum(node_pow_frequencies)

        return neg_sampling_prob

    def get_batch_pairs(self, batch_size, window_size=None):

        while len(self.node_pair_cache) < batch_size: 
            random_walk = self.input_file.readline()

            if random_walk is None or random_walk == '':
                if len(self.node_pair_cache) == 0:
                    # File ended and there is no remaining node pairs in cache.
                    # Opening file again for next epoch
                    self.input_file = open(self.input_file_path)                
                
                break

            node_ids = []
            for node in random_walk.strip().split(' '):
                node_ids.append(self.node_to_index[node])

            if window_size is None:
                for i in range(len(node_ids)-1):
                    for j in range(i+1, len(node_ids)):
                        u = node_ids[i]
                        v = node_ids[j]
                        if i == j:
                            continue

                        self.node_pair_cache.append((u, v))
            else:
                for i in range(len(node_ids)):
                    for j in range(max(i - window_size, 0), min(i + window_size + 1, len(node_ids))):
                        u = node_ids[i]
                        v = node_ids[j]
                        if i == j:
                            continue

                        self.node_pair_cache.append((u, v))

        batch_pairs = []
        for _ in range(batch_size):
            if len(self.node_pair_cache) == 0:
                break 
            batch_pairs.append(self.node_pair_cache.popleft())

        return batch_pairs

    def get_neg_v_neg_sampling(self, pos_node_pair, neg_sample_count, rng=np.random.default_rng(1234)):
        batch_neg_samples = []
        
        for i, pair in enumerate(pos_node_pair):
            v = pair[1]
            v_category = self.node_index_to_category[v]
            neg_samples = set()    

            while len(neg_samples) < neg_sample_count:
                if self.neg_sampling_prob is None:
                    neg_v = rng.choice(self.node_list_per_type[v_category])
                else:
                    neg_v = rng.choice(self.node_list_per_type[v_category], p = self.neg_sampling_prob[v_category])

                # neg_v is the name of the node not the index. 
                # Because node_list_per_type is a dictionary containing the list of node names for each category
                if self.node_to_index[neg_v] != v:
                    neg_samples.add(self.node_to_index[neg_v])

            batch_neg_samples.append(list(neg_samples))

        return batch_neg_samples


# def test():
#     a = InputData('./corpus.txt')
#     f2 = open('output.txt', mode="w")

#     while True:
#         pos_pairs = a.get_batch_pairs(16)
#         if len(pos_pairs) == 0:
#             break
#         neg_v = a.get_neg_v_neg_sampling(pos_pairs, 5)

#         f2.write("******************\n")
#         f2.write(f'{pos_pairs}\n')
#         f2.write("******************\n")
#         f2.write(f"{neg_v}\n")
#         f2.write("\n")



# if __name__ == '__main__':
#     test()
