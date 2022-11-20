import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipGramModel(nn.Module):
    """Skip gram model of word2vec.

    Attributes:
        num_embeddings: Number of embeddings we will learn.
        embedding_dim: Embedding dimention.
        u_embedding: Embedding for center node.
        v_embedding: Embedding for neighbour nods.
    """

    def __init__(self, num_embeddings, embedding_dim):
        """Initialize model parameters.

        Declare two embedding layers and Initialize layer weights.

        Args:
            num_embeddings: Number of embeddings we will learn.
            embedding_dim: Embedding dimention.

        Returns:
            None
        """
        super(SkipGramModel, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.u_embeddings = nn.Embedding(num_embeddings, embedding_dim, sparse=True)
        self.v_embeddings = nn.Embedding(num_embeddings, embedding_dim, sparse=True)
        self.init_embeddings()

    def init_embeddings(self):
        """Initialize embedding weight like word2vec original c/c++ implementation.

        The u_embedding is initialized using a uniform distribution in [-0.5/em_size, 0.5/num_embeddings] 
        and the elements of v_embedding are initialized with zeroes.

        Returns:
            None
        """
        # Using torch.nn.init module to initiliaze the weights rather than using tensors member function uniform_
        initrange = 0.5 / self.embedding_dim
        nn.init.uniform_(self.u_embeddings.weight, -initrange, initrange)
        nn.init.zeros_(self.v_embeddings.weight)

        # print(self.u_embeddings.weight.requires_grad, self.v_embeddings.weight.requires_grad)

        # self.u_embeddings.weight.data.uniform_(-initrange, initrange)
        # self.v_embeddings.weight.data.uniform_(-0, 0)

    def forward(self, pos_u, pos_v, neg_v):
        """Forward process.

        Args:
            pos_u: list of center node indices for positive node pairs.
            pos_v: list of neighbour node indices for positive node pairs.
            neg_v: list of node indices for negative node pairs.

        Returns:
            Loss, a pytorch variable.
        """
        
        pos_u_embedding = self.u_embeddings(pos_u) 
        pos_v_embedding = self.v_embeddings(pos_v) 
        pos_score = torch.mul(pos_u_embedding, pos_v_embedding).squeeze() 
        pos_score = torch.sum(pos_score, dim=1) 
        pos_score = F.logsigmoid(pos_score) 

        neg_v_embedding = self.v_embeddings(neg_v) 
        neg_score = torch.bmm(neg_v_embedding, pos_u_embedding.unsqueeze(2)).squeeze()
        neg_score = F.logsigmoid(-1 * neg_score)
        
        return -1 * (torch.sum(pos_score)+torch.sum(neg_score))

    def save_embedding(self, index_to_node, file_path="node_representations.txt", use_cuda=True):
        """Save all embeddings to file.

        Args:
            index_to_node: map from node index to node name.
            file_path: file path.
        Returns:
            None.
        """
        if use_cuda:
            all_embeddings = self.u_embeddings.weight.cpu().data.numpy()
        else:
            all_embeddings = self.u_embeddings.weight.data.numpy()

        fout = open(file_path, 'w')
        fout.write(f'{len(index_to_node)} {self.embedding_dim}\n')
        for node_index, node in index_to_node.items():
            embedding = all_embeddings[node_index]
            embedding = ' '.join(map(lambda x: str(x), embedding))
            fout.write(f'{node} {embedding}\n')

def test():
    torch.manual_seed(1234)
    model = SkipGramModel(100, 100)
    index_to_node = dict()
    for i in range(100):
        index_to_node[i] = str(i)
    model.save_embedding(index_to_node)


if __name__ == '__main__':
    test()
