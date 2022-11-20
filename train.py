from unittest import skip
from data.input_data import InputData
import numpy
from model.skipgram_model import SkipGramModel
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import sys
import argparse
import yaml
import numpy as np

def data_loader(input_data, batch_size):
    while True:
        pos_pairs = input_data.get_batch_pairs(batch_size)
        if len(pos_pairs) == 0:
            break

        yield pos_pairs

def train(input_data,
    skip_gram_model,
    use_cuda,
    batch_size=64,
    num_epochs=5,
    learning_rate=0.025,
    neg_sample_cnt=5,
    **kwargs
    ):

    if use_cuda:
        skip_gram_model.cuda()
    
    optimizer = optim.SGD(
        skip_gram_model.parameters(), lr=learning_rate)

    # skip_gram_model.save_embedding(
    #     input_data.id2word, 'begin_embedding.txt', use_cuda)
    
    for i in range(num_epochs):
        progress_bar = tqdm(data_loader(input_data, batch_size))
        for pos_pairs in progress_bar:
            neg_v = input_data.get_neg_v_neg_sampling(pos_pairs, neg_sample_cnt)
            pos_u = [pair[0] for pair in pos_pairs]
            pos_v = [pair[1] for pair in pos_pairs]

            pos_u = torch.LongTensor(pos_u)
            pos_v = torch.LongTensor(pos_v)
            neg_v = torch.LongTensor(neg_v)

            if use_cuda:
                pos_u = pos_u.cuda()
                pos_v = pos_v.cuda()
                neg_v = neg_v.cuda()

            optimizer.zero_grad()
            loss = skip_gram_model.forward(pos_u, pos_v, neg_v)
            loss.backward()
            optimizer.step()

            progress_bar.set_description(f"Loss: {loss.data.item():0.8f}, lr: {optimizer.param_groups[0]['lr']:0.6f}")
            
            # if i * batch_size % 100000 == 0:
            #     lr = learning_rate * (1.0 - 1.0 * i / batch_count)
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] = lr
    
    return skip_gram_model

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=str, dest = "input_file", default='corpus.txt', help='input file path')
    parser.add_argument('--output-file', type=str,dest = "output_file", default='node_representations.txt', help='ouput file path')

    arguments = parser.parse_args()
    # print(arguments)
    return arguments


if __name__ == "__main__":
    args = parse_arguments()
    config = yaml.safe_load(open("config.yaml"))
    seed = config['seed']
    model_config = config['model']
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(seed)

    input_data = InputData(args.input_file)
    skip_gram_model = SkipGramModel(input_data.node_count, model_config['embedding_dim'])
    
    skip_gram_model = train(input_data, skip_gram_model, use_cuda, **model_config)
    skip_gram_model.save_embedding(input_data.index_to_node, args.output_file, use_cuda)

