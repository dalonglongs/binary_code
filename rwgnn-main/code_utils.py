import numpy as np

import torch
import torch.utils.data as utils

from math import ceil
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import networkx as nx
import random


def load_data(ds_name, name):
    adj = []
    features = []
    if name == './datasets/code/train.pkl':
        num = 30000
    else:
        num = 10000
    df = pd.read_pickle(name)
    for i in range(0, num):
        edg_1 = []
        edg_2 = []
        sample = df.iloc[i]
        bin = sample[ds_name]
        g = nx.read_gpickle(bin)
        for (u,v) in g.edges:
            edg_1.append(u-1)
            edg_1.append(v-1)
            edg_2.append(v-1)
            edg_2.append(u-1)
        A = csr_matrix((np.ones(2*len(g.edges())), (edg_1, edg_2)),
                       shape=(max(g.nodes()), max(g.nodes())))
        x = A.sum(axis=1)
        adj.append(A)
        features.append(x)


    return adj, features


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row,
                                          sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def generate_batches(adj_1, adj_2, features_1,features_2, device, str,  shuffle=False):

    adj_lst = list()
    features_lst = list()
    graph_indicator_lst = list()
    y_lst = list()
    if str == 'train':
        number = 30000
    else:
        number = 10000

    for i in range(0, number):
        j = np.random.randint(0, number, size=2)
        for num in j:
            if num != i:
                n_nodes_1 = adj_1[i].shape[0]
                n_nodes_2 = adj_2[num].shape[0]
                adj_batch = lil_matrix((n_nodes_1+n_nodes_2, n_nodes_1+n_nodes_2))
                features_batch = np.zeros((n_nodes_1+n_nodes_2, 1))
                adj_batch[0:n_nodes_1, 0:n_nodes_1] = adj_1[i]
                adj_batch[n_nodes_1:, n_nodes_1:] = adj_2[num]
                features_batch[0:n_nodes_1, :] = features_1[i]
                features_batch[n_nodes_1:, :] = features_2[num]
                y_batch = np.zeros(1)
                graph_indicator_batch = np.zeros(n_nodes_1+n_nodes_2)
                adj_lst.append(sparse_mx_to_torch_sparse_tensor(adj_batch).to(device))
                features_lst.append(torch.FloatTensor(features_batch).to(device))
                graph_indicator_lst.append(torch.LongTensor(graph_indicator_batch).to(device))
                y_lst.append(torch.LongTensor(y_batch).to(device))
        n_nodes_1 = adj_1[i].shape[0]
        n_nodes_2 = adj_2[i].shape[0]
        adj_batch = lil_matrix((n_nodes_1 + n_nodes_2, n_nodes_1 + n_nodes_2))
        features_batch = np.zeros((n_nodes_1 + n_nodes_2, 1))
        adj_batch[0:n_nodes_1, 0:n_nodes_1] = adj_1[i]
        adj_batch[n_nodes_1:, n_nodes_1:] = adj_2[i]
        features_batch[0:n_nodes_1, :] = features_1[i]
        features_batch[n_nodes_1:, :] = features_2[i]
        y_batch = np.ones(1)
        graph_indicator_batch = np.zeros(n_nodes_1+n_nodes_2)
        adj_lst.append(sparse_mx_to_torch_sparse_tensor(adj_batch).to(device))
        features_lst.append(torch.FloatTensor(features_batch).to(device))
        graph_indicator_lst.append(torch.LongTensor(graph_indicator_batch).to(device))
        y_lst.append(torch.LongTensor(y_batch).to(device))
    return adj_lst, features_lst, graph_indicator_lst, y_lst






def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count