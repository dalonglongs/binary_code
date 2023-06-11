import numpy as np

import torch
import torch.utils.data as utils

from math import ceil
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import networkx as nx
import json
import random
def read(ds_name, filenaem, lable):

    with open(filenaem, 'r') as load_f:
        load_dict = json.load(load_f)
        dicts = {}
        if lable == 0:

            for i in load_dict:
                if load_dict[i][ds_name]==[]:
                    dicts[i]=[]
                else:
                    dicts[i] = load_dict[i][ds_name][0]
        elif lable == 1:

            for i in load_dict:
                if load_dict[i][ds_name] != [] and 0 < load_dict[i][ds_name][1] <= 5  :

                    dicts[i] = load_dict[i][ds_name][0]
                else:
                    dicts[i] = []
        elif lable == 2:

            for i in load_dict:
                if load_dict[i][ds_name] != [] and 5 < load_dict[i][ds_name][1] <= 20  :

                    dicts[i] = load_dict[i][ds_name][0]
                else:
                    dicts[i] = []
        elif lable == 3:

            for i in load_dict:
                if load_dict[i][ds_name] != [] and load_dict[i][ds_name][1] > 20  :

                    dicts[i] = load_dict[i][ds_name][0]
                else:
                    dicts[i] = []
        return dicts

def load_data(ds_name, name, param = 0):
    adj = []
    features = []
    label = []
    if name == './datasets/code/train.pkl':
        num = 30000
    else:
        num = 10000
    # file = read(ds_name, name, param)

    df = pd.read_pickle(name)
    for i in range(0, 10):
    # for num, i in enumerate(file):
        edg_1 = []
        edg_2 = []
        sample = df.iloc[i]
        bin = sample[ds_name]

        # if file[i] != []:
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
        label.append(1)

        features.append(x)
        # else:
        #     adj.append(-1)
        #     features.append(-1)
        #     label.append(0)


    return label, adj, features

def json_load_data(ds_name, name, param = 0):
    adj = []
    features = []
    label = []
    # if name == './datasets/code/train.pkl':
    #     num = 30000
    # else:
    #     num = 10000
    file = read(ds_name, name, param)

    # df = pd.read_pickle(name)
    # for i in range(0, num):
    for num, i in enumerate(file):
        edg_1 = []
        edg_2 = []
        # sample = df.iloc[i]
        # bin = sample[ds_name]

        if file[i] != []:
            g = nx.read_gpickle(file[i])
            for (u,v) in g.edges:
                edg_1.append(u-1)
                edg_1.append(v-1)
                edg_2.append(v-1)
                edg_2.append(u-1)
            A = csr_matrix((np.ones(2*len(g.edges())), (edg_1, edg_2)),
                           shape=(max(g.nodes()), max(g.nodes())))
            x = A.sum(axis=1)
            adj.append(A)
            label.append(1)
            if A.shape[0] !=len(x):
                print(1)
            features.append(x)
        else:
            adj.append(-1)
            features.append(-1)
            label.append(0)


    return label, adj, features


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row,
                                          sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def generate_batches(lab_1, lab_2, adj_1, adj_2, features_1, features_2, device, str,  shuffle=False):

    adj_lst_1 = list()
    features_lst_1 = list()
    adj_lst_2 = list()
    features_lst_2 = list()
    graph_indicator_lst = list()
    y_lst = list()
    # if str == 'train':
    #     number = 30000
    # else:
    #     number = 10000
    number_1 = len(lab_1)
    number_2 = len(lab_2)

    # for i in range(0, number_1):
    for i in range(0, number_1):
        j = np.random.randint(0, number_2, size=2)
        for num in j:
            if num != i and lab_2[num] == 1 and lab_1[i] == 1:
                n_nodes_1 = adj_1[i].shape[0]
                n_nodes_2 = adj_2[num].shape[0]
                adj_batch_1 = lil_matrix((n_nodes_1, n_nodes_1))
                adj_batch_2 = lil_matrix((n_nodes_2, n_nodes_2))
                features_batch_1 = np.zeros((n_nodes_1, 1))
                features_batch_2 = np.zeros((n_nodes_2, 1))
                adj_batch_1[0:n_nodes_1, 0:n_nodes_1] = adj_1[i]
                adj_batch_2[0:n_nodes_2, 0:n_nodes_2] = adj_2[num]
                features_batch_1[0:n_nodes_1, :] = features_1[i]
                features_batch_2[0:n_nodes_2:, :] = features_2[num]
                y_batch = np.array([0], dtype=np.float)
                graph_indicator_batch =  [n_nodes_1, n_nodes_2]
                adj_lst_1.append(sparse_mx_to_torch_sparse_tensor(adj_batch_1).to(device))
                features_lst_1.append(torch.FloatTensor(features_batch_1).to(device))
                adj_lst_2.append(sparse_mx_to_torch_sparse_tensor(adj_batch_2).to(device))
                features_lst_2.append(torch.FloatTensor(features_batch_2).to(device))
                graph_indicator_lst.append(torch.LongTensor(graph_indicator_batch).to(device))
                y_lst.append(torch.LongTensor(y_batch).to(device))
        if lab_2[i] == 1 and lab_1[i] == 1:
            n_nodes_1 = adj_1[i].shape[0]
            n_nodes_2 = adj_2[i].shape[0]
            adj_batch_1 = lil_matrix((n_nodes_1, n_nodes_1))
            adj_batch_2 = lil_matrix((n_nodes_2, n_nodes_2))
            features_batch_1 = np.zeros((n_nodes_1, 1))
            features_batch_2 = np.zeros((n_nodes_2, 1))
            adj_batch_1[0:n_nodes_1, 0:n_nodes_1] = adj_1[i]
            adj_batch_2[0:n_nodes_2, 0:n_nodes_2] = adj_2[i]
            features_batch_1[0:n_nodes_1, :] = features_1[i]
            features_batch_2[0:n_nodes_2, :] = features_2[i]
            y_batch = np.ones(1)
            graph_indicator_batch = [n_nodes_1, n_nodes_2]
            adj_lst_1.append(sparse_mx_to_torch_sparse_tensor(adj_batch_1).to(device))
            adj_lst_2.append(sparse_mx_to_torch_sparse_tensor(adj_batch_2).to(device))
            features_lst_1.append(torch.FloatTensor(features_batch_1).to(device))
            features_lst_2.append(torch.FloatTensor(features_batch_2).to(device))
            graph_indicator_lst.append(torch.LongTensor(graph_indicator_batch).to(device))
            y_lst.append(torch.LongTensor(y_batch).to(device))
    return adj_lst_1, adj_lst_2, features_lst_1, features_lst_2, graph_indicator_lst, y_lst



def test_generate_batches(lab_1, lab_2, adj_1, adj_2, features_1, features_2, device, str,  shuffle=False):

    adj_lst_1 = list()
    features_lst_1 = list()
    adj_lst_2 = list()
    features_lst_2 = list()
    graph_indicator_lst = list()
    y_lst = list()
    number = 10000

    for i in range(0, number):
        j = np.random.randint(0, number, size=9)
        for num in j:
            if num == i:
                num = num + 1
            if lab_1[i] == 1 and lab_2[num] == 1:
                n_nodes_1 = adj_1[i].shape[0]
                n_nodes_2 = adj_2[num].shape[0]
                adj_batch_1 = lil_matrix((n_nodes_1, n_nodes_1))
                adj_batch_2 = lil_matrix((n_nodes_2, n_nodes_2))
                features_batch_1 = np.zeros((n_nodes_1, 1))
                features_batch_2 = np.zeros((n_nodes_2, 1))
                adj_batch_1[0:n_nodes_1, 0:n_nodes_1] = adj_1[i]
                adj_batch_2[0:n_nodes_2, 0:n_nodes_2] = adj_2[num]
                features_batch_1[0:n_nodes_1, :] = features_1[i]
                features_batch_2[0:n_nodes_2:, :] = features_2[num]
                y_batch = np.array([0], dtype=np.float)
                graph_indicator_batch =  [n_nodes_1, n_nodes_2]
                adj_lst_1.append(sparse_mx_to_torch_sparse_tensor(adj_batch_1).to(device))
                features_lst_1.append(torch.FloatTensor(features_batch_1).to(device))
                adj_lst_2.append(sparse_mx_to_torch_sparse_tensor(adj_batch_2).to(device))
                features_lst_2.append(torch.FloatTensor(features_batch_2).to(device))
                graph_indicator_lst.append(torch.LongTensor(graph_indicator_batch).to(device))
                y_lst.append(torch.LongTensor(y_batch).to(device))
        if lab_1[i] == 1 and lab_2[i] == 1:
            n_nodes_1 = adj_1[i].shape[0]
            n_nodes_2 = adj_2[i].shape[0]
            adj_batch_1 = lil_matrix((n_nodes_1, n_nodes_1))
            adj_batch_2 = lil_matrix((n_nodes_2, n_nodes_2))
            features_batch_1 = np.zeros((n_nodes_1, 1))
            features_batch_2 = np.zeros((n_nodes_2, 1))
            adj_batch_1[0:n_nodes_1, 0:n_nodes_1] = adj_1[i]
            adj_batch_2[0:n_nodes_2, 0:n_nodes_2] = adj_2[i]
            features_batch_1[0:n_nodes_1, :] = features_1[i]
            features_batch_2[0:n_nodes_2, :] = features_2[i]
            y_batch = np.ones(1)
            graph_indicator_batch = [n_nodes_1, n_nodes_2]
            adj_lst_1.append(sparse_mx_to_torch_sparse_tensor(adj_batch_1).to(device))
            adj_lst_2.append(sparse_mx_to_torch_sparse_tensor(adj_batch_2).to(device))
            features_lst_1.append(torch.FloatTensor(features_batch_1).to(device))
            features_lst_2.append(torch.FloatTensor(features_batch_2).to(device))
            graph_indicator_lst.append(torch.LongTensor(graph_indicator_batch).to(device))
            y_lst.append(torch.LongTensor(y_batch).to(device))
    return adj_lst_1, adj_lst_2, features_lst_1, features_lst_2, graph_indicator_lst, y_lst


def accuracy(output, labels):
    #cos相似度
    # if labels > 0 and output > 0:
    #    correct = 1
    # elif labels < 0 and output < 0:
    #    correct = 1
    # else:
    #    correct = 0

    #欧式距离度量
    # if labels == 1 and output >= 0.002:
    #     correct = 1
    # elif labels == 0 and output < 0.002:
    #     correct = 1
    # else:
    #     correct = 0
    # return correct
    #神经网络拟合
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def rank(mrr):
    length = 10
    top = 0
    num = mrr[9][:, 1]
    # print(num)
    for i in range(9):
        if num > mrr[i][:, 1] or mrr[i].max(1)[1] == -1: #-1 or 0
            length = length - 1
    if length == 1:
        top = top + 1
    return 1/length, top

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