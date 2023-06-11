import angr
import pickle
import os
from scipy.sparse import csr_matrix, lil_matrix
import networkx as nx
from utils_1 import sparse_mx_to_torch_sparse_tensor
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim
import argparse
from code_mode import RW_NN
import torch.nn.functional as F
import signal
import matplotlib.pyplot as plt
import math
from auc import *

# Argument parser
parser = argparse.ArgumentParser(description='RW_NN')
parser.add_argument('--dataset', default='IMDB-BINARY', help='Dataset name')
parser.add_argument('--use-node-labels', action='store_true', default=False, help='Whether to use node labels')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR', help='Initial learning rate')
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')
parser.add_argument('--batch-size', type=int, default=10, metavar='N', help='Input batch size for training')
parser.add_argument('--epochs', type=int, default=2, metavar='N', help='Number of epochs to train')
parser.add_argument('--hidden-graphs', type=int, default=16, metavar='N', help='Number of hidden graphs')
parser.add_argument('--size-hidden-graphs', type=int, default=5, metavar='N',
                    help='Number of nodes of each hidden graph')
parser.add_argument('--hidden-dim', type=int, default=4, metavar='N', help='Size of hidden layer of NN')
parser.add_argument('--penultimate-dim', type=int, default=32, metavar='N', help='Size of penultimate layer of NN')
parser.add_argument('--max-step', type=int, default=2, metavar='N', help='Max length of walks')
parser.add_argument('--normalize', action='store_true', default=False, help='Whether to normalize the kernel values')

args = parser.parse_args()
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cpu")
n_classes = 2
model = RW_NN(1, args.max_step, args.hidden_graphs, args.size_hidden_graphs, args.hidden_dim,
                  args.penultimate_dim, args.normalize, n_classes, args.dropout, device).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

def generate_batches(features_1, features_2, str,  shuffle=False):

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


    keys = list(features_2.keys())

    length = len(keys)
    # for i in range(0, number_1):
    for i in features_1:
        datas = [x for x in keys if x != i]
        j = np.random.randint(0, len(datas), size=2)
        for num in j:
            n_nodes_1 = features_1[i][0].shape[0]
            n_nodes_2 = features_2[datas[num]][0].shape[0]
            adj_batch_1 = lil_matrix((n_nodes_1, n_nodes_1))
            adj_batch_2 = lil_matrix((n_nodes_2, n_nodes_2))
            features_batch_1 = np.zeros((n_nodes_1, 1))
            features_batch_2 = np.zeros((n_nodes_2, 1))
            adj_batch_1[0:n_nodes_1, 0:n_nodes_1] = features_1[i][0]
            adj_batch_2[0:n_nodes_2, 0:n_nodes_2] = features_2[datas[num]][0]
            features_batch_1[0:n_nodes_1, :] = features_1[i][1]
            features_batch_2[0:n_nodes_2:, :] = features_2[datas[num]][1]
            y_batch = np.array([0], dtype=np.float)
            graph_indicator_batch = [n_nodes_1, n_nodes_2]
            adj_lst_1.append(sparse_mx_to_torch_sparse_tensor(adj_batch_1))
            features_lst_1.append(torch.FloatTensor(features_batch_1))
            adj_lst_2.append(sparse_mx_to_torch_sparse_tensor(adj_batch_2))
            features_lst_2.append(torch.FloatTensor(features_batch_2))
            graph_indicator_lst.append(torch.LongTensor(graph_indicator_batch))
            y_lst.append(torch.LongTensor(y_batch))
        if i in keys:
            n_nodes_1 = features_1[i][0].shape[0]
            n_nodes_2 = features_2[i][0].shape[0]
            adj_batch_1 = lil_matrix((n_nodes_1, n_nodes_1))
            adj_batch_2 = lil_matrix((n_nodes_2, n_nodes_2))
            features_batch_1 = np.zeros((n_nodes_1, 1))
            features_batch_2 = np.zeros((n_nodes_2, 1))
            adj_batch_1[0:n_nodes_1, 0:n_nodes_1] =features_1[i][0]
            adj_batch_2[0:n_nodes_2, 0:n_nodes_2] = features_2[i][0]
            features_batch_1[0:n_nodes_1, :] = features_1[i][1]
            features_batch_2[0:n_nodes_2, :] = features_2[i][1]
            y_batch = np.ones(1)
            graph_indicator_batch = [n_nodes_1, n_nodes_2]
            adj_lst_1.append(sparse_mx_to_torch_sparse_tensor(adj_batch_1))
            adj_lst_2.append(sparse_mx_to_torch_sparse_tensor(adj_batch_2))
            features_lst_1.append(torch.FloatTensor(features_batch_1))
            features_lst_2.append(torch.FloatTensor(features_batch_2))
            graph_indicator_lst.append(torch.LongTensor(graph_indicator_batch))
            y_lst.append(torch.LongTensor(y_batch))
    return adj_lst_1, adj_lst_2, features_lst_1, features_lst_2, graph_indicator_lst, y_lst

def gener_node_dict(name):
    dict = {}
    x = 1
    for i in name:
        dict[i] = x
        x = x + 1

    return dict
def load_data(file_name, param = False):
    adj = []
    features = []
    label = []
    dict = {}
    with open(file_name, 'rb') as f:
        file = pickle.load(f)


    for num, i in enumerate(file):
        edg_1 = []
        edg_2 = []


        if file[i] != []:
            g = file[i][0]
            node_dict = gener_node_dict(g.nodes)

            for (u,v) in g.edges:
                edg_1.append(node_dict[u]-1)
                edg_1.append(node_dict[v]-1)
                edg_2.append(node_dict[v]-1)
                edg_2.append(node_dict[u]-1)
            A = csr_matrix((np.ones(2*len(g.edges())), (edg_1, edg_2)),
                           shape=(g.number_of_nodes(), g.number_of_nodes()))
            x = A.sum(axis=1)
            adj.append(A)
            label.append(1)
            if A.shape[0] != len(x):

                features.append(x)
        else:
            adj.append(-1)
            features.append(-1)
            label.append(0)
        dict[i] = []
        dict[i].append(A)
        dict[i].append(x)
        dict[i].append(file[i][1])


    return dict

def set_timeout(num, callback):
  def wrap(func):
    def handle(signum, frame): # 收到信号 SIGALRM 后的回调函数，第一个参数是信号的数字，第二个参数是the interrupted stack frame.
      raise RuntimeError
    def to_do(*args, **kwargs):
      try:
        signal.signal(signal.SIGALRM, handle) # 设置信号和回调函数
        signal.alarm(num) # 设置 num 秒的闹钟
        # print('start alarm signal.')
        r = func(*args, **kwargs)
        # print('close alarm signal.')
        signal.alarm(0) # 关闭闹钟
        return r
      except RuntimeError as e:
        callback()
    return to_do
  return wrap
def after_timeout(): # 超时后的处理函数
  print("Time out!")
@set_timeout(600, after_timeout) # 限时 2 秒超时
def get_cfg(file_name):
    dict = {}
    try:
        project = angr.Project(file_name, load_options={'auto_load_libs': False})

        cfg = project.analyses.CFGFast()
        funcs_addr_set = cfg.kb.functions.function_addrs_set
        total = 0
        a = 0
        b = 0
        c = 0

        for func_addr in iter(funcs_addr_set):
            total = total + 1
            func = cfg.kb.functions[func_addr]
            name = func.name  # str, function name
            dict[name] = []
            func_graph = func.graph
            dict[name].append(func_graph)
            dict[name].append(func_graph.number_of_nodes())
            if func_graph.number_of_nodes()<=5:
                a = a + 1
            elif 5 < func_graph.number_of_nodes() <= 20:
                b = b + 1

            else:
                c = c + 1
    except:
        dict = {}

    # with open('data_com/arm_graph.pkl', 'wb') as f:
    #     pickle.dump(dict, f)
    return dict

def test(adj_test_1, adj_test_2, features_test_1, features_test_2, graph_indicator_test, y_test):
    #神经网络拟合
    output = model(adj_test_1, adj_test_2, features_test_1, features_test_2, graph_indicator_test)
    loss_test = F.cross_entropy(output, y_test)
    return output, loss_test


if __name__ == '__main__':

    features = []

    group = os.walk('data')
    for path, dir_list, file_list in group:
        for i in file_list:
            file_path = os.path.join(path, i)
            get_cfg(file_path)

    group = os.walk('data_com')
    for path, dir_list, file_list in group:
        for i in file_list:
            file_path = os.path.join(path, i)
            if 'train' not in file_path:
                print(file_path)
                mess = load_data(file_path)
                features.append(mess)
    adj_test_1, adj_test_2, features_test_1, features_test_2, graph_indicator_test, y_test = generate_batches(features[0], features[1], 'test')
    y_true = []
    y_scores = []
    fpr = []
    tpr = []
    print("Loading checkpoint!")
    checkpoint = torch.load('model_best_loca.pth.tar')
    epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    for i in range(len(y_test)):
        output, loss = test(adj_test_1[i], adj_test_2[i], features_test_1[i], features_test_2[i],
                            graph_indicator_test[i], y_test[i])
        preds = output.max(1)[1].type_as(y_test[i])
        y_true.append(float(y_test[i]))
        y_scores.append(math.exp(float(output.data[0][1])))
    fpr.append(y_true)
    tpr.append(y_scores)

    roc(fpr, tpr)