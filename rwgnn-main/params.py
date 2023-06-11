import argparse
import time
import numpy as np
from math import ceil
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from code_mode import RW_NN
from utils_1 import load_data, generate_batches, accuracy, AverageMeter, rank, test_generate_batches, json_load_data
from torchsummary import summary

import matplotlib.pyplot as plt
import random
from thop import profile
from thop import clever_format
import torchsummary


# Argument parser
parser = argparse.ArgumentParser(description='RW_NN')
parser.add_argument('--dataset', default='IMDB-BINARY', help='Dataset name')
parser.add_argument('--use-node-labels', action='store_true', default=False, help='Whether to use node labels')
parser.add_argument('--lr', type=float, default=1e-5, metavar='LR', help='Initial learning rate')
parser.add_argument('--dropout', type=float, default=0, help='Dropout rate (1 - keep probability).')
parser.add_argument('--batch-size', type=int, default=10, metavar='N', help='Input batch size for training')
parser.add_argument('--epochs', type=int, default=300, metavar='N', help='Number of epochs to train')
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

#自定义欧氏距离损失函数
class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        self.margin = 0.002

    def forward(self, output1, output2, y_test):
        dist = F.pairwise_distance(output1, output2)
        loss = y_test * torch.square(torch.max(torch.from_numpy(np.array([0., self.margin - dist.data])))) + (1 - y_test) * torch.square(dist)
        return  loss


n_classes = 2
model = RW_NN(1, args.max_step, args.hidden_graphs, args.size_hidden_graphs, args.hidden_dim,
                  args.penultimate_dim, args.normalize, n_classes, args.dropout, device).to(device)
lab_train_1, adj_lst_train_1, features_lst_train_1 = load_data('gcc-x64-O3', './datasets/code/train.pkl')
lab_train_2, adj_lst_train_2, features_lst_train_2 = load_data('gcc-arm-O3', './datasets/code/train.pkl')
adj_train_1, adj_train_2, features_train_1, features_train_2, graph_indicator_train, y_train = generate_batches(lab_train_1, lab_train_2, adj_lst_train_1, adj_lst_train_2, features_lst_train_1, features_lst_train_2, device,'train')
flops, params = profile(model.to(device), inputs=(adj_train_1[1], adj_train_2[1], features_train_1[1], features_train_2[1], graph_indicator_train[1]))
flops, params = clever_format([flops, params], "%.3f")
print(flops, params)