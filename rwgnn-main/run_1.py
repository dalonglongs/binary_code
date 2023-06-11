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


lab_train_1, adj_lst_train_1, features_lst_train_1 = load_data('gcc-x64-O3', './datasets/code/train.pkl')
lab_train_2, adj_lst_train_2, features_lst_train_2 = load_data('gcc-arm-O3', './datasets/code/train.pkl')
# lab_val_1, adj_lst_val_1, features_lst_val_1 = load_data('gcc-x64-O3', './datasets/code/valid.pkl', param=True)
# lab_val_2, adj_lst_val_2, features_lst_val_2 = load_data('gcc-arm-O3', './datasets/code/valid.pkl', param=True)
# lab_test_1, adj_lst_test_1, features_lst_test_1 = load_data('gcc-x64-O1', './datasets/code/test.pkl', param=True)
# lab_test_2, adj_lst_test_2, features_lst_test_2 = load_data('gcc-arm-O3', './datasets/code/test.pkl', param=True)
# lab_train_1, adj_lst_train_1, features_lst_train_1 = load_data('gcc-x64-O3', '腾讯的训练集.json')
# lab_train_2, adj_lst_train_2, features_lst_train_2 = load_data('gcc-arm-O3', '腾讯的训练集.json')
lab_val_1, adj_lst_val_1, features_lst_val_1 = json_load_data('gcc-x64-O3', '腾讯的验证集.json')
lab_val_2, adj_lst_val_2, features_lst_val_2 = json_load_data('gcc-arm-O3', '腾讯的验证集.json')
# lab_test_1, adj_lst_test_1, features_lst_test_1 = load_data('gcc-x64-O2', '腾讯的测试集.json', param=True)
# lab_test_2, adj_lst_test_2, features_lst_test_2 = load_data('gcc-arm-O2', '腾讯的测试集.json', param=True)
adj_train_1, adj_train_2, features_train_1, features_train_2, graph_indicator_train, y_train = generate_batches(lab_train_1, lab_train_2, adj_lst_train_1, adj_lst_train_2, features_lst_train_1, features_lst_train_2, device,'train')
adj_val_1, adj_val_2, features_val_1, features_val_2, graph_indicator_val, y_val = generate_batches(lab_val_1, lab_val_2, adj_lst_val_1, adj_lst_val_2, features_lst_val_1, features_lst_val_2, device, 'val')
# adj_test_1, adj_test_2, features_test_1, features_test_2, graph_indicator_test, y_test = test_generate_batches(lab_test_1, lab_test_2, adj_lst_test_1,adj_lst_test_2, features_lst_test_1, features_lst_test_2, device, 'test')
# adj_test_1, adj_test_2, features_test_1, features_test_2, graph_indicator_test, y_test = generate_batches(lab_test_1, lab_test_2, adj_lst_test_1,adj_lst_test_2, features_lst_test_1, features_lst_test_2, device, 'test')

print(adj_val_1[0])
print(adj_val_2[0])

# N = len(adj_lst)
#
# features_dim = features_lst[0].shape[1]
#
# enc = LabelEncoder()
# class_labels = enc.fit_transform(class_labels)

n_classes = 2
# y = [np.array(class_labels[i]) for i in range(class_labels.size)]

kf = KFold(n_splits=4, shuffle=True, random_state=13)
it = 0
accs = list()

model = RW_NN(1, args.max_step, args.hidden_graphs, args.size_hidden_graphs, args.hidden_dim,
                  args.penultimate_dim, args.normalize, n_classes, args.dropout, device).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

def train(adj_train_1, adj_train_2, features_train_1, features_train_2,  graph_indicator_train, y_train):
    optimizer.zero_grad()
    # 神经网络拟合
    # output = model(adj_train_1, adj_train_2, features_train_1, features_train_2,  graph_indicator_train)
    # loss_train = F.cross_entropy(output, y_train)

    #cos度量
    output1, output2 = model(adj_train_1, adj_train_2, features_train_1, features_train_2, graph_indicator_train)
    loss_train = torch.nn.CosineEmbeddingLoss()(output1, output2, y_train)
    output = torch.cosine_similarity(output1, output2, dim=1)

    #欧氏距离度量
    # output1, output2 = model(adj_train_1, adj_train_2, features_train_1, features_train_2, graph_indicator_train)
    # loss_train = MyLoss()(output1, output2, y_train)
    # output = F.pairwise_distance(output1, output2)


    loss_train.backward()
    optimizer.step()
    return output, loss_train


def test(adj_test_1, adj_test_2, features_test_1, features_test_2, graph_indicator_test, y_test):
    #神经网络拟合
    output = model(adj_test_1, adj_test_2, features_test_1, features_test_2, graph_indicator_test)
    loss_test = F.cross_entropy(output, y_test)

    # cos度量
    # output1, output2 = model(adj_test_1, adj_test_2, features_test_1, features_test_2, graph_indicator_test)
    # loss_test = torch.nn.CosineEmbeddingLoss()(output1, output2, y_test)
    # output = torch.cosine_similarity(output1, output2, dim=1)

    # 欧氏距离度量
    # output1, output2 = model(adj_test_1, adj_test_2, features_test_1, features_test_2, graph_indicator_test)
    # loss_test = MyLoss()(output1, output2, y_test)
    # output = F.pairwise_distance(output1, output2)

    return output, loss_test

def disrupt(length):
    a = []
    for i in range(length):
        a.append(i)
    random.shuffle(a)
    return a


best_acc = 0
Loss_list = []
Accuracy_list = []

print(len(adj_train_1))
print(len(adj_val_1))
# print(len(adj_test_1))
for epoch in range(args.epochs):
    start = time.time()
    model.train()
    train_loss = AverageMeter()
    train_acc = AverageMeter()

        # Train for one epoch
    a = disrupt(len(adj_train_1))
    for i in a:
        output, loss = train(adj_train_1[i], adj_train_2[i], features_train_1[i], features_train_2[i], graph_indicator_train[i], y_train[i])
        train_loss.update(loss.item(), output.size(0))
        train_acc.update(accuracy(output.data, y_train[i].data), output.size(0))

        # Evaluate on validation set
    model.eval()
    val_loss = AverageMeter()
    val_acc = AverageMeter()
    a = disrupt(len(adj_val_1))
    for i in a:
        output, loss = test(adj_val_1[i], adj_val_2[i], features_val_1[i], features_val_2[i], graph_indicator_val[i], y_val[i])
        val_loss.update(loss.item(), output.size(0))
        val_acc.update(accuracy(output.data, y_val[i].data), output.size(0))

    scheduler.step()

    Loss_list.append(val_loss.avg)
    Accuracy_list.append(val_acc.avg)

        # Print results
    print("epoch:", '%03d' % (epoch + 1), "train_loss=",
              "{:.5f}".format(train_loss.avg),
              "train_acc=", "{:.5f}".format(train_acc.avg), "val_loss=", "{:.5f}".format(val_loss.avg),
              "val_acc=", "{:.5f}".format(val_acc.avg), "time=", "{:.5f}".format(time.time() - start))

        # Remember best accuracy and save checkpoint
    is_best = val_acc.avg >= best_acc
    best_acc = max(val_acc.avg, best_acc)
    if is_best:
        early_stopping_counter = 0
        torch.save({
            'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, 'model_best_O3.pth.tar')

x1 = range(0, args.epochs)
x2 = range(0, args.epochs)
y1 = Accuracy_list
y2 = Loss_list
plt.subplot(2, 1, 1)
plt.plot(x1, y1)
plt.title('val accuracy vs. epoches' + str(it))
plt.ylabel('val accuracy')
plt.subplot(2, 1, 2)
plt.plot(x2, y2)
plt.xlabel('Test loss vs. epoches')
plt.ylabel('Test loss')
plt.show()

print("Optimization finished!")

#     # Testing
# test_loss = AverageMeter()
# test_acc = AverageMeter()
# print("Loading checkpoint!")
# checkpoint = torch.load('model_best_loca.pth.tar')
# epoch = checkpoint['epoch']
# model.load_state_dict(checkpoint['state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer'])
#
# # mrr和rank1评价指标
# MRR10 = []
# RANK1 = []
# for i in range(0, len(adj_test_1), 10):
#     output, loss = test(adj_test_1[i], adj_test_2[i], features_test_1[i], features_test_2[i], graph_indicator_test[i], y_test[i])
#     test_loss.update(loss.item(), output.size(0))
#     test_acc.update(accuracy(output.data, y_test[i].data), output.size(0))
#     MRR = []
#     for gen in range(i, i+10):
#         output, loss = test(adj_test_1[gen], adj_test_2[gen], features_test_1[gen], features_test_2[gen], graph_indicator_test[gen], y_test[gen])
#         test_loss.update(loss.item(), output.size(0))
#         test_acc.update(accuracy(output.data, y_test[i].data), output.size(0))
#         MRR.append(output.data)
#     mrr10, rank1 = rank(MRR)
#     MRR10.append(mrr10)
#     RANK1.append(rank1)
#
#
#
#
# accs.append(test_acc.avg.cpu().numpy())
#
#     # Print results
# print("test_loss=", "{:.5f}".format(test_loss.avg), "test_acc=", "{:.5f}".format(test_acc.avg))
#
#
#
#
# print("avg_test_acc=", "{:.5f}".format(np.mean(accs)))
# print("MRR10:", sum(MRR10)/(len(adj_test_1)/10))
# print("RANK1:", sum(RANK1)/(len(adj_test_1)/10))
# x3 = range(0, 10)
# y3 = accs
# plt.plot(x1, y1)
# plt.title('Test accuracy')
# plt.xlabel('epoches')
# plt.ylabel('Test acc')
# plt.show()
