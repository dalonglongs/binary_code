import os
from read_imagic import generate_batches, get_cfg, load_data
import pickle
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import time
from code_mode import RW_NN
import random
from sklearn.metrics import roc_auc_score, roc_curve
import math
from utils_1 import accuracy, AverageMeter

parser = argparse.ArgumentParser(description='RW_NN')
parser.add_argument('--dataset', default='IMDB-BINARY', help='Dataset name')
parser.add_argument('--use-node-labels', action='store_true', default=False, help='Whether to use node labels')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR', help='Initial learning rate')
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
device = torch.device("cpu")
model = RW_NN(1, args.max_step, args.hidden_graphs, args.size_hidden_graphs, args.hidden_dim,
                  args.penultimate_dim, args.normalize, 2, args.dropout, device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

def disrupt(length):
    a = []
    for i in range(length):
        a.append(i)
    random.shuffle(a)
    return a

def train(adj_train_1, adj_train_2, features_train_1, features_train_2,  graph_indicator_train, y_train):
    optimizer.zero_grad()
    # 神经网络拟合
    output = model(adj_train_1, adj_train_2, features_train_1, features_train_2,  graph_indicator_train)
    loss_train = F.cross_entropy(output, y_train)
    loss_train.backward()
    optimizer.step()
    return output, loss_train
def test(adj_test_1, adj_test_2, features_test_1, features_test_2, graph_indicator_test, y_test):
    #神经网络拟合
    output = model(adj_test_1, adj_test_2, features_test_1, features_test_2, graph_indicator_test)
    loss_test = F.cross_entropy(output, y_test)
    return output, loss_test

if __name__ == '__main__':

    features = []
    arm_dict = {}
    x86_dict = {}
    # group = os.walk('datas')
    # for path, dir_list, file_list in group:
    #     for i in file_list:
    #         file_path = os.path.join(path, i)
    #         print(file_path)
    #         a = get_cfg(file_path)
    #         if 'arm' in file_path:
    #             arm_dict.update(a)
    #         else:
    #             x86_dict.update(a)
    # with open('datas_com/arm.pkl', 'wb') as f:
    #     pickle.dump(arm_dict, f)
    # with open('datas_com/x64.pkl', 'wb') as x86:
    #     pickle.dump(x86_dict, x86)
    #
    # arm_dict = {}
    # x86_dict = {}
    # group = os.walk('eva')
    # for path, dir_list, file_list in group:
    #     for i in file_list:
    #         file_path = os.path.join(path, i)
    #         print(file_path)
    #         a = get_cfg(file_path)
    #         if 'arm' in file_path:
    #             arm_dict.update(a)
    #         else:
    #             x86_dict.update(a)
    #
    #
    # with open('eva_com/arm.pkl', 'wb') as f:
    #     pickle.dump(arm_dict, f)
    # with open('eva_com/x64.pkl', 'wb') as x86:
    #     pickle.dump(x86_dict, x86)
    x = 0
    group = os.walk('datas')
    for path, dir_list, file_list in group:
        for i in file_list:
            file_path = os.path.join(path, i)
            x = x + 1
    print(x)
    x=0
    group = os.walk('datas_com')
    for path, dir_list, file_list in group:
        for i in file_list:
            file_path = os.path.join(path, i)
            print(file_path)
            mess = load_data(file_path)
            x = x + len(mess)
            features.append(mess)
    print(x)
    group = os.walk('eva_com')
    for path, dir_list, file_list in group:
        for i in file_list:
            file_path = os.path.join(path, i)
            print(file_path)
            mess = load_data(file_path)
            features.append(mess)
    adj_train_1, adj_train_2, features_train_1, features_train_2, graph_indicator_train, y_train = generate_batches(features[0], features[1],'train')
    adj_val_1, adj_val_2, features_val_1, features_val_2, graph_indicator_val, y_val = generate_batches(features[2], features[3], 'val')

    best_auc = 0
    best_acc = 0
    Loss_list = []
    Accuracy_list = []
    for epoch in range(args.epochs):
        train_true = []
        train_scores = []
        val_true = []
        val_scores = []

        start = time.time()
        model.train()
        train_loss = AverageMeter()
        train_acc = AverageMeter()


        # Train for one epoch
        a = disrupt(len(adj_train_1))
        for i in a:
            output, loss = train(adj_train_1[i], adj_train_2[i], features_train_1[i], features_train_2[i],
                                 graph_indicator_train[i], y_train[i])
            # preds = output.max(1)[1].type_as(y_train[i])
            # train_true.append(float(y_train[i]))
            # train_scores.append(math.exp(float(output.data[0][1])))
            train_loss.update(loss.item(), output.size(0))
            train_acc.update(accuracy(output.data, y_train[i].data), output.size(0))


            # Evaluate on validation set
        model.eval()
        val_loss = AverageMeter()
        val_acc = AverageMeter()
        a = disrupt(len(adj_val_1))
        for i in a:
            output, loss = test(adj_val_1[i], adj_val_2[i], features_val_1[i], features_val_2[i],
                                graph_indicator_val[i], y_val[i])
            # preds = output.max(1)[1].type_as(y_val[i])
            val_true.append(float(y_val[i]))
            val_scores.append(math.exp(float(output.data[0][1])))
            val_loss.update(loss.item(), output.size(0))
            val_acc.update(accuracy(output.data, y_val[i].data), output.size(0))
        Loss_list.append(val_loss.avg)
        Accuracy_list.append(val_acc.avg)
        val_auc = roc_auc_score(val_true, val_scores)
        # Print results
        print("epoch:", '%03d' % (epoch + 1), "train_loss=",
              "{:.5f}".format(train_loss.avg),
              "train_acc=", "{:.5f}".format(train_acc.avg), "val_loss=", "{:.5f}".format(val_loss.avg),
              "val_acc=", "{:.5f}".format(val_acc.avg), "time=", "{:.5f}".format(time.time() - start), "val_auc=", "{:.5f}".format(val_auc))

        # Remember best accuracy and save checkpoint
        # is_best = val_acc.avg >= best_acc
        # best_acc = max(val_acc.avg, best_acc)
        # if is_best:
        #     early_stopping_counter = 0
        #     torch.save({
        #         'epoch': epoch + 1,
        #         'state_dict': model.state_dict(),
        #         'optimizer': optimizer.state_dict(),
        #     }, 'model_best_O2_7.pth.tar')

        # scheduler.step()
        # train_auc = a = roc_auc_score(train_true, train_scores)

        # print('epoch' + str(epoch) + ':' + 'train-auc=' + str
        # (train_auc) + "; " + 'eval_auc=' + str(val_auc) + '; ' + "time=", "{:.5f}".format(time.time() - start))
        epoch_auc = val_auc

        if epoch_auc > best_auc:
            early_stopping_counter = 0
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, 'model_best_O2.pth.tar')
            best_auc = epoch_auc
