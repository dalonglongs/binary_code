import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from utils_1 import load_data, generate_batches, json_load_data
import torch
from torch import optim
import argparse
from code_mode import RW_NN
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import matplotlib
from sklearn.metrics import roc_auc_score, roc_curve

def auc():

    y_true = np.array([1, 1, 2, 2])
    y_scores = np.array([0.6, 0.6, 0.35, 0.8])
    a = roc_auc_score(y_true, y_scores)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=2)
    print(a)
    plt.plot(fpr, tpr, label='ROC')

    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.show()

# Argument parser
parser = argparse.ArgumentParser(description='RW_NN')
parser.add_argument('--dataset', default='IMDB-BINARY', help='Dataset name')
parser.add_argument('--use-node-labels', action='store_true', default=False, help='Whether to use node labels')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', help='Initial learning rate')
parser.add_argument('--dropout', type=float, default=0.4, help='Dropout rate (1 - keep probability).')
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

def test(adj_test_1, adj_test_2, features_test_1, features_test_2, graph_indicator_test, y_test):
    #神经网络拟合
    output = model(adj_test_1, adj_test_2, features_test_1, features_test_2, graph_indicator_test)
    loss_test = F.cross_entropy(output, y_test)
    return output, loss_test

def roc(x,y):

    for i in range(len(x)):
        y_true = np.array(x[i])
        y_scores = np.array(y[i])
        a = roc_auc_score(y_true, y_scores)
        fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)

        if i == 0:
            plt.plot(fpr, tpr, label='node number:1-5, auc='+str(a))


        if i == 1:
            plt.plot(fpr, tpr, label='node number:6-20, auc='+str(a))
        if i == 2:
            plt.plot(fpr, tpr, label='node number:20~, auc='+str(a))

    plt.legend(loc=0)  # 说明所在位置
    plt.show()

if __name__ == '__main__':
    # auc和roc评价指标
    fpr = []
    tpr = []
    for i in range(1, 4):
        lab_test_1, adj_lst_test_1, features_lst_test_1 = json_load_data('gcc-x64-O3', '腾讯的测试集.json', param=i)

        lab_test_2, adj_lst_test_2, features_lst_test_2 = json_load_data('gcc-arm-O3', '腾讯的测试集.json', param=i)
        adj_test_1, adj_test_2, features_test_1, features_test_2, graph_indicator_test, y_test = generate_batches(
            lab_test_1, lab_test_2, adj_lst_test_1, adj_lst_test_2, features_lst_test_1, features_lst_test_2, device,
            'test')
        y_true = []
        y_scores = []
        print("Loading checkpoint!")
        checkpoint = torch.load('model_best_O3.pth.tar')
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        for i in range(len(y_test)):
            output, loss = test(adj_test_1[i], adj_test_2[i], features_test_1[i], features_test_2[i], graph_indicator_test[i], y_test[i])
            preds = output.max(1)[1].type_as(y_test[i])
            # print(output.data[0])
            # print(float(output.data[0][0]))
            # print('--------')
            # print(float(y_test[i]))
            # print('---------')
            # print(preds)
            y_true.append(float(y_test[i]))
            # if int(y_test[0][0]) == 1:
            #     y_scores.append(math.exp(float(output.data[0][1])))
            # else:
            #     y_scores.append(math.exp(float(output.data[0][0])))
            y_scores.append(math.exp(float(output.data[0][1])))
        fpr.append(y_true)
        tpr.append(y_scores)
    roc(fpr, tpr)