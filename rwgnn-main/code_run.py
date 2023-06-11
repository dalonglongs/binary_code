import argparse
import time
import numpy as np
from math import ceil
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn.functional as F
from torch import optim

from code_mode import RW_NN
from code_utils import load_data, generate_batches, accuracy, AverageMeter
from torchsummary import summary

import matplotlib.pyplot as plt

# Argument parser
parser = argparse.ArgumentParser(description='RW_NN')
parser.add_argument('--dataset', default='IMDB-BINARY', help='Dataset name')
parser.add_argument('--use-node-labels', action='store_true', default=False, help='Whether to use node labels')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', help='Initial learning rate')
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')
parser.add_argument('--batch-size', type=int, default=10, metavar='N', help='Input batch size for training')
parser.add_argument('--epochs', type=int, default=500, metavar='N', help='Number of epochs to train')
parser.add_argument('--hidden-graphs', type=int, default=16, metavar='N', help='Number of hidden graphs')
parser.add_argument('--size-hidden-graphs', type=int, default=5, metavar='N',
                    help='Number of nodes of each hidden graph')
parser.add_argument('--hidden-dim', type=int, default=4, metavar='N', help='Size of hidden layer of NN')
parser.add_argument('--penultimate-dim', type=int, default=32, metavar='N', help='Size of penultimate layer of NN')
parser.add_argument('--max-step', type=int, default=2, metavar='N', help='Max length of walks')
parser.add_argument('--normalize', action='store_true', default=False, help='Whether to normalize the kernel values')

args = parser.parse_args()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

adj_lst_train_1, features_lst_train_1 = load_data('gcc-x64-O0', './datasets/code/train.pkl')
adj_lst_train_2, features_lst_train_2 = load_data('gcc-x64-O1', './datasets/code/train.pkl')
adj_lst_val_1, features_lst_val_1 = load_data('gcc-x64-O0', './datasets/code/valid.pkl')
adj_lst_val_2, features_lst_val_2 = load_data('gcc-x64-O1', './datasets/code/valid.pkl')
adj_lst_test_1, features_lst_test_1 = load_data('gcc-x64-O0', './datasets/code/test.pkl')
adj_lst_test_2, features_lst_test_2 = load_data('gcc-arm-O0', './datasets/code/valid.pkl')
adj_train, features_train, graph_indicator_train, y_train = generate_batches(adj_lst_train_1,adj_lst_train_2, features_lst_train_1, features_lst_train_2, device,'train')
adj_val, features_val, graph_indicator_val, y_val = generate_batches(adj_lst_val_1,adj_lst_val_2, features_lst_val_1, features_lst_val_2, device, 'val')
adj_test, features_test, graph_indicator_test, y_test = generate_batches(adj_lst_test_1,adj_lst_test_2, features_lst_test_1, features_lst_test_2, device, 'test')



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

def train(adj_train, features_train, graph_indicator_train, y_train):
    optimizer.zero_grad()
    output = model(adj_train, features_train, graph_indicator_train)
    loss_train = F.cross_entropy(output, y_train)
    loss_train.backward()
    optimizer.step()
    return output, loss_train


def test(adj_test, features_test, graph_indicator_test, y_test):
    output = model(adj_test, features_test, graph_indicator_test)
    loss_test = F.cross_entropy(output, y_test)
    return output, loss_test


best_acc = 0
Loss_list = []
Accuracy_list = []

print(len(adj_train))
print(len(adj_val))
print(len(adj_test))

for epoch in range(args.epochs):
    start = time.time()
    model.train()
    train_loss = AverageMeter()
    train_acc = AverageMeter()

        # Train for one epoch
    for i in range(len(adj_train)):
        output, loss = train(adj_train[i], features_train[i], graph_indicator_train[i], y_train[i])
        train_loss.update(loss.item(), output.size(0))
        train_acc.update(accuracy(output.data, y_train[i].data), output.size(0))

        # Evaluate on validation set
    model.eval()
    val_loss = AverageMeter()
    val_acc = AverageMeter()

    for i in range(len(adj_val)):
        output, loss = test(adj_val[i], features_val[i], graph_indicator_val[i], y_val[i])
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
            }, 'model_best_loca.pth.tar')

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

    # Testing
test_loss = AverageMeter()
test_acc = AverageMeter()
print("Loading checkpoint!")
checkpoint = torch.load('model_best_loca.pth.tar')
epoch = checkpoint['epoch']
model.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])

for i in range(len(adj_test)):
    output, loss = test(adj_test[i], features_test[i], graph_indicator_test[i], y_test[i])
    test_loss.update(loss.item(), output.size(0))
    test_acc.update(accuracy(output.data, y_test[i].data), output.size(0))
accs.append(test_acc.avg.cpu().numpy())

    # Print results
print("test_loss=", "{:.5f}".format(test_loss.avg), "test_acc=", "{:.5f}".format(test_acc.avg))
print()



print("avg_test_acc=", "{:.5f}".format(np.mean(accs)))
x3 = range(0, 10)
y3 = accs
plt.plot(x1, y1)
plt.title('Test accuracy')
plt.xlabel('epoches')
plt.ylabel('Test acc')
plt.show()
