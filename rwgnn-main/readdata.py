import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import json

# print(df.columns)                   # 33 columns

# sample = df.iloc[0]
# src, bin = sample['c_label'], sample['gcc-x64-O0']
# print(bin)
# # print(src)                          # character-level source code
# g = nx.read_gpickle(bin)
def write(xx, filename):
    df = pd.read_pickle(filename)
    x = 0
    y = 0
    z = 0
    x_list = []
    y_list = []
    z_list = []
    lable = {}
    if xx == 0:
        num = 30000
    else:
        num = 10000
    for i in range(0, num):
        lable[i] = []
        path = {}
        for col in df.columns:
            if col == 'c_label':
                continue
            edg_1 = []
            edg_2 = []
            a = 0
            sample = df.iloc[i]
            bin = sample[col]
            # print(bin)
            g = nx.read_gpickle(bin)
            for (u, v) in g.edges:
                edg_1.append((u, v))
                edg_2.append((v, u))
                for s in range(0, len(edg_1)):
                    if edg_1[s] in edg_2:
                        a = 1
                        break
                if a==1:
                    break
            if a==1:
                path[col] = ['NULL', 0]
                continue
            else:
                if 0 < len(g.nodes) <= 5:
                    x = x + 1
                    path[col] = [bin, len(g.nodes)]
                elif 5 < len(g.nodes) <= 20:
                    y = y + 1
                    path[col] = [bin, len(g.nodes)]
                else:
                    z = z + 1
                    path[col] = [bin, len(g.nodes)]
        if path != {}:
            lable[i] = path

    # path = {}
    # path['0-5'] = x_list
    # path['6-20'] = y_list
    # path['21-'] = z_list
    if xx == 0:
        write_json("腾讯的训练集.json",lable)
    elif xx == 1:
        write_json("腾讯的测试集.json", lable)
    else:
        write_json("腾讯的验证集.json", lable)
    print('节点数小于5的占比为:{:.2%}'.format(x / (x + y + z)))
    print('节点数大于5小于20的占比为:{:.2%}'.format(y / (x + y + z)))
    print('节点数大于20的占比为:{:.2%}'.format(z / (x + y + z)))
    print('总数为：{}'.format(x + y + z))

def write_json(filename,lable):
    with open(filename,'w',encoding='utf-8') as f:
        json.dump(lable, f,ensure_ascii=False)



def read():
    with open("腾讯的测试集.json", 'r') as load_f:
        load_dict = json.load(load_f)
        # print(load_dict['199']['gcc-x64-O2'][0])
        print(len(load_dict))

# def read_json(ds_name, filenaem, lable = False):
def read_json(filenaem, lable=False):
    a = 0
    b = 0
    c = 0
    d = 0
    e = 0
    # with open(filenaem, 'r') as load_f:
    #     load_dict = json.load(load_f)
    #     for i in load_dict:
    #         if load_dict[i][ds_name] != [] and load_dict[i][ds_name][1] == 1:
    #             a = a+1
    #         elif load_dict[i][ds_name] != [] and load_dict[i][ds_name][1] == 2:
    #             b = b + 1
    #         elif load_dict[i][ds_name] != [] and load_dict[i][ds_name][1] == 3:
    #             c = c + 1
    #         elif load_dict[i][ds_name] != [] and load_dict[i][ds_name][1] == 4:
    #             d = d + 1
    #         elif load_dict[i][ds_name] != [] and load_dict[i][ds_name][1] == 5:
    #             e = e + 1
    #     totle = a + b + c + d + e
    #     print('节点数等于1的占比为:{:.2%}'.format(a / totle))
    #     print('节点数等于2的占比为:{:.2%}'.format(b / totle))
    #     print('节点数等于3的占比为:{:.2%}'.format(c / totle))
    #     print('节点数等于4的占比为:{:.2%}'.format(d / totle))
    #     print('节点数等于5的占比为:{:.2%}'.format(e / totle))
    #     print('节点总数为:{}'.format(totle))

    with open(filenaem, 'r') as load_f:
        load_dict = json.load(load_f)
        for i in load_dict:
            for x in load_dict[i]:
                if load_dict[i][x] != [] :
                    a = a + 1
    return  a
if __name__ == '__main__':
    name = ['./datasets/code/train.pkl', './datasets/code/test.pkl', './datasets/code/valid.pkl']
    # for num, i in enumerate(name):
    #     a = write(num, i)

    a = read_json('腾讯的测试集.json')
    b = read_json('腾讯的验证集.json')
    c = read_json('腾讯的训练集.json')
    print(a+b+c)
# print(g.graph)                      # binary code literal features, we only use c_int and c_str
# print(g.nodes.data('feat'))         # binary code CFG features

