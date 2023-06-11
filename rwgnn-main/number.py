import networkx
import os
import pickle
from new8 import get_cfg
import angr

from collections import Counter

inpath = './data/x86-64'
outpath = './output'

pathlist = []
paths = {}
funname = {}
nodenumber = []
com = []
group = os.walk(inpath)
for path, dir_list, file_list in group:
    for i in file_list:

        a = os.path.join(path, i)
        print(a)
        c, b = get_cfg(a)
        # for index, i in enumerate(c):
        #     if 5 <= i <= 7:
        #         com.append(b[index])


# y = dict(Counter(com))
# print ([key for key,value in y.items()if value > 1])

# def 统计节点比率
#     a = 0
#     b = 0
#     c = 0
#     totle = 0
#     for i in nodenumber:
#         # f = open(i, 'rb')
#         # obj = pickle.load(f)
#         # f.close()
#         if i <= 5:
#             a = a + 1
#             totle = totle + 1
#         elif 5 < i <= 20:
#             b = b + 1
#             totle = totle + 1
#         else:
#             c = c + 1
#             totle = totle + 1
#     print('节点数小于5的占比为:{:.2%}'.format(a/(a+b+c)))
#     print('节点数大于5小于20的占比为:{:.2%}'.format(b/(a+b+c)))
#     print('节点数大于20的占比为:{:.2%}'.format(c/(a+b+c)))
#     print(totle)


