import angr
import numpy as np
from angrutils import *
import matplotlib.pyplot as plt
import time
import networkx as nx

def node_distance(cfg):
    graph = list(cfg.graph.nodes)
    dict_node = {}
    n = 0

    for i in graph:
        dict_node[n] = i
        n = n + 1
    node_dis_dict = {}
    node_dis = dict(nx.all_pairs_dijkstra_path_length(cfg.graph))

    for i in dict_node:
        for j in dict_node:
            if dict_node[j] in node_dis[dict_node[i]].keys():
                node_dis_dict[i] = {j: node_dis[dict_node[i]][dict_node[j]]}
                print(node_dis[dict_node[i]][dict_node[j]])

    # for i in dict:
    #     for j in dict:
    #         try:
    #             # node_dis_dict[i] = [j, nx.astar_path_length(cfg.graph, dict[i], dict[j])]
    #             node_dis_dict = nx.floyd_warshall)
    #         except:
    #             node_dis_dict[i] = [j, -1]
    return node_dis_dict, dict_node
# if __name__ == '__main__':
#     '''
#     all_pairs_bellman_ford_path_length(cfg.graph)
#     nx.astar_path_length
#     '''
#     proj = angr.Project('../ProjectData/ExperimentData/test/arm-32/binutils-2.30-O0/ar', load_options={'auto_load_libs': False})
#     cfg = proj.analyses.CFGFast()
#     print('11111')
#     node = list(cfg.graph.nodes)
#     print(len(node))
#     dict = {}
#     n = 0
#     # leng = dict(nx.all_pairs_bellman_ford_path_length(cfg.graph))
#     # print(leng)
#     for i in node:
#         dict[n] = i
#         n = n + 1
#     node_dis_dict = {}
#     for i in dict:
#         for j in dict:
#             print(j)
#             try:
#                 node_dis_dict[i] = [j, nx.bellman_ford_path_length(cfg.graph, dict[i], dict[j])]
#             except:
#                 node_dis_dict[i] = [j, -1]
#
#
#
#     t_4 = time.time()
#     print(node_dis_dict)
#     print('111111')
