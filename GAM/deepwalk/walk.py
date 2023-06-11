"""
随机游走
"""
import random
from deepwalk import graph


def random_walks_generator(edge_list_file, number_walks, walk_length, alpha, seed):
    """
    生成随机游走
    :param alpha:
    :param edge_list_file: 记录CFG的边的文件
    :param number_walks: 每个节点的游走的次数
    :param walk_length: 每次随机游走的长度
    :param seed: 随机种子
    :return:
    """
    graph_dict = graph.load_edgelist(edge_list_file)
    walks = graph.build_deepwalk_corpus(graph_dict, num_paths=number_walks, path_length=walk_length, alpha=alpha,
                                        rand=random.Random(seed))

    return walks

if __name__ == '__main__':
    edgelist_file = 'D:\\Work\\BinaryDiff\\ProjectData\\Output\\test\\simple0\\edge_list'
    random_walks_generator(edgelist_file, number_walks=2, walk_length=5, alpha=0, seed=0)