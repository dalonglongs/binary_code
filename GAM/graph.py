"""
图生成器
"""
import os
import angr
from angrutils import *
from preprocessing import *

def static_graph_generate(binary_file):
    """
    使用 Angr 生成二进制文件的静态 CFG
    :param binary_file: 二进制文件
    :return:
    """
    print(binary_file)
    try:
        proj = angr.Project(binary_file, load_options={'auto_load_libs': False})
        cfg = proj.analyses.CFGFast()
        node_list = list(cfg.graph.nodes)
        edge_list = list(cfg.graph.edges)

    except:
        cfg = []
        node_list = []
        edge_list = []
        dict = []
        return cfg, node_list, edge_list
    return cfg, node_list, edge_list


def show_static_graph(binary_file, output_dir):
    """
    可视化ICFG
    :param binary_file: 二进制文件
    :param output_file: 输出路径
    :return:
    """
    (path, file_name) = os.path.split(binary_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    src_proj = angr.Project(binary_file, load_options={'auto_load_libs': False})
    cfg = src_proj.analyses.CFGFast()
    plot_cfg(cfg, os.path.join(output_dir, file_name), asminst=True)


if __name__ == '__main__':
    file = '../ProjectData/ExperimentData'
    output_file = "../ProjectData/Output"
    show_static_graph(file, output_file)
