"""
训练数据集
"""
import io
import os
import re
import collections
import numpy as np
from config import get_default_config


def build_label(config):
    """
    创建标签，不同编译器下的同名文件标为 1 (相似)，不同文件名标为 -1 (不相似)
    :param config:
    :return:
    """
    filename_conf = config['file_names']
    output_dir = config['output_dir']

    binary_output_path_file = os.path.join(output_dir, filename_conf['binary_output_path'])
    if not os.path.exists(binary_output_path_file) or not os.path.isfile(binary_output_path_file):
        raise ValueError('请先构建二进制文件的信息')


    binary_output_paths = []
    filenames = []
    binary_dataset_type = []
    dataset_type = config['graph_match']['dataset_type']
    for line in io.open(binary_output_path_file, 'r'):
        path_str = re.findall(r"^(.*)", line)[0]
        if len(path_str) > 0:
            if os.path.exists(path_str) and os.path.isdir(path_str):
                path, filename = os.path.split(path_str)
                binary_output_paths.append(path_str)
                filenames.append(filename)
                for i in range(len(dataset_type)):
                    if os.sep + dataset_type[i] + os.sep in path_str:
                        binary_dataset_type.append(i)

    if len(binary_output_paths) == 0:
        raise ValueError('请先构建二进制文件的信息')

    label_writer = io.open(os.path.join(output_dir, filename_conf['graph_match_labels']), 'w')
    for i in range(len(binary_output_paths)):
        for j in range(i+1, len(binary_output_paths)):
            if filenames[i] == filenames[j] and binary_dataset_type[i] == binary_dataset_type[j]:
                label_writer.write(str(1) + '\n')
            else:
                label_writer.write(str(-1) + '\n')
            label_writer.write(binary_output_paths[i] + '\n')
            label_writer.write(binary_output_paths[j] + '\n\n')
