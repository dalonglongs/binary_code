import io
import os
import re
import collections
import numpy as np
from config import get_default_config

# with open('/home/h/Downloads/binary_diff-master_2/ProjectData/Output/test/coreutils/binaries/coreutils-5.93-O0/basename/node_index_to_embedding') as f:
#     for text in f.readlines():
#         print(type(text))
from config import get_default_config
import os
import re
import io

# 统计字符出现频率，生成映射表
def count_frequency(articles_files):
    chars = []
    ret = []
    dict = {}
    for file in articles_files:
        with open(file, 'r') as f:
            for texts in f.readlines():
                texts = texts.replace("\n", "").lower().split(" ")
                for char in texts:
                    char = char.replace(" ", "")
                    if char not in chars:
                        chars.append(char)
                        dict[char] = []
                        dict[char].append(0)
                    dict[char][0] = int(dict[char][0]) + 1
    for file in articles_files:
        with open(file, 'r') as f:
            texts = f.read()
            print(type(texts))
            break


    for key in dict:
        ret.append((key, int(dict[key][0])))



    return ret, dict

def build_data(config):
    filename_conf = config['file_names']
    binary_output_path_file = os.path.join(config['output_dir'], filename_conf['binary_output_path'])
    if not os.path.exists(binary_output_path_file) or not os.path.isfile(binary_output_path_file):
        raise ValueError('请先构建二进制文件的信息')

    article_files = []
    for line in io.open(binary_output_path_file, 'r'):
        filename = re.findall(r"^(.*)", line)[0]
        if len(filename) > 0:
            article_file = os.path.join(filename, filename_conf['article'])
            if os.path.exists(article_file) and os.path.isfile(article_file):
                article_files.append(article_file)

    if len(article_files) == 0:
        raise ValueError('请先构建二进制文件的信息')
    return article_files
import math
if __name__ == '__main__':
    # config = get_default_config()
    # articles_files = build_data(config)
    # text = 'The text to encode: the text'
    #
    # char_frequency, dict = count_frequency(articles_files)
    a = [1,2,3]
    b = [3]
    for i in a:
        b.append(i)
    print(b)