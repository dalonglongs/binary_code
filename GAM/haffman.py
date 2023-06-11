from config import get_default_config
import os
import re
import io
import math

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


    for key in dict:
        ret.append((key, int(dict[key][0])))




    return ret, dict


# 节点类
class Node:
    def __init__(self, frequency):
        self.left = None
        self.right = None
        self.father = None
        self.frequency = frequency

    def is_left(self):
        return self.father.left == self


# 创建叶子节点
def create_nodes(frequency_list):
    return [Node(frequency) for frequency in frequency_list]


# 创建Huffman树
def create_huffman_tree(nodes):
    queue = nodes[:]

    while len(queue) > 1:
        queue.sort(key=lambda item: item.frequency)
        node_left = queue.pop(0)
        node_right = queue.pop(0)
        node_father = Node(node_left.frequency + node_right.frequency)
        node_father.left = node_left
        node_father.right = node_right
        node_left.father = node_father
        node_right.father = node_father
        queue.append(node_father)

    queue[0].father = None
    return queue[0]


# Huffman编码
def huffman_encoding(nodes, root):
    huffman_code = [''] * len(nodes)

    for i in range(len(nodes)):
        node = nodes[i]
        while node != root:
            if node.is_left():
                huffman_code[i] = '0' + huffman_code[i]
            else:
                huffman_code[i] = '1' + huffman_code[i]
            node = node.father

    return huffman_code


# 编码整个字符串
def encode_str(text, char_frequency, codes):
    ret = ''
    for char in text:
        i = 0
        for item in char_frequency:
            if char == item[0]:
                ret += codes[i]
            i += 1

    return ret


# 解码整个字符串
def decode_str(huffman_str, char_frequency, codes):
    ret = ''
    while huffman_str != '':
        i = 0
        for item in codes:
            if item in huffman_str and huffman_str.index(item) == 0:
                ret += char_frequency[i][0]
                huffman_str = huffman_str[len(item):]
            i += 1

    return ret
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

def generator(lens = 10):
    sen = []
    for i in range(0, lens):
       sen.append(int(-1))

    return sen

def spilt_(sen, x ,y):
    sens = []
    for i in sen:
        sens.append(int(i)*x*y)
    return sens
def IDF(articles_files, dict):
    for key in dict:
        dict[key].append(0)

    for article in articles_files:
        with open(article, 'r') as f:
            texts = f.read()
            for key in dict:
                if key in texts:
                    dict[key][1] += 1

    for key in dict:
        dict[key][1] = math.log10(len(articles_files)/(dict[key][1]+1))
    return dict

def TF(articles_file, word):
    texts = []
    with open(articles_file, 'r') as f:
        i = 0
        for text in f.readlines():
            if i % 3 == 1:
                text = text.replace("\n", '').split(" ")
                for w in text:
                    texts.append(w)
            i += 1
    print(texts)
    tf = (texts.count(word))/(len(texts))
    print(tf)
    return tf


if __name__ == '__main__':
    config = get_default_config()
    articles_files = build_data(config)
    text = 'The text to encode: the text'
    char_frequency, dict = count_frequency(articles_files)
    dict = IDF(articles_files, dict)
    nodes = create_nodes([item[1] for item in char_frequency])
    root = create_huffman_tree(nodes)
    codes = huffman_encoding(nodes, root)

    huffman_str = encode_str(text, char_frequency, codes)
    origin_str = decode_str(huffman_str, char_frequency, codes)
    i = 0
    with open('recode', 'w') as f:
        for key in dict:
            f.write(str(i) + '\n')
            f.write(str(key) + '\n')
            f.write(str(dict[key]) + "\n")
            dict[key].append(codes[i])
            f.write(codes[i] + '\n')
            i += 1
            f.write('\n')

    for article in articles_files:
        i = 1
        prearticle = article
        article = article.replace('article', 'node_index_to_code')
        with open(article, 'r') as f:
            title = article.replace("node_index_to_code", 'huffman_embedding')
            with open(title, 'w') as g:
                num = 0
                for text in f.readlines():
                    if i % 3 == 2:
                        g.write(str(num) + '\n')
                        words = text.replace('\n', '').split(',')
                        sentence = []
                        for word in words:
                            word = word.replace(" ", "")
                            if word in dict.keys():
                                x = dict[word][2]
                                tf = TF(prearticle, word)
                                sentence = sentence + spilt_(x, tf, dict[word][1])
                            else:
                                sentence = sentence + generator()
                        if len(sentence) < 64:
                            sentence = sentence + generator(64 - len(sentence))
                        elif len(sentence) > 64:
                            sentence = sentence[0:64]
                        if sentence[-1] == ',':
                            print('-1')
                        g.write(str(sentence) + "\n")
                        g.write("\n")
                        num += 1

                    i += 1






