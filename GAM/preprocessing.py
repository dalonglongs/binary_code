"""
预处理
"""
import os
from graph import static_graph_generate
import pickle
from node_distance import node_distance

# 8位寄存器
register_list_8_byte = ['rax', 'rcx', 'rdx', 'rbx', 'rsi', 'rdi', 'rsp', 'rbp',
                        'r8', 'r9', 'r10', 'r11', 'r12', 'r13', 'r14', 'r15']
# 4位寄存器
register_list_4_byte = ['eax', 'ecx', 'edx', 'ebx', 'esi', 'edi', 'esp', 'ebp',
                        'r8d', 'r9d', 'r10d', 'r11d', 'r12d', 'r13d', 'r14d', 'r15d']
# 2位寄存器
register_list_2_byte = ['ax', 'cx', 'dx', 'bx', 'si', 'di', 'sp', 'bp',
                        'r8w', 'r9w', 'r10w', 'r11w', 'r12w', 'r13w', 'r14w', 'r15w']
# 1位寄存器
register_list_1_byte = ['al', 'cl', 'dl', 'bl', 'sil', 'dil', 'spl', 'bpl',
                        'r8b', 'r9b', 'r10b', 'r11b', 'r12b', 'r13b', 'r14b', 'r15b']


def node_dict_generate(node_list):
    """
    对节点从0开始进行编号，并使用字典进行保存，k=节点名称，v=编号
    :param src_node_list:
    :param des_node_list:
    :return:
    """
    node_dict = {}
    for i in range(len(node_list)):
        node_dict[node_list[i]] = i
    return node_dict


def normalization(opstr):
    """
    归一化
    :param opstr:
    :return:
    """
    optoken = ''
    if 'ptr' in opstr:
        optoken = 'ptr'
    elif opstr.startswith('0x') or opstr.startswith('-0x') \
            or opstr.replace('.', '', 1).replace('-', '', 1).isdigit():
        optoken = 'im'
    elif opstr in register_list_1_byte:
        optoken = 'reg1'
    elif opstr in register_list_2_byte:
        optoken = 'reg2'
    elif opstr in register_list_4_byte:
        optoken = 'reg4'
    elif opstr in register_list_8_byte:
        optoken = 'reg8'
    elif opstr.startswith('[') and opstr.endswith(']'):
        optoken = 'im'
    else:
        optoken = str(opstr)
    return optoken

def write_cfg(cfg, output_dir, filename_config):
    with open(os.path.join(output_dir, 'cfg_file.pkl'), 'wb') as cfg_file:
        pickle.dump(cfg, cfg_file)

def write_node_distance(cfg, output_dir, filename_config):
    distance, node_dict = node_distance(cfg)
    with open(os.path.join(output_dir, 'node_distance.pkl'), 'wb') as distance_file:
        pickle.dump(distance, distance_file)
    with open(os.path.join(output_dir, 'node_dict.pkl'), 'wb') as dict:
        pickle.dump(node_dict, dict)



def node_index_to_code_generate(node_list, node_dict, output_dir, filename_config):
    """
    对节点进行处理 (对操作地址进行序列化)
    :param node_list:
    :param node_dict:
    :param output_dir
    :return:
    """
    block_idx_to_tokens = {}
    per_block_neighbors_bids = {}
    with open(os.path.join(output_dir, filename_config['node_index_to_code']), 'w') as node_to_index:
        for node in node_list:
            predecessors = node.predecessors  # 当前节点的前身节点
            successors = node.successors  # 当前节点的后继节点

            predecessors_ids = []
            successors_ids = []
            for pred in predecessors:
                predecessors_ids.append(node_dict[pred])
            for succ in successors:
                successors_ids.append(node_dict[succ])

            per_block_neighbors_bids[node_dict[node]] = [predecessors_ids, successors_ids]

            if node.block is None:
                continue

            tokens = []
            for insn in node.block.capstone.insns:
                token = str(insn.mnemonic)
                opstrs = insn.op_str.split(", ")

                for opstr in opstrs:
                        optoken = normalization(opstr)
                        if optoken != '':
                            token += optoken
                tokens.append(token)


            block_idx_to_tokens[str(node_dict[node])] = tokens

            node_to_index.write(str(node_dict[node]) + ':\n')
            node_to_index.write(', '.join(tokens) + '\n\n')

    return block_idx_to_tokens, per_block_neighbors_bids


def edge_list_generate(edge_list, node_dict, output_dir, filename_config):
    """
    记录 ICFG 的边
    :param edge_list:
    :param node_dict:
    :param edge_list_file:
    :return:
    """

    with open(os.path.join(output_dir, filename_config['edge_list']), 'w') as edge_list_file:
        for (src, target) in edge_list:
            edge_list_file.write(str(node_dict[src]) + ' ' + str(node_dict[target]) + '\n')


def preprocessing(binary_file, output_dir, filename_config):
    """
    对二进制文件进行预处理
    :param binary_file: 二进制文件
    :param output_dir: 记录信息的路径
    :return:
    """

    cfg, node_list, edge_list = static_graph_generate(binary_file)
    if cfg == [] or len(node_list) > 500:
        block_idx_to_tokens=[]
        per_block_neighbors_bids=[]
        node_dict=[]
        cfg=[]
        return block_idx_to_tokens, per_block_neighbors_bids, node_dict, cfg

    else:
        node_dict = node_dict_generate(node_list)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # write_cfg(cfg, output_dir, filename_config)
        write_node_distance(cfg, output_dir, filename_config)

        block_idx_to_tokens, per_block_neighbors_bids = node_index_to_code_generate(node_list, node_dict, output_dir,
                                                                                    filename_config)

        edge_list_generate(edge_list, node_dict, output_dir, filename_config)

        return block_idx_to_tokens, per_block_neighbors_bids, node_dict, cfg


if __name__ == '__main__':
    file = 'D:\\Work\\BinaryDiff\\ProjectData\\ExperimentData\\test\\simple0\\simple0'
    output_dir = "D:\\Work\\BinaryDiff\\ProjectData\\Output\\test\\simple0\\"
    preprocessing(file, output_dir, dict(edge_list='edge_list', node_index_to_code='node_index_to_code'))
