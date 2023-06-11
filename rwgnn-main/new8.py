import angr
import networkx as nx
import matplotlib.pyplot as plt
from angrutils import *
def get_cfg(file_name):
    project = angr.Project(file_name, load_options={'auto_load_libs': False})

    cfg = project.analyses.CFGFast()
    funcs_addr_set = cfg.kb.functions.function_addrs_set
    a = []
    b = []
    for func_addr in iter(funcs_addr_set):
        func = cfg.kb.functions[func_addr]
        # func_graph = func.graph
        #
        # # function info
        #
        # # 函数所在的二进制文件名称
        # binary_name = func.binary_name  # str
        #
        # # 基本块的地址
        # block_addrs = func.block_addrs  # dict
        # block_addrs_set = func.block_addrs_set  # set
        #
        # has_return = func.has_return  # bool
        #
        # # 检测的不精确性有关。有时无法检测到间接跳转/调用的目标，如果发生在函数内，则为True
        # has_unresolved_calls = func.has_unresolved_calls  # bool
        # has_unresolved_jumps = func.has_unresolved_jumps  # bool
        #
        # # 函数是否为 PLT 条目
        # is_plt = func.is_plt  # bool
        #
        # # 函数原型
        # # is_prototype_guessed = func.is_prototype_guessed  # bool
        # # 函数是否为简单程序
        # # is_simprocedure = func.is_simprocedure  # bool
        #
        # # 是否为系统调用
        # is_syscall = func.is_syscall  # bool
        #
        # 函数名称
        name = func.name  # str, function name
        b.append(name)
        # # 指定了name，否则 name = self._get_initial_name()
        # is_default_name = func.is_default_name  #bool
        # # 结构的函数名。如果 name[0:2] == "_Z" 则 parse(self.name)
        # demangled_name = func.demangled_name  # str
        #
        # # 是否归一化(参考Angr与IDA的区别)
        # normalized = func.normalized

        # print('function name:', name)
        func_graph = func.graph
        # print('节点数：', len(list(func_graph.nodes())))
        a.append(len(list(func_graph.nodes())))
        # # print(project.factory.block(func.addr).instructions)
        # print('------------------------------')
        # for n in func_graph:
        #     block = project.factory.block(n.addr)
        #     block.pp()
        #     print('---------')

        # if(func.name == 'MHD_set_connection_value'):
        #     func_graph = func.graph
        #     nx.draw_shell(func_graph)
        #     plt.show()
    return a,b