import angr
from angrutils import *
import networkx
proj = angr.Project("add.exe",load_options={'auto_load_libs': False})
# cfg = proj.analyses.CFG(normalize=True)
# f = cfg.kb.functions
# for addr, func in f.items():
#     # print(hex(addr), func.name)
#     if func.name == 'main':
#         add = int(addr, 16)
# print(type(add))
# cfg = proj.analyses.CFGEmulated(starts=[add])
#从符号表获取main函数对象
# main = proj.loader.main_object.get_symbol("_main")
#生成起点从main函数开始的cfg

# cfg = proj.analyses.CFGEmulated(starts=[main.rebased_addr])
# cfg = proj.analyses.CFGEmulated()
# print(cfg.get_any_node())
# print(len(cfg.graph.nodes()))

#寻找main函数
# main = proj.loader.find_symbol('main')
# print(main)
# print(main.rebased_addr)
# main_obj=proj.loader.main_object.get_symbol("main")
# print(main_obj)

# cfg = proj.analyses.CFGEmulated()
# print(len(cfg.graph.nodes()))
# cfg = proj.analyses.CFGEmulated(starts=[main.rebased_addr])
# print(len(cfg.graph.nodes()))
# start_state=proj.factory.blank_state(addr=main_obj.addr)
# cfg=proj.analyses.CFGAccurate(fail_fast=True,starts=[main_obj.addr],initial_state=start_state)

# main = proj.loader.main_object.get_symbol("main")
# print(main)
# start_state = proj.factory.blank_state(addr=main.rebased_addr)
# cfg = proj.analyses.CFGFast()


print('1')
# 调用angrutils的方法来画图
# plot_cfg(cfg,"cfg",format='png', asminst=True,remove_imports=True)
# plot_cfg(cfg, "static", asminst=True, remove_imports=True, remove_path_terminator=True)
# plot_cfg(cfg,"myappmain",asminst=True,remove_imports=True,remove_path_terminator=True)

cfg = p.analyses.CFG(normalize=True)
f = cfg.kb.functions
for addr, func in f.items():
    if func.name == name:
        print(hex(addr), func.name)
        plot_cfg(cfg, "%s/%s"%(CFGAddr, func.name), asminst=True, vexinst=False, func_addr={addr: True},
                 debug_info=False, remove_imports=True, remove_path_terminator=True) # asminst 基本块内汇编指令可视化； vexinst 基本块内VEX IR可视化