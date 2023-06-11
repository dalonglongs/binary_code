import angr
from angrutils import plot_cfg
import pickle as p
import os
# def function_name(b):
#     cfg = proj.analyses.CFGEmulated()
#     fun_name = []
#     for addr, func in cfg.kb.functions.items():
#         fun_name.append(func.name)
#
#     addrs = []
#     # for i in fun_name:
#     #     print(i)
#     print(fun_name)
#     addr = proj.loader.find_symbol('sub_237ec')
#     addr = addr.rebased_addr
#     addrs.append(addr)
#     return fun_name, addrs
#
#
#
# def analyze(b, fun_name, addrs, name=None):
#     for i in range(len(fun_name)):
#         start_state = b.factory.blank_state(addr=addrs[i])
#     # start_state.stack_push(0x0)
#         cfg = b.analyses.CFGEmulated(fail_fast=True, starts=[addrs[i]], initial_state=start_state)
#         plot_cfg(cfg, "%s_%s_cfg" % (name, fun_name[i]), asminst=True, vexinst=False, func_addr={addr: True},
#                      debug_info=False, remove_imports=True, remove_path_terminator=True)

def analyze(b, fun_name, name=None):
    addr = b.loader.find_symbol(fun_name)
    start_state = b.factory.blank_state(addr=addr.rebased_addr)
    # start_state.stack_push(0x0)
    cfg = b.analyses.CFGEmulated(fail_fast=True, starts=[addr.rebased_addr], initial_state=start_state
                                 )
    # for addr, func in proj.kb.functions.items():
    #     print(func.name)
    #     if func.name in ['main', 'verify']:
    f = open(fun_name, 'wb')
    p.dump(cfg, f)
    f.close()
    plot_cfg(cfg, "%s_%s_cfg" % (name, fun_name), asminst=True, vexinst=False, func_addr={addr: True},
                     debug_info=False, remove_imports=True, remove_path_terminator=True)

    # plot_cfg(cfg, "%s_cfg" % (name), asminst=True, vexinst=False, debug_info=False, remove_imports=True,
    #          remove_path_terminator=True)
    # plot_cfg(cfg, "%s_cfg_full" % (name), asminst=True, vexinst=True, debug_info=True, remove_imports=False,
    #          remove_path_terminator=False)


if __name__ == "__main__":
    proj = angr.Project("./data/x86-32/coreutils-8.15-O3/sha224sum", load_options={'auto_load_libs': False})
    print('1')
    cfg = proj.analyses.CFGFast()
    print('2')
    fun_name = []
    for addr, func in cfg.kb.functions.items():
        fun_name.append(func.name)
    print(fun_name)
    names = ['hard_locale','strcmp']

    os.makedirs('./xxx')
    with open('./xxx/fun_name.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        p.dump(fun_name, f)

    # Getting back the objects:
    with open('./xxx/fun_name.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
        print(p.load(f))
    # for i in fun_name:
    #     analyze(proj, i, "test1")
