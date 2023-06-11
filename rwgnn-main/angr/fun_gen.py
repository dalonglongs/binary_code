
import os
import pickle as p
import angr
from cre_fil import cre_fil
import signal
import time

def set_timeout(num, callback):
  def wrap(func):
    def handle(signum, frame): # 收到信号 SIGALRM 后的回调函数，第一个参数是信号的数字，第二个参数是the interrupted stack frame.
      raise RuntimeError
    def to_do(*args, **kwargs):
      try:
        signal.signal(signal.SIGALRM, handle) # 设置信号和回调函数
        signal.alarm(num) # 设置 num 秒的闹钟
        # print('start alarm signal.')
        r = func(*args, **kwargs)
        # print('close alarm signal.')
        signal.alarm(0) # 关闭闹钟
        return r
      except RuntimeError as e:
        callback()
    return to_do
  return wrap
def after_timeout(): # 超时后的处理函数
  print("Time out!")
@set_timeout(600, after_timeout) # 限时 2 秒超时
def fun_gen(path):
    exist_fun = []
    try:
        proj = angr.Project(path, load_options={'auto_load_libs': False})
        cfg = proj.analyses.CFGEmulated()

        fun_name = []
        for addr, func in cfg.kb.functions.items():
            fun_name.append(func.name)
        if(len(fun_name) != 0):
            out_path = path.replace('data', 'output')
            print(out_path)
            cre_fil(out_path)
            print('1')


        for i in fun_name:
            addr = proj.loader.find_symbol(i)
            if addr.rebased_addr != 'None':
                exist_fun.append(i)
                start_state = proj.factory.blank_state(addr=addr.rebased_addr)
                cfg = proj.analyses.CFGEmulated(fail_fast=True, starts=[addr.rebased_addr], initial_state=start_state)
        # plot_cfg(cfg, "%s_%s_cfg" % (name, fun_name), asminst=True, vexinst=False, func_addr={addr: True},
        #          debug_info=False, remove_imports=True, remove_path_terminator=True)
                f = open(os.path.join(out_path, i+'.pkl'), 'wb')
                p.dump(cfg, f)
                f.close()
        del fun_name
    except:
        exist_fun = []
    return exist_fun