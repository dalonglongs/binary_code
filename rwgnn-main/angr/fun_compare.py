import angr

def function_name(proj):
    cfg = proj.analyses.CFGEmulated()
    fun_name = []
    for addr, func in cfg.kb.functions.items():
        fun_name.append(func.name)
    return fun_name

def idf_fun(proj):
    idf_funs = []
    idfer = proj.analyses.Identifier()
    for funcInfo in idfer.func_info:
        idf_funs.append(funcInfo.name)
if __name__ == '__main__':
    proj_arm_O0 = angr.Project("./arm-32/binutils-2.30-O0/ar", load_options={'auto_load_libs': False})
    proj_arm_O1 = angr.Project("./arm-32/binutils-2.30-O1/ar", load_options={'auto_load_libs': False})
    proj_x86_O0 = angr.Project("./x86-64/binutils-2.30-O0/ar", load_options={'auto_load_libs': False})
    fun_name1 = function_name(proj_arm_O0)
    fun_name2 = function_name(proj_arm_O1)
    fun_name3 = function_name(proj_x86_O0)
    idf_fun1 = idf_fun(proj_arm_O0)
    idf_fun2 = idf_fun(proj_arm_O1)
    idf_fun3 = idf_fun(proj_x86_O0)
    he = [x for x in fun_name1 if x in fun_name2]
    sun = [x for x in he if x in fun_name3]
    # idfer = [x for x in sun if x in idf_fun1]
    # idfer1 = [x for x in idfer if x in idf_fun2]
    # idfer2 = [x for x in idfer1 if x in idf_fun3]
    print(idf_fun1)
    print(type(idf_fun1))
    # print(len(sun))
    # print(sun)
    # print("--------------------")
    # print("--------------------")
    # print(len(idfer2))
    # print(idfer2)
