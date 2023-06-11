import os
import json
import gc
import sys
from angr.fun_gen import fun_gen

if __name__ == '__main__':
    inpath = './data'
    outpath = './output2'

    path_dict = {}
    # cre_fil(outpath)
    except_file = {}
    group = os.walk(inpath)
    for path, dir_list, file_list in group:
        for i in file_list:
            file_path = os.path.join(path, i)
            print(file_path)
            x = fun_gen(file_path)
            if x == []:
                except_file[path] = i
            else:
                path_dict[file_path] = x

            gc.collect()
            print('函数控制流程图生成的路径字典占用的内存为：' + str(sys.getsizeof(path_dict)))
            print('储存angr反编译失败的源程序的字典所占用内存为：' + str(sys.getsizeof(except_file)))
        gc.collect()
    gc.collect()



    json_str = json.dumps(path_dict, indent=4, ensure_ascii=False)
    with open('binary_function_name', 'w') as json_file:
        json_file.write(json_str)

    json_str = json.dumps(except_file, indent=4, ensure_ascii=False)
    with open('expect_file_path', 'w') as json_file:
        json_file.write(json_str)



