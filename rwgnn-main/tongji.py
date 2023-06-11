import os
inpath = './统计文件夹'
a = 0
group = os.walk(inpath)
for path, dir_list, file_list in group:
    for i in file_list:
        file_path = os.path.join(path, i)
        a = a + 1
print(a)