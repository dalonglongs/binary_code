import os
def cre_fil(path):
    floder = os.path.exists(path)
    if not floder:
        os.makedirs(path)
        print(path + '路径创建成功')