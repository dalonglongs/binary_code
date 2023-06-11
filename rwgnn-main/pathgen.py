import os
group = os.walk('./arm-32')
for path, dir_list, file_list in group:
    print(file_list)