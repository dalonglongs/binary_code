import os
import stat


def is_elf_file(filepath):
    """
    判断文件是否是elf文件
    :param filepath:
    :return:
    """
    if not os.path.exists(filepath):
        return False
    try:
        file_states = os.stat(filepath)
        file_mode = file_states[stat.ST_MODE]
        if not stat.S_ISREG(file_mode) or stat.S_ISLNK(file_mode):
            return False
        with open(filepath, 'rb') as f:
            header = (bytearray(f.read(4))[1:4]).decode(encoding="utf-8")
            if header in ["ELF"]:
                return True
    except UnicodeDecodeError as e:
        pass

    return False
