import angr
project = angr.Project('datas/x86/libtomcrypt-1.18.2-O2/libtomcrypt.a', load_options={'auto_load_libs': False})

cfg = project.analyses.CFGFast()