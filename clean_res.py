# 此脚本用于删除无效results

import os, shutil
from datetime import datetime

os.chdir('results')
fs = os.listdir('.')

def cond(f):
    if len(os.listdir(f)) > 3: return False # 已有4个文件
    if (datetime.now() - datetime.strptime(f, '%Y%m%d%H%M%S')).total_seconds() < 6 * 3600:
        return False # 时间差小于 h
    return True

[shutil.rmtree(f) for i,f in enumerate(fs) if cond(f)]
