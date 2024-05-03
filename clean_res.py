# 此脚本用于删除无效results

import os, shutil
from datetime import datetime

os.chdir('results')
fs = os.listdir('')

def cond(f):
    if 1 < len(os.listdir(f)): return False
    if (datetime.now() - datetime.fromtimestamp(f)).total_seconds() < 6 * 3600:
        return False
    return True

[shutil.rmtree(f) for i,f in enumerate(fs) if cond(f)]
