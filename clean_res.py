# 此脚本用于删除无效results

import os, shutil
from datetime import datetime

os.chdir('results')
fs = [item for item in os.listdir('.') if not item.startswith('[')]

def cond(f):
    # print(f'正在检查 {f}')
    if len(os.listdir(f)) > 3:
        # print('已有4个文件')
        return False # 已有4个文件
    if (datetime.now() - datetime.strptime(f, '%Y%m%d%H%M%S')).total_seconds() < 1 * 3600:
        # print('时间差小于 3h')
        return False # 时间差小于 6h
    return True

if not any([cond(f) for f in fs]):
    print('无符合条件的项')
    exit(0)

for i,f in enumerate(fs):
    if cond(f):
        print(f'删除 {f}')
        shutil.rmtree(f)
