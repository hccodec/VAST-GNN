# 此脚本用于删除无效results

import os, shutil
from datetime import datetime

os.chdir('results')
fs = [item for item in os.listdir('.') if not item.startswith('[')]

for f in fs:
    if (datetime.now() - datetime.strptime(f, '%Y%m%d%H%M%S')).total_seconds() < 1 * 3600:
        continue
    files = os.listdir(f)
    if not len(files):
        print(f'删除 {f}')
        shutil.rmtree(f)
    for d in files:
        if d.endswith('_incomplete'):
            to_del = os.path.join(f, d)
            print(f'删除 {to_del}')
            shutil.rmtree(to_del)

print('完成')