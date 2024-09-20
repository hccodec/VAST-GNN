import os, shutil, argparse
from utils.custom_datetime import *

parser = argparse.ArgumentParser()
parser.add_argument('--resdir', default='results', help='目录')
args = parser.parse_args()

result_dir = args.resdir
res = dict()

for subdir in os.listdir(result_dir)[:-1]:
    path = os.path.join(result_dir, subdir)
    if len(os.listdir(path)) < 3:
        print('删除', subdir)
        shutil.rmtree(path)
        continue
    with open(os.path.join(path, 'results_jp.csv')) as f:
        lines = f.readlines()
        if len(lines) < 10:
            print('epoch 不足 10，删除该 epoch')
            shutil.rmtree(path)
        else:
            res[subdir] = len(lines) - 1

print("每个时间的实验所用 epoch 数：")
for i, key in enumerate(res):
    date = date2str(str2date(key, "%Y%m%d%H%M%S"), "%Y年%m月%d日 %H:%m:%S")
    print(f"{i:3d}: [{date}] {res[key]} 个 epoch")