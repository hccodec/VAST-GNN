'''
    改文件从 log.txt 中提取实验结果
    分别是第三波和第四波的 (21,7) (21,14) (21,21) 共六次的实验结果
    每次都分
        test mae,
        test rmse,
        early_stop_epoch,
        result dir timestamp
    四次
'''
import os, re
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("-d", "--dir", default="results/test/exp_test001")
args = parser.parse_args()


result_dir = args.dir
exp = 0
files = os.listdir(result_dir)
assert not len(files) % 6
files = [sorted(files[i: i + 6], key=lambda i: int(i.split('_')[0]) * 100 + int(i.split('_')[2])) for i in range(len(files) // 6)]
# sorted()
files = files[exp]
res = ()
for exp in files:
    path = os.path.join(result_dir, exp, "log.txt")
    with open(path) as f: lines = f.readlines()
    for i in range(len(lines)):
        if "训练完毕，开始评估" in lines[i]:
            break
    try:
        test_metrices = re.search(r"\[test.*\]\ (.*)", lines[i + 1]).groups()
        test_metrices = test_metrices[0].split('/')
        epoch, test_loss = re.search(r"\[Epoch\]\ (\d+).*\[Loss.*?/([\d.]+)", lines[i - 2]).groups()
        res += (*test_metrices, epoch, exp.split('_')[-1])
    except Exception as e:
        res += (('-',) * 4)
print('\t'.join(res))
    
