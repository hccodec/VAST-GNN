from utils.test import test
from utils.utils import get_exp_desc
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-c", "--country", choices=['EN', 'FR', 'ES', 'IT', 'JP', 'h1n1', 'h3n2', 'BY', 'BV'], default='EN', help="国家")
parser.add_argument("-m", "--model", choices=['lstm', 'mpnn_lstm', 'mpnn_tl', 'vast_gnn'], default='vast_gnn', help="模型名称")
parser.add_argument("-n", "--node-observed-ratio", choices=['80', '50'], default=80, help="节点保留比例")
parser.add_argument("-y", "--ydays", choices=['3', '7', '14'], default=3, help="预测未来第几天") # 不使用 ydays，避免与 main.py 使用的 argumentparser 冲突
args = parser.parse_args()

if args.country in ['EN', 'FR', 'ES', 'IT']: args.dataset='dataforgood'
elif args.country in ['h1n1', 'h3n2', 'BY', 'BV']: args.dataset='flunet'
elif args.country in ['JP']: args.dataset='japan'
else: raise KeyError('Arg ERROR')

res, meta_data, args = test(f"checkpoints/{args.dataset}_o{args.node_observed_ratio}_y{args.ydays}_{args.country}_{args.model}.pth", logger_disable=True)

print()
print(get_exp_desc(args.model, args.xdays, args.ydays, args.window, args.shift, args.node_observed_ratio, language='en'))
print(f'for country {args.country} on {args.dataset} dataset.')
print(f'[MAE] {res["mae_test"]:.3f}')
if "hits10_test" in res: print(f'[HITS@10] {res["hits10_test"]:.3f}')
print()
