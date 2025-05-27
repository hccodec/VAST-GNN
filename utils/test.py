import torch, os, re, io, types
from train_test import validate_test_process, eval_process
from eval import compute_mae_rmse
from utils.datasets import Datasets
from utils.logger import logger
from utils.args import get_parser, process_args
import pandas as pd
from argparse import ArgumentParser, Namespace

from tqdm.auto import tqdm
from best_results import paths

from utils.model_selector import select_model
from utils.utils import font_green, font_hide, font_underlined, font_red, get_country, get_exp_desc

def get_args(config_str, **kwargs):
    '''
    使用安全的方式处理读入的 args.txt 文件，
    '''

    def convert(s):
        s = s.strip()
        if s.isdigit(): return int(s)
        elif s.replace('.', '', 1).isdigit(): return float(s)
        elif s.startswith("cuda:"): return torch.device(s).index
        elif s.startswith("device("):
            device = torch.device
            return eval(s).index
        elif s == "False": return False
        elif s == "True": return True
        else: return s
    
    def convert_arg_str(v):
        v_replaced = re.sub(r'\n([a-zA-Z0-9_]+:)', r'<SEP>\1', v)
        res = {e.split(": ")[0]: convert(e.split(": ")[1]) for e in v_replaced.split("<SEP>") if ": " in e}
        if "lambda_graph_loss" in res:
            res["lambda_graph_loss"] = pd.read_csv(io.StringIO(res["lambda_graph_loss"]), sep=r'\s+')
        return res
            
    def convert_dic_str(v):
        # return {e.split(": ")[0]: convert(e.split(": ")[1]) for e in v.split("\n") if ": " in e}
        return {e.split("':")[0]: convert(e.split("':")[1]) for e in v[2:-1].split(", '")}
        
    # 用正则提取 [args] 和 [model_args 开头的块
    pattern = r"(\[(?:args|model_args:[^\]]+)\])(.*?)(?=\[(?:args|model_args:[^\]]+)\]|$)"
    config_blocks = re.findall(pattern, config_str, re.DOTALL)
    
    # 转换为字典
    config_blocks = {block[0][1:-1]: block[1].strip() for block in config_blocks}
    # # 将配置字符串分割成不同的部分
    # config_blocks = [e.split("]\n") for e in config_str.split("[")[1:]]
    # # 修复分割 bug：只保留开头是 "[args]" 和 "[model_args" 的，对于其他，都和前一个合并。提示：从后往前
    # for i in range(len(config_blocks) - 1, 0, -1):
    #     if len(config_blocks[i]) == 1:
    #         config_blocks[i - 1][1] += config_blocks[i][0]
    #         config_blocks.pop(i)

    # config_blocks = {e[0]: e[1].strip() for e in config_blocks if len(e) == 2}

    assert [k == 'args' or k.startswith("model_args") for k in config_blocks]
    device = torch.device


    args, model_args = (
        convert_arg_str(config_blocks["args"]),
        {k.split(":")[-1]: convert_dic_str(v) for k, v in config_blocks.items() if k.startswith("model_args")})
    
    args = types.SimpleNamespace(**args)

    args_old = get_parser().parse_args()
    
    # 若 args_old 和 args 中都有，则用 args 的新值替换掉 args_old 的旧值
    for k, v in vars(args_old).items():
        if k in vars(args):
            setattr(args_old, k, getattr(args, k))
    # 手动矫正一些值
    args_old.node_observed_ratio = args.node_observed_ratio * 100

    # 使用 kwargs 进行覆盖 （比如传入新的 device 号等）
    for k, v in kwargs.items(): setattr(args_old, k, v)
    

    args = process_args(args_old, record_log=False)
    # # 把 args_old 中 args 没有的值加到 args 上
    # for k, v in vars(args_old).items():
    #     if not hasattr(args, k):
    #         setattr(args, k, v)
    

    return args, model_args

country_names = {'EN': "England", 'FR': "France", 'IT': 'Italy', 'ES': 'Spain', 'NZ': 'NewZealand', 'JP': 'Japan'}

def test(
        fn_model = 'results/results_test/tmp/dataforgood/dynst_7_3_w7_s0_20241005231704/model_EN_best.pth',
        logger_disable = None,
        device=7,
        extra_args = None
):

    if logger_disable is True: logger.info = lambda x: None

    country = re.search(r"model_(.*?)_", fn_model).groups()[0]

    arg_path = os.path.join(os.path.dirname(fn_model),
                            (country[0] + 'IM' + country[1]) if country[0] == 'S' and country[1].isdigit() else country_names[country] if country in country_names else country,
                            'args.txt')
    
    if not os.path.exists(arg_path):
        arg_path = os.path.join(os.path.dirname(fn_model), 'args.txt')

    with open(arg_path, encoding='utf-8') as f: args = f.read()
    
    args, model_args = get_args(args, device=device)

    if extra_args is not None:
        for k, v in extra_args.items():
            setattr(args, k, v)
    
    logger.info("参数处理完毕：" + str(args))

    exp_desc = get_exp_desc(args.model, args.xdays, args.ydays, args.window, args.shift, args.node_observed_ratio)

    logger.info(font_green(f"执行测试 [{exp_desc}]"))

    # meta_data = load_data(args.dataset_cache_dir, args.data_dir, args.dataset, args.batch_size,
    #                       args.xdays, args.ydays, args.window, args.shift,
    #                       args.train_ratio, args.val_ratio, args.node_observed_ratio)
    
    meta_data = Datasets(args.dataset_cache_dir, args.data_dir, args.dataset, args.batch_size,
                    args.xdays, args.ydays, args.window, args.shift, args.train_ratio, args.val_ratio,
                    args.node_observed_ratio, args.seed_dataset).load_data()
    
    country_code, country_name = get_country(country, meta_data)

    # country_code = meta_data["country_codes"][i_country]

    train_loader, val_loader, test_loader = meta_data['data'][country_name][0]
    # loss_val, y_real_val, y_hat_val, adj_real_val, adj_hat_val = validate_test_process(trained_model, criterion, val_loader)
    # loss_test, y_real_test, y_hat_test, adj_real_test, adj_hat_test = validate_test_process(trained_model, criterion, test_loader)
    #
    # metrics_val = compute_mae_rmse(y_hat_val.float(), y_real_val.float())
    # metrics_test = compute_mae_rmse(y_hat_test.float(), y_real_test.float())
    
    # 假设你的模型类是 YourModelClass
    trained_model, model_args = select_model(args, train_loader)  # 实例化模型
    trained_model.load_state_dict(torch.load(fn_model, map_location=args.device))
    trained_model.to(args.device)  # 将模型移动到指定设备

    criterion = torch.nn.MSELoss()


    res = eval_process(args, trained_model, criterion, train_loader, val_loader, test_loader, comp_last = False)

    logger.info(font_green(f"完成测试 [{exp_desc}]"))

    return res, meta_data, args

def test_main(paths, k, key, silent=False):
    model_dir = paths[k].loc[key].path
    res, meta_data, args = test(model_dir, silent)

    expected_value, actual_value = float(paths[k].loc[key].mae), float(res['mae_test'])
    
    err_percentage = (actual_value - expected_value) / expected_value

    if abs(err_percentage) < 0.01:
        msg = f'[{font_green("PASSED")}] {k} {str(key):23}'
    else:
        msg = f'[{font_green("FAILED") if err_percentage < 0 else font_red("FAILED")}] {k} {str(key):23} {actual_value:7} ({expected_value:7}) {err_percentage * 100:7.2f}%'
    return msg, model_dir

if __name__ == '__main__':
    # parser = ArgumentParser()
    # parser.add_argument("--model-dir",default='results/results_test/tmp/dataforgood/dynst_7_3_w7_s0_20241005231704/')
    # parser.add_argument("--country-code", default='EN')
    # args = parser.parse_args()
    
    k, key = 'o50', (14, 'ES', 'dynst')
    # k, key = None, None

    # 统计 for k in paths.keys(): for key in paths[k].index: 的和

    if key is None:
        # qbar = tqdm(total=len(paths) * len(paths['o50'].index), desc="Testing models", unit="model")
        i = 1
        for k in paths.keys():
            for key in paths[k].index:
                # if not (k == 'o50' and key[:2] == (3, 'ES')): continue
                msg, model_dir = test_main(paths, k, key, True)
                if 'FAILED' in msg: 
                    print(f"{i:3} {msg} {font_underlined(font_hide(model_dir))}")
                # qbar.write(f"{i} {msg} {model_dir}")
                i += 1
                # qbar.update()
    else:
        msg, model_dir = test_main(paths, k, key)