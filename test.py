import torch, os, re, io, types
from train_test import validate_test_process, eval_process
from eval import compute_mae_rmse
from utils.logger import logger
from utils.args import get_parser, process_args
from utils.data_process.dataforgood import split_dataset, load_data
import pandas as pd
from argparse import ArgumentParser, Namespace

from utils.model_selector import select_model
from utils.utils import font_green, get_country, get_exp_desc

def get_args(config_str):
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
        
    # 将配置字符串分割成不同的部分
    config_blocks = [e.split("]\n") for e in config_str.split("[")[1:]]
    config_blocks = {e[0]: e[1].strip() for e in config_blocks if len(e) == 2}

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
    

    args = process_args(args_old, record_log=False)
    # # 把 args_old 中 args 没有的值加到 args 上
    # for k, v in vars(args_old).items():
    #     if not hasattr(args, k):
    #         setattr(args, k, v)
    

    return args, model_args

def test(fn_model = 'results/results_test/tmp/dataforgood/dynst_7_3_w7_s0_20241005231704/model_EN_best.pth', country = "EN"):
    
    arg_path = os.path.join(os.path.dirname(fn_model), 'args.txt')
    with open(arg_path, encoding='utf-8') as f: args = f.read()
    
    args, model_args = get_args(args)

    exp_desc = get_exp_desc(args.model, args.xdays, args.ydays, args.window, args.shift, args.node_observed_ratio)

    logger.info(font_green(f"执行测试 [{exp_desc}]"))

    meta_data = load_data(args.dataset_cache_dir, args.data_dir, args.dataset, args.batch_size,
                          args.xdays, args.ydays, args.window, args.shift,
                          args.train_ratio, args.val_ratio, args.node_observed_ratio)
    
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

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model-dir",default='results/results_test/tmp/dataforgood/dynst_7_3_w7_s0_20241005231704/')
    parser.add_argument("--country-code", default='EN')
    args = parser.parse_args()
    res = test(args.model_dir, args.country_code)
    logger.info(res)