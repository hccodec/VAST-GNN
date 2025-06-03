from argparse import ArgumentParser
import zipfile
import torch, os, re, io, types
from train_test import eval_process
from utils.datasets import Datasets
from utils.logger import logger
from utils.args import get_parser, process_args
import pandas as pd

from tqdm.auto import tqdm
from best_results import expexted_maes, expexted_maes_flunet

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

country_names = {
    'EN': "England", 'FR': "France", 'IT': 'Italy', 'ES': 'Spain', 'NZ': 'NewZealand', 'JP': 'Japan',
    'h1n1': 'h1n1', 'h3n2': 'h3n2', 'BY': 'BY', 'BV': 'BV'
    }

def test(
        model_path = 'results/results_test/tmp/dataforgood/dynst_7_3_w7_s0_20241005231704/model_EN_best.pth',
        logger_disable = None,
        device=7,
        extra_args = None
):

    if logger_disable is True: logger.info = lambda x: None

    # 从模型文件提取原信息
    if re.search(r"model_(.*?)_", model_path) is not None:
        country = re.search(r"model_(.*?)_", model_path).groups()[0]
    else:
        dataset, observed_ratio, y, country, model_name = re.search(r"(.*?)_o(.*?)_y(.*?)_(.*?)_(.*).pth", os.path.basename(model_path)).groups()

    # region 处理 args

    args = get_parser().parse_args()
    args.model = model_name
    if model_name == 'mpnn_tl': args.model, args.maml = 'mpnn_lstm', True
    args.dataset = dataset
    args.country = country_names[country]
    args.shift = int(y) - 1
    args.node_observed_ratio = int(observed_ratio)
    args.device = device
    args = process_args(args, record_log=False)

    if extra_args is not None:
        for k, v in extra_args.items():
            setattr(args, k, v)
    
    logger.info("参数处理完毕：" + str(args))
    # endregion

    exp_desc = get_exp_desc(args.model, args.xdays, args.ydays, args.window, args.shift, args.node_observed_ratio)

    logger.info(font_green(f"执行测试 [{exp_desc}]"))

    meta_data = Datasets(args.dataset_cache_dir, args.data_dir, args.dataset, args.batch_size,
                    args.xdays, args.ydays, args.window, args.shift, args.train_ratio, args.val_ratio,
                    args.node_observed_ratio, args.seed_dataset).load_data()
    
    country_code, country_name = get_country(country, meta_data)

    train_loader, val_loader, test_loader = meta_data['data'][country_name][0]
    # loss_val, y_real_val, y_hat_val, adj_real_val, adj_hat_val = validate_test_process(trained_model, criterion, val_loader)
    # loss_test, y_real_test, y_hat_test, adj_real_test, adj_hat_test = validate_test_process(trained_model, criterion, test_loader)
    #
    # metrics_val = compute_mae_rmse(y_hat_val.float(), y_real_val.float())
    # metrics_test = compute_mae_rmse(y_hat_test.float(), y_real_test.float())
    
    trained_model, model_args = select_model(args, train_loader)  # 实例化模型
    trained_model.load_state_dict(torch.load(model_path, map_location=args.device))
    trained_model.to(args.device)  # 将模型移动到指定设备

    criterion = torch.nn.MSELoss()


    res = eval_process(args, trained_model, criterion, train_loader, val_loader, test_loader, comp_last = False)

    logger.info(font_green(f"完成测试 [{exp_desc}]"))

    return res, meta_data, args

def test_main(model_path, expected_mae_value, observed_ratio, y, country_code, model_name, silent=False, device='cpu'):

    res, meta_data, args = test(model_path, logger_disable=silent, device=device)

    expected_mae_value, actual_mae_value = float(expected_mae_value), float(res['mae_test'])
    
    err_percentage = (actual_mae_value - expected_mae_value) / expected_mae_value

    if abs(err_percentage) < 0.01:
        msg = f'[{font_green("PASSED")}] {observed_ratio} {"{:2} {:4} {:9}".format(y, country_code, model_name)}'
    else:
        msg = f'''[{
            font_green("FAILED") if err_percentage < 0 else font_red("FAILED")
            }] {observed_ratio} {"{:2} {:4} {:9}".format(y, country_code, model_name)} {actual_mae_value:7} ({expected_mae_value:7}) {err_percentage * 100:7.2f}%'''
    return msg, model_path

if __name__ == '__main__':

    device = 9

    dataset, observed_ratio, y, country_code, model_name = 'dataforgood', 'o50', 14, 'ES', 'dynst'
    dataset, observed_ratio, y, country_code, model_name = None, None, None, None, None

    qbar_enabled = True

    if dataset is None:
        pth_zip_filename = "checkpoints.zip"
        pth_zip_dirname = "checkpoints"
        if os.path.exists(pth_zip_filename):
            if not os.path.exists(pth_zip_dirname): zipfile.ZipFile(pth_zip_filename).extractall()
            
            if qbar_enabled: qbar = tqdm(
                total=len(os.listdir(pth_zip_dirname)),
                desc="Testing models", unit="model")
            i = 1

            keys = [re.search(r"(.*?)_(.*?)_y(.*?)_(.*?)_(.*).pth", fn).groups() for fn in os.listdir(pth_zip_dirname) if fn.endswith('.pth')]
            i = 1
            for dataset, observed_ratio, y, country_code, model_name in keys:
                paths_dataset = expexted_maes_flunet if dataset == 'flunet' else expexted_maes
                model_path = os.path.join(pth_zip_dirname, "{}_{}_y{}_{}_{}.pth".format(dataset, observed_ratio, y, country_code, model_name))
                args_path = os.path.join(pth_zip_dirname, "{}_{}_y{}_{}_{}_args.txt".format(dataset, observed_ratio, y, country_code, model_name))
                msg, model_path = test_main(
                    model_path,
                    paths_dataset[observed_ratio].loc[int(y), country_code, model_name].mae,
                    observed_ratio, y, country_code, model_name, True, device)
                msg_print = f"{i:3} {msg} {font_underlined(font_hide(model_path))}"
                if True or 'FAILED' in msg:
                    if qbar_enabled: qbar.write(msg_print)
                    else: print(msg_print)
                i += 1
                if qbar_enabled: qbar.update()
            print()
        else:
            print(f"Please check if checkpoint zip file {pth_zip_filename} exists.")
    else:
        paths_dataset = expexted_maes_flunet if dataset == 'flunet' else expexted_maes
        msg, model_path = test_main(expexted_maes[observed_ratio].loc[y, country_code, model_name].path,
                                    expexted_maes[observed_ratio].loc[y, country_code, model_name].mae,
                                    observed_ratio, y, country_code, model_name, False, device)