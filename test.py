import torch, os
from train_test import validate_test_process, eval_process
from eval import compute_mae_rmse
from utils.logger import logger
from utils.args import parse_args
from utils.data_process.dataforgood import split_dataset, load_data
import pandas as pd
from argparse import ArgumentParser

from utils.model_selector import select_model
from utils.utils import font_green, get_country, get_exp_desc

def test(fn_model = 'results/results_test/tmp/dataforgood/dynst_7_3_w7_s0_20241005231704/model_EN_best.pth', country = "EN"):
    args = parse_args(record_log=False)
    
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


    res = eval_process(trained_model, criterion, train_loader, val_loader, test_loader, comp_last = False)

    logger.info(font_green(f"完成测试 [{exp_desc}]"))

    return res, meta_data, args

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model-dir",default='results/results_test/tmp/dataforgood/dynst_7_3_w7_s0_20241005231704/')
    parser.add_argument("--country-code", default='EN')
    args = parser.parse_args()
    res = test(args.model_dir, args.country_code)
    print(res)