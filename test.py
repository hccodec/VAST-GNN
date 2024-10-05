import torch, os
from train_test import validate_test_process, eval_process
from eval import compute_mae_rmse
from utils.args import parse_args
from utils.data_process.dataforgood import split_dataset, load_data
import pandas as pd

pth_name = 'best_model_jp.pth'
log_name = "log.txt"

if __name__ == '__main__':

    args = parse_args(False)
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--model-dir", default='results/results_test/tmp/dataforgood/dynst_7_7_w7_s0_20241003000136/')
    parser.add_argument("--country-code", default='EN')

    args.model_dir = parser.parse_args().model_dir
    args.country_code = parser.parse_args().country_code

    trained_model = torch.load(os.path.join(args.model_dir, f"model_{args.country_code}_best.pth"))

    criterion = torch.nn.MSELoss()

    meta_data = load_data(args)
    i_country = meta_data["country_codes"].index(args.country_code)
    country_name = meta_data["country_names"][i_country]
    # country_code = meta_data["country_codes"][i_country]

    train_loader, val_loader, test_loader = meta_data['data'][country_name][0]
    # loss_val, y_real_val, y_hat_val, adj_real_val, adj_hat_val = validate_test_process(trained_model, criterion, val_loader)
    # loss_test, y_real_test, y_hat_test, adj_real_test, adj_hat_test = validate_test_process(trained_model, criterion, test_loader)
    #
    # metrics_val = compute_mae_rmse(y_hat_val.float(), y_real_val.float())
    # metrics_test = compute_mae_rmse(y_hat_test.float(), y_real_test.float())
    res = eval_process(trained_model, criterion, train_loader, val_loader, test_loader, comp_last = False)
    print(res)
