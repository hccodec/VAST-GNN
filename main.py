import os, sys, torch

import pandas as pd

# sys.path.append(os.getcwd())
from tensorboardX import SummaryWriter

from meta import maml_train
from train_test import train_process, eval_process

from utils.logger import logger
from utils.model_selector import select_model
from utils.utils import font_green, font_yellow, set_random_seed, get_exp_desc
from utils.args import parse_args

from utils.datasets import Datasets
from utils.datetime import datetime

from show_result import show_result

def exp_main(args):
    result_dir          = args.result_dir
    country             = args.country
    model               = args.model
    dataset_cache_dir   = args.dataset_cache_dir
    data_dir            = args.data_dir
    dataset             = args.dataset
    batch_size          = args.batch_size
    xdays               = args.xdays
    ydays               = args.ydays
    window              = args.window
    shift               = args.shift
    train_ratio         = args.train_ratio
    val_ratio           = args.val_ratio
    seed                = args.seed
    seed_dataset        = args.seed_dataset
    node_observed_ratio = args.node_observed_ratio


    if ',' in country:
        result_paths = {
            "args": os.path.join(result_dir, "args.txt"),
        }
    else:
        result_paths = {
            "args": os.path.join(result_dir, country, "args.txt"),
        }
        os.makedirs(os.path.join(result_dir, country), exist_ok=True)

    with open(result_paths["args"], "w", encoding="utf-8") as f:
        f.write("[args]\n")
        for k in args: f.write("{}: {}\n".format(k, args[k]))

    exp_desc = get_exp_desc(model, xdays, ydays, window, shift, node_observed_ratio)

    starttime = datetime.now()
    logger.info(f"Exp [{exp_desc}] begins at {starttime.strftime('%Y-%m-%d %H:%M:%S')}")

    # if args.dataset == 'dataforgood':
    # #     from utils.data_process.dataforgood import load_data
    # elif args.dataset == 'sim':
    #     from utils.data_process.sim import load_data
    # elif args.dataset == 'japan':
    #     from utils.data_process.japan import load_data
    # meta_data = load_data(dataset_cache_dir, data_dir, dataset, batch_size,
    #                     xdays, ydays, window, shift,
    #                     train_ratio, val_ratio, node_observed_ratio)
    meta_data = Datasets(dataset_cache_dir, data_dir, dataset, batch_size,
                        xdays, ydays, window, shift, train_ratio, val_ratio,
                        node_observed_ratio, seed_dataset).load_data()
    
    # DEBUG 若仅生成数据集则打印信息并退出整个程序
    if args.gendata:
        logger.info(f"仅生成数据集，不进行训练")
        return


    logger.info(f"")
    # 根据是否指定国家进行相应训练
    if "country" not in args:
        for i_country in range(len(meta_data["country_names"])):
            train_country(args, result_paths, meta_data, i_country)
    else:
        # 处理 args.country 使之接受形如 "England,Spain" 的参数并整理成数组
        countries = [e for e in country.split(',')]
        if set(countries).issubset(set(meta_data["country_names"])):
            for country in countries:
                i_country = meta_data["country_names"].index(country)
                if args.maml is True:
                    logger.info(f"Meta learning on country {country}")
                    maml_train(args, result_paths, meta_data, i_country)
                else:
                    logger.info(f"Training on country {country}")
                    train_country(args, result_paths, meta_data, i_country)
        else:
            logger.error(f'Arg Error args.country: {country} in {set(meta_data["country_names"])}')

    endtime = datetime.now()
    logger.info(f"Exp [{exp_desc}] finished。\nTotal time cost {(datetime.min + (endtime - starttime)).strftime('%H:%M:%S')} （{starttime.strftime('%Y-%m-%d %H:%M:%S')} - {endtime.strftime('%Y-%m-%d %H:%M:%S')}）")

def train_country(args, result_paths, meta_data, i_country):
    country_name = meta_data["country_names"][i_country]
    country_code = meta_data["country_codes"][i_country]

    # 从 args 读取数据
    lr = args.lr
    lr_min = args.lr_min
    lr_scheduler_stepsize = args.lr_scheduler_stepsize
    lr_weight_decay = args.lr_weight_decay
    lr_scheduler_gamma = args.lr_scheduler_gamma
    epochs = args.epochs
    device = args.device
    early_stop_patience = args.early_stop_patience
    node_observed_ratio = args.node_observed_ratio
    case_normalize_ratio = args.case_normalize_ratio

    seed = args.seed
    

    comp_last = args.comp_last

    # graph_lambda = args.lambda_graph_loss[country_name][(1 + args.shift) if args.ydays == 1 else args.ydays] if country_name in args.lambda_graph_loss else 0
    graph_lambda = args.graph_lambda
    # graph_lambda_0 = args.graph_lambda_0
    # graph_lambda_n = args.graph_lambda_n
    # graph_lambda_epoch_max = args.graph_lambda_epoch_max
    # graph_lambda_method = args.graph_lambda_method
    set_random_seed(seed)  # 设置随机种子

    train_loader, val_loader, test_loader = meta_data['data'][country_name][0]
    del meta_data['data'][country_name]

    logger.info(f"Start training {country_name} ...")

    # 记录实验参数
    result_paths.update(
        {
            "model": os.path.join(
                args.result_dir, f"model_{country_code}_best.pth"
            ),
            "model_latest": os.path.join(
                args.result_dir, f"model_{country_code}_latest.pth"
            ),
            # "csv": os.path.join(args.result_dir, f"results_{country_code}.csv"),
            "tensorboard": os.path.join(
                args.result_dir, f"tensorboard_{country_code}"
            ),
        }
    )

    # 选择模型
    model, model_args = select_model(args, train_loader)
    # criterion = torch.nn.MSELoss()
    criterion = torch.nn.functional.mse_loss

    # 记录模型参数
    with open(result_paths["args"], "a", encoding="utf-8") as f:
        f.write(f"\n[model_args:{country_name}]\n")
        f.write(str(model_args) + "\n")

    # 初始化 tensorboard 记录器
    with SummaryWriter(result_paths["tensorboard"]) as writer:
        losses, trained_model, epoch_best, loss_best = train_process(
            model, criterion, epochs,
            lr, lr_min, lr_scheduler_stepsize, lr_scheduler_gamma, lr_weight_decay,
            train_loader, val_loader, test_loader,
            early_stop_patience, node_observed_ratio, case_normalize_ratio,
            graph_lambda,
            # graph_lambda_0,
            # graph_lambda_n,
            # graph_lambda_epoch_max,
            # graph_lambda_method,
            device, writer, result_paths, comp_last)

        # writer.close()
        torch.save(trained_model.state_dict(), result_paths["model_latest"])

        logger.info("")
        logger.info(f"Training complete. Evaluating: {country_name}")

        logger.info("-" * 20)
        logger.info(font_yellow(f"[Latest (epoch {len(losses['train']) - 1})]"))
        metrics_latest = eval_process(args, result_paths["model_latest"], criterion, train_loader, val_loader, test_loader, comp_last)
        logger.info("-" * 20)
        logger.info(font_yellow(f"[Min val loss (epoch {epoch_best})]"))
        metrics_minvalloss = eval_process(args, result_paths["model"], criterion, train_loader, val_loader, test_loader, comp_last)

        writer.add_hparams(
            {
                **{
                    k: (v if isinstance(v, (int, float, str, bool, torch.Tensor)) else str(v))
                    for k, v in vars(args).items() if k in ["xdays", "ydays", "window", "batch_size", "lr", "lr_min", "seed"]
                },
                **{"country_name": country_name, "country_code": country_code},
            },
            {
                **{f"{k}_minvalloss": float(v) for k, v in metrics_minvalloss.items() if not k == 'outputs'},
                **{f"{k}_latest": float(v) for k, v in metrics_latest.items() if not k == 'outputs'},
            },
        )

def main():
    args = parse_args()
    logger.info(f"Exp results will be saved at {args.result_dir}")
    try:
        exp_main(args)
    finally:
        logger.info(f"Exp results have been saved at {args.result_dir}")

    print()

    # 显示结果
    show_result(os.path.dirname(os.path.dirname(args.result_dir)))

if __name__ == "__main__":
    main()
