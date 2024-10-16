import os, sys, torch

import pandas as pd

# sys.path.append(os.getcwd())
from tensorboardX import SummaryWriter

from meta import maml_train
from train_test import train_process, eval_process

from utils.logger import logger
from utils.utils import font_green, font_yellow, select_model, set_random_seed, get_exp_desc
from utils.args import parse_args

from utils.custom_datetime import datetime

from show_result import show_result

def exp_main(args):
    exp_desc = get_exp_desc(args.model, args.xdays, args.ydays, args.window, args.shift, args.node_observed_ratio)

    starttime = datetime.now()
    logger.info(f"实验 [{exp_desc}] 开始于 {starttime.strftime('%Y-%m-%d %H:%M:%S')}")

    from utils.data_process.dataforgood import load_data

    meta_data = load_data(args)

    logger.info(f"")

    result_paths = {
        "log": os.path.join(args.result_dir, "log.txt"),
        "args": os.path.join(args.result_dir, "args.txt"),
    }

    with open(result_paths["args"], "w") as f:
        f.write("[args]\n")
        for k in args: f.write("{}: {}\n".format(k, args[k]))


    # 根据是否指定国家进行相应训练
    if args.maml is True:
        maml_train(meta_data)
    elif "country" not in args:
        for i_country in range(len(meta_data["country_names"])):
            train_country(args, result_paths, meta_data, i_country)
    else:
        # 处理 args.country 使之接受形如 "England,Spain" 的参数并整理成数组
        countries = [e.title() for e in args.country.split(',')]
        if set(countries).issubset(set(meta_data["country_names"])):
            logger.info(f"将对国家 {args.country} 的数据进行训练")
            for country in countries:
                i_country = meta_data["country_names"].index(country)
                train_country(args, result_paths, meta_data, i_country)
        else:
            print(f"参数错误 args.country: {args.country}")

    endtime = datetime.now()
    logger.info(f"实验 [{exp_desc}] 结束。\n总用时 {(datetime.min + (endtime - starttime)).strftime('%H:%M:%S')} （{starttime.strftime('%Y-%m-%d %H:%M:%S')} - {endtime.strftime('%Y-%m-%d %H:%M:%S')}）")

def train_country(args, result_paths, meta_data, i_country):
    country_name = meta_data["country_names"][i_country]
    country_code = meta_data["country_codes"][i_country]

    train_loader, val_loader, test_loader = meta_data['data'][country_name][0]

    logger.info(f"开始训练 {country_name} ...")

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
    with open(result_paths["args"], "a") as f:
        f.write(f"\n[model_args:{country_name}]\n")
        f.write(str(model_args) + "\n")

    # 初始化 tensorboard 记录器
    with SummaryWriter(result_paths["tensorboard"]) as writer:
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

        comp_last = args.comp_last

        graph_lambda_0 = args.graph_lambda_0
        graph_lambda_n = args.graph_lambda_n
        graph_lambda_epoch_max = args.graph_lambda_epoch_max
        graph_lambda_method = args.graph_lambda_method

        losses, trained_model, epoch_best, loss_best = train_process(
            model,
            criterion,
            epochs,
            lr,
            lr_min,
            lr_scheduler_stepsize,
            lr_scheduler_gamma,
            lr_weight_decay,
            train_loader,
            val_loader,
            test_loader,
            early_stop_patience,
            node_observed_ratio,
            case_normalize_ratio,
            graph_lambda_0,
            graph_lambda_n,
            graph_lambda_epoch_max,
            graph_lambda_method,
            device,
            writer,
            result_paths,
            comp_last
        )

        # writer.close()
        torch.save(trained_model, result_paths["model_latest"])

        logger.info("")
        logger.info(f"训练完毕，开始评估: {country_name}")

        logger.info("-" * 20)
        logger.info(font_yellow(f"[最新 (epoch {len(losses['train']) - 1})]"))
        metrics_latest = eval_process(
            result_paths["model_latest"],
            criterion,
            train_loader,
            val_loader,
            test_loader,
            comp_last
        )
        logger.info("-" * 20)
        logger.info(font_yellow(f"[最小 val loss (epoch {epoch_best})]"))
        metrics_minvalloss = eval_process(
            result_paths["model"],
            criterion,
            train_loader,
            val_loader,
            test_loader,
            comp_last
        )

        writer.add_hparams(
            {
                **{
                    k: (
                        v
                        if isinstance(v, (int, float, str, bool, torch.Tensor))
                        else str(v)
                    )
                    for k, v in vars(args).items()
                    if k
                       in [
                           "xdays",
                           "ydays",
                           "window",
                           "batch_size",
                           "lr",
                           "lr_min",
                           "seed",
                       ]
                },
                **{"country_name": country_name, "country_code": country_code},
            },
            {
                **{
                    f"{k}_minvalloss": float(v)
                    for k, v in metrics_minvalloss.items()
                },
                **{f"{k}_latest": float(v) for k, v in metrics_latest.items()},
            },
        )

def main():
    args = parse_args()
    set_random_seed(args.seed)
    logger.info(f"运行结果将保存至 {args.result_dir}")
    try:
        if args.dataset == "dataforgood":
            exp_main(args)
        else:
            raise ValueError(f"数据集 {args.dataset} 不存在")
    finally:
        logger.info(f"实验结果已保存至 {args.result_dir}")

    print()

    # 显示结果
    show_result(os.path.dirname(os.path.dirname(args.result_dir)))

if __name__ == "__main__":
    main()
