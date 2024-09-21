import os, sys, torch

sys.path.append(os.getcwd())
from tensorboardX import SummaryWriter

from train_test import train_process, eval_process

from utils.logger import logger
from utils.utils import font_green, font_yellow, select_model, set_random_seed, parse_args

def exp_main(args):

    from utils.data_process.dataforgood import load_data, meta_info

    meta_data = load_data(args)

    result_paths = {
        "log": os.path.join(args.result_dir, "log.txt"),
        "help": os.path.join(args.result_dir, "help.txt"),
    }

    for i_country in range(len(meta_data)):
        country_name = meta_info["name"][i_country]
        country_code = meta_info["code"][i_country]

        (
            (train_loader, validation_loader, test_loader),
            (train_origin, validation_origin, test_origin),
            (train_indices, validation_indices, test_indices),
        ) = meta_data[country_name]

        logger.info(f"开始训练 {country_name}")

        # 记录实验参数
        result_paths.update(
            {
                "model": os.path.join(
                    args.result_dir, f"model_{country_code}_best.pth"
                ),
                "model_latest": os.path.join(
                    args.result_dir, f"model_{country_code}_latest.pth"
                ),
                "csv": os.path.join(args.result_dir, f"results_{country_code}.csv"),
                "tensorboard": os.path.join(
                    args.result_dir, f"tensorboard_{country_code}"
                ),
            }
        )

        with open(result_paths["help"], "w") as f:
            f.write("[args]\n")
            for k, v in args._get_kwargs():
                f.write("{}: {}\n".format(k, v))

        # 选择模型
        model, model_args = select_model(args, train_loader)
        # criterion = torch.nn.MSELoss()
        criterion = torch.nn.functional.mse_loss

        # 记录模型参数
        with open(result_paths["help"], "a") as f:
            f.write("\n[model_args]\n")
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
            case_normalize_ratio = args.case_normalize_ratio
            enable_graph_learner = args.enable_graph_learner
            (
                graph_lambda_0,
                graph_lambda_k,
                graph_lambda_method,
            ) = (
                args.graph_lambda_0,
                args.graph_lambda_k,
                args.graph_lambda_method,
            )

            # start_date, end_date, x_days, y_days = args.startdate, args.enddate, args.xdays, args.ydays
            # data_dir, case_normalize_ratio, text_normalize_ratio = args.data_dir, args.case_normalize_ratio, args.text_normalize_ratio

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
                validation_loader,
                test_loader,
                early_stop_patience,
                case_normalize_ratio,
                graph_lambda_0,
                graph_lambda_k,
                graph_lambda_method,
                device,
                writer,
                result_paths,
            enable_graph_learner
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
                validation_loader,
                test_loader,
                args.ydays,
                case_normalize_ratio,
                device,
            )
            
            logger.info("[val(MAE/RMSE)] {:.3f}/{:.3f}, [test(MAE/RMSE)] {:.3f}/{:.3f}".format(*list(metrics_latest.values())[:4]))
            logger.info(f"[err_val] {metrics_latest['err_val']:.3f}, [err_test] {font_green(metrics_latest['err_test'])}")

            logger.info("-" * 20)
            logger.info(font_yellow(f"[最小 val loss (epoch {epoch_best})]"))
            metrics_minvalloss = eval_process(
                result_paths["model"],
                criterion,
                train_loader,
                validation_loader,
                test_loader,
                args.ydays,
                case_normalize_ratio,
                device,
            )
            
            logger.info("[val(MAE/RMSE)] {:.3f}/{:.3f}, [test(MAE/RMSE)] {:.3f}/{:.3f}".format(*list(metrics_minvalloss.values())[:4]))
            logger.info(f"[err_val] {metrics_minvalloss['err_val']:.3f}, [err_test] {font_green(metrics_minvalloss['err_test'])}")

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
                            "batchsize",
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
            # writer.close()

    logger.info(f"实验 dataforgood（ 预测范围 {args.xdays}->{args.ydays} w{args.window}）结束")


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


if __name__ == "__main__":
    main()
