from argparse import ArgumentParser
import os, sys, torch
sys.path.append(os.getcwd())
from torch.utils.tensorboard import SummaryWriter

from utils.custom_datetime import date2str, datetime
from utils.data_process import load_data, split_dataset 
from train_test import train_process, validate_test_process, compute
from eval import get_correlation

from utils.logger import set_logger, logger
from utils.utils import select_model, set_random_seed, set_device, models

def parse_args():
    args = ArgumentParser()
    args.add_argument("--data-dir", default="data", help="数据集目录")
    args.add_argument("--preprocessed-data-dir", default="data_preprocessed", help="处理后的数据集目录")
    args.add_argument("--exp", default="", help="实验编号.-1 表示不编号")
    args.add_argument("--model", default="sabgnn", choices=models, help="设置实验所用模型")
    args.add_argument("--result-dir", default="results_test", help="")
    args.add_argument("--seed", default=5, help='随机种子')
    args.add_argument("--device", default=7, help="GPU号")
    args.add_argument("--xdays", type=int, default=21, help="预测所需历史天数")
    args.add_argument("--ydays", type=int, default=7, help="预测未来天数")
    # args.add_argument("--startdate", type=str, default="20200414", help="预测开始天数")
    # args.add_argument("--enddate", type=str, default="20210207", help="预测结束天数")
    args.add_argument("--wave", type=int, default=4, choices=[3, 4], help="预测波次")
    args.add_argument("--case-normalize-ratio", type=float, default=100., help="训练集比例百分点")
    args.add_argument("--text-normalize-ratio", type=float, default=100., help="训练集比例百分点")
    args.add_argument("--trainratio", type=int, default=70, help="训练集比例百分点")
    args.add_argument("--validateratio", type=int, default=10, help="验证集比例百分点")
    args.add_argument("--epochs", type=int, default=1000)
    args.add_argument("--batchsize", type=int, default=8)
    args.add_argument("--lr", type=float, default=1e-3)
    args.add_argument("--lr-min", type=float, default=1e-4)
    args.add_argument("--databinfile", type=str, default='dataset', help='处理后的数据集文件名称。其实际文件名为 args.databinfile_wave_xdays_ydays.bin')
    args.add_argument("--enable-graph-learner", default=False, help='是否启用图学习器')
    args.add_argument("--desc", help="该实验的说明")
    args.add_argument("--f", help="兼容 jupyter")
    args = args.parse_args()

    now = date2str(datetime.now(), "%Y%m%d%H%M%S")
    if args.exp == "":
        args.result_dir = os.path.join(
            "results", args.result_dir,
            f"{args.wave}_{args.xdays}_{args.ydays}_{now}"
        )
    else:
        args.result_dir = os.path.join(
            "results", args.result_dir, f"exp_{args.exp}",
            f"{args.wave}_{args.xdays}_{args.ydays}_{now}"
        )
    os.makedirs(args.result_dir, exist_ok=True)
    
    set_logger(os.path.join(args.result_dir, "log.txt"))

    args.device = set_device(args.device)
    args.databinfile = f"{args.databinfile}_{args.wave}_{args.xdays}_{args.ydays}.bin"

    args.trainratio /= 100
    args.validateratio /= 100

    args.startdate = "20200414" if args.wave == 3 else "20200720"
    args.enddate = "20210207" if args.wave == 3 else "20210515"

    return args

def main():
    args = parse_args()

    logger.info(f"运行结果将保存至 {args.result_dir}")

    with open(os.path.join(args.result_dir, 'help.txt'), 'w') as f:
        for k, v in args._get_kwargs():
            f.write("{}: {}\n".format(k, v))


    set_random_seed(args.seed)

    data_origin, date_all = load_data(args)
    
    # 分割数据集
    train_loader, validation_loader, test_loader, train_origin, validation_origin, test_origin, train_indices, validation_indices, test_indices = split_dataset(args, data_origin, date_all)

    model = select_model(args, train_loader)
    criterion = torch.nn.MSELoss()

    logger.info("数据准备完成，开始训练")

    writer = SummaryWriter(args.result_dir)
    losses, trained_model = train_process(
        args, model, criterion, 
        train_loader, validation_loader, test_loader,
        writer
    )
    writer.close()

    logger.info("训练完毕，开始评估")

    validation_result, validation_hat, validation_real = validate_test_process(trained_model, criterion, validation_loader)
    test_result, test_hat, test_real = validate_test_process(trained_model, criterion, test_loader)

    metrices = compute(
        validation_hat, validation_real,
        test_hat, test_real, args.case_normalize_ratio
    )
    
    train_result, train_hat, train_real = validate_test_process(trained_model, criterion, train_loader)
    get_correlation(
        train_hat, train_real, validation_hat, validation_real, test_hat, test_real, args.ydays
        )

    logger.info(f"实验（波次 {args.wave}, 预测范围 {args.xdays}->{args.ydays}）结束")
    logger.info(f"实验结果已保存至 {args.result_dir}")

if __name__ == "__main__":
    main()