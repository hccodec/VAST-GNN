import yaml, os, torch
from argparse import ArgumentParser
from utils.custom_datetime import date2str, datetime
from utils.logger import set_logger, logger

models_list = ["selfmodel", "sabgnn", "sabgnn_case_only", "lstm", "dynst", "mpnn_lstm"]
graph_lambda_methods = ['exp']

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="cfg/config.yaml", help="配置文件路径"
    )
    parser.add_argument("--data-dir", default="data", help="数据集目录")
    parser.add_argument("--dataset", default='dataforgood', help="选择的数据集")
    parser.add_argument(
        "--databinfile",
        type=str,
        default="dataset",
        help="处理后的数据集文件名称。其实际文件名为 parser.databinfile_xdays_ydays.bin",
    )
    parser.add_argument(
        "--preprocessed-data-dir",
        default="data_preprocessed",
        help="处理后的数据集目录",
    )
    parser.add_argument("--exp", default="-1", help="实验编号.-1 表示不编号")
    parser.add_argument(
        "--model", default="dynst", choices=models_list, help="设置实验所用模型"
    )
    parser.add_argument("--result-dir", default="results_test", help="")
    parser.add_argument("--seed", default=5, help="随机种子")
    parser.add_argument("--device", default=7, help="GPU号")
    parser.add_argument("--xdays", type=int, default=7, help="预测所需历史天数")
    parser.add_argument("--ydays", type=int, default=3, help="预测未来天数")
    parser.add_argument("--window", type=int, default=-1, help="作为特征的历史天数窗口大小，值为-1时和xdays相同")
    # parser.add_argument("--wave", type=int, default=4, choices=[3, 4], help="预测波次")
    parser.add_argument(
        "--case-normalize-ratio", type=float, default=100.0, help="训练集比例百分点"
    )
    parser.add_argument(
        "--text-normalize-ratio", type=float, default=100.0, help="训练集比例百分点"
    )
    parser.add_argument("--trainratio", type=int, default=70, help="训练集比例百分点")
    parser.add_argument(
        "--validateratio", type=int, default=10, help="验证集比例百分点"
    )
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batchsize", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-min", type=float, default=1e-3)
    parser.add_argument("--lr-weight-decay", type=float, default=5e-4)
    parser.add_argument("--lr-scheduler-stepsize", type=float, default=10)
    parser.add_argument("--lr-scheduler-gamma", type=float, default=0.7)
    parser.add_argument("--early-stop-patience", type=float, default=100)
    parser.add_argument("--graph-lambda-0", type=float, default=0.8)
    parser.add_argument("--graph-lambda-k", type=float, default=1e-2)
    parser.add_argument("--graph-lambda-method", choices=graph_lambda_methods, default='exp')
    parser.add_argument(
        "--enable-graph-learner", action="store_true", help="是否启用图学习器"
    )
    parser.add_argument(
        "--train-with-extrainfo",
        action="store_true",
        help="是否结合除病例数外的数据进行训练",
    )
    parser.add_argument("--desc", help="该实验的说明")
    parser.add_argument("--f", help="兼容 jupyter")
    args = parser.parse_args()

    args = process_args(args)
    return args

def process_args(args):

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    now = date2str(datetime.now(), "%Y%m%d%H%M%S")

    args.window = args.xdays if args.window == -1 else args.window

    # subdir = f"{args.wave}_{args.xdays}_{args.ydays}_{args.model}"
    subdir = f"{args.xdays}_{args.ydays}_w{args.window}_{args.model}"
    if args.train_with_extrainfo: subdir += "_text"
    if args.enable_graph_learner: subdir += "_graphlearner"
    subdir += f"_{now}"

    args.data_dir += "/" + args.dataset

    if args.exp == "" or args.exp == "-1":
        args.result_dir = os.path.join("results", args.result_dir, "tmp", args.dataset, subdir)
    else:
        args.result_dir = os.path.join(
            "results", args.result_dir, f"exp_{args.exp}", args.dataset, subdir
        )
    os.makedirs(args.result_dir, exist_ok=True)

    set_logger(args.result_dir)

    args.device = set_device(args.device)
    # args.databinfile = f"{args.databinfile}_{args.wave}_{args.xdays}_{args.ydays}.bin"
    args.databinfile = f"{args.databinfile}_{args.dataset}_{args.xdays}_{args.ydays}_w{args.window}.bin"

    args.trainratio /= 100
    args.validateratio /= 100

    # args.startdate = "20200414" if args.wave == 3 else "20200720"
    # args.enddate = "20210207" if args.wave == 3 else "20210515"

    if args.dataset == 'dataforgood':
        args.case_normalize_ratio = 1
        args.text_normalize_ratio = 1

    return args

def set_device(device_id):
    device_id = int(device_id)
    if device_id == -1:
        logger.info("使用 CPU 进行计算")
        return torch.device("cpu")
    elif torch.cuda.is_available():
        if 0 <= device_id < torch.cuda.device_count():
            logger.info(f"使用 GPU {device_id} 进行计算")
            return torch.device(f"cuda:{device_id}")
        else:
            logger.info(f"警告: 无效的 GPU 号 {device_id}, 切换到CPU")
            return torch.device("cpu")
    else:
        logger.info("警告: 未检测到可用的GPU, 使用CPU进行计算")
        return torch.device("cpu")
