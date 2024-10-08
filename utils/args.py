import yaml, os, torch
from argparse import ArgumentParser

from easydict import EasyDict

from utils.custom_datetime import date2str, datetime
from utils.logger import set_logger, logger

models_list = ["lstm", "dynst", "mpnn_lstm"]
graph_lambda_methods = ['exp', 'cos']

def get_parser(parent_parser=None):
    if parent_parser is None:
        parser = ArgumentParser()
    else:
        assert isinstance(parent_parser, ArgumentParser)
        parser = parent_parser
    # 配置文件以及各种目录
    parser.add_argument("--config", type=str, default="cfg/config.yaml", help="配置文件路径")
    parser.add_argument("--country", type=str, default=None, help="指定训练某个国家")
    # parser.add_argument("--data-dir", default="data", help="数据集目录")
    # parser.add_argument("--dataset", default='dataforgood', help="选择的数据集")
    # parser.add_argument("--databinfile", type=str, default="",
    #                     help="处理后的数据集文件名称。其实际文件名为 <databinfile>_xdays_ydays_window_shift.bin")
    # parser.add_argument("--preprocessed-data-dir", default="data_preprocessed",help="处理后的数据集目录")
    parser.add_argument("--exp", default="-1", help="实验编号.-1 表示不编号")
    parser.add_argument("--model", default="dynst", choices=models_list, help="设置实验所用模型")
    parser.add_argument("--result-dir", default="results_test", help="")
    # # 实验设置：随机种子和设备号
    # parser.add_argument("--seed", default=5, help="随机种子")
    parser.add_argument("--device", default=None, help="GPU号")
    # 实验设置：历史 xdays 天（以包括自身的前 window 天为当天特征）隔 shift 天预测未来 ydays 天
    parser.add_argument("--xdays", type=int, default=7, help="预测所需历史天数")
    parser.add_argument("--ydays", type=int, default=3, help="预测未来天数")
    parser.add_argument("--window", type=int, default=-1, help="作为特征的历史天数窗口大小，值为-1时和xdays相同")
    parser.add_argument("--shift", type=int, default=0,
                        help="大于 0 则启用隔 shift 天预测。如xdays=7, ydays=2, shift=1 即 0-6 天预测 8-9 天。")
    # # 实验设置：比率
    # parser.add_argument("--nodes-observed-ratio", type=float, default=100.0, help="观测到的结点百分点")
    # parser.add_argument("--case-normalize-ratio", type=float, default=100.0, help="训练集比例百分点")
    # parser.add_argument("--train-ratio", type=int, default=70, help="训练集比例百分点")
    # parser.add_argument("--val-ratio", type=int, default=10, help="验证集比例百分点")
    #
    # parser.add_argument("--epochs", type=int, default=500)
    # parser.add_argument("--batch-size", type=int, default=8)
    # # 实验设置：学习率
    # parser.add_argument("--lr", type=float, default=1e-3)
    # parser.add_argument("--lr-min", type=float, default=1e-3)
    # parser.add_argument("--lr-weight-decay", type=float, default=5e-4)
    # parser.add_argument("--lr-scheduler-stepsize", type=float, default=10)
    # parser.add_argument("--lr-scheduler-gamma", type=float, default=0.7)
    # parser.add_argument("--early-stop-patience", type=float, default=100)
    #
    # # 实验：图结构相关参数设置
    # parser.add_argument("--graph-lambda-0", type=float, default=0.8)
    # parser.add_argument("--graph-lambda-n", type=float, default=0)
    # parser.add_argument("--graph-lambda-epoch-max", type=float, default=-1)
    # parser.add_argument("--graph-lambda-method", choices=graph_lambda_methods, default='cos')
    parser.add_argument("--maml", action="store_true", help="是否启用元学习")
    parser.add_argument("--no_graph_gt", action="store_true", help="图学习器是否不融合历史真实图结构。默认不带该参即融合")

    parser.add_argument("--comp-last",action="store_true",help="在多天预测的情形下只比对最后一天的结果")
    parser.add_argument("--desc", help="该实验的说明")
    parser.add_argument("--f", help="兼容 jupyter")
    return parser

def parse_args(record_log = True, parent_parser = None):
    args = get_parser(parent_parser).parse_args()
    args = process_args(args, record_log)
    return args

def process_args(args, record_log):

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # 遍历 args 中的每个属性
    for key, value in vars(args).items():
        # cfg 优先
        # if key not in cfg: cfg[key] = value
        # args 优先
        if value is not None: cfg[key] = value

    args = EasyDict(cfg)

    now = date2str(datetime.now(), "%Y%m%d%H%M%S")

    # args.comp_last 仅支持多天连续预测
    if args.comp_last:
        assert args.shift == 0, f"参数 comp-last ({args.comp_last}) 仅支持多天连续预测."

    # 通过 args.window 的默认值规范 args.window
    args.window = args.xdays if args.window == -1 else args.window


    # 通过规范 subdir 规范 args.result_dir
    subdir = f"{args.model}_{args.xdays}_{args.ydays}_w{args.window}_s{args.shift}"
    subdir += f"_{now}"

    # 通过 args.dataset 锁定数据集最终位置
    args.data_dir += "/" + args.dataset

    # 通过数据集以及 arg.exp 锁定实验结果的最终保存位置
    args.exp = str(args.exp)
    if args.exp == "" or args.exp == "-1":
        args.result_dir = os.path.join("results", args.result_dir, "tmp", args.dataset, subdir)
    else:
        args.result_dir = os.path.join(
            "results", args.result_dir, f"exp_{args.exp}", args.dataset, subdir
        )

    # 在实验结果最终保存位置创建 log 文件
    if record_log:
        os.makedirs(args.result_dir, exist_ok=True)
        print("结果目录:", args.result_dir)
        set_logger(args.result_dir)

    # 设置 GPU 设备
    args.device = set_device(args.device)

    # 确定处理好的数据集的位置
    if args.databinfile == "":
        args.databinfile = f"{args.dataset}_x{args.xdays}_y{args.ydays}_w{args.window}_s{args.shift}.bin"
    else:
        args.databinfile = f"{args.databinfile}_{args.dataset}_x{args.xdays}_y{args.ydays}_w{args.window}_s{args.shift}.bin"
    args.databinfile = os.path.join(args.preprocessed_data_dir, args.databinfile)

    # 通过百分点确定隐藏结点比例
    assert 0 <= args.node_observed_ratio <= 100
    args.node_observed_ratio /= 100

    # 通过百分点确定训练集划分比例
    assert 0 < args.train_ratio < 100 and 0 < args.val_ratio < 100
    args.train_ratio /= 100
    args.val_ratio /= 100

    if args.dataset == 'dataforgood':
        args.case_normalize_ratio = 1



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
