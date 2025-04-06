import yaml, os, torch
from argparse import ArgumentParser

from easydict import EasyDict

from utils.datetime import date2str, datetime
from utils.logger import set_logger, logger

import pandas as pd

from utils.utils import set_random_seed

models_list = ["lstm", "dynst", "mpnn_lstm"]
graph_lambda_methods = ["exp", "cos"]


def get_parser(parent_parser=None):
    if parent_parser is None:
        parser = ArgumentParser()
    else:
        assert isinstance(parent_parser, ArgumentParser)
        parser = parent_parser
    # 配置文件以及各种目录
    parser.add_argument(
        "--config", type=str, default="cfg/config_base.yaml", help="基础配置文件路径"
    )
    parser.add_argument("--country", type=str, default=None, help="指定训练某个国家")
    # parser.add_argument("--data-dir", default="data", help="数据集目录")
    parser.add_argument("--dataset", default=None, help="选择的数据集")
    # parser.add_argument("--databinfile", type=str, default="",
    #                     help="处理后的数据集文件名称。其实际文件名为 <databinfile>_xdays_ydays_window_shift_nodeobservationratio_seed.bin")
    # parser.add_argument("--dataset-cache", default="data_preprocessed",help="处理后的数据集目录")
    parser.add_argument("--exp", default="-1", help="实验编号.-1 表示不编号")
    parser.add_argument(
        "--model", default="dynst", choices=models_list, help="设置实验所用模型"
    )
    parser.add_argument("--result-dir", default="results_test", help="")
    # # 实验设置：随机种子和设备号
    parser.add_argument("--seed", type=int, default=3407, help="随机种子")
    parser.add_argument("--seed-dataset", type=int, default=5, help="数据集随机种子")
    parser.add_argument("--device", default=None, help="GPU号")
    # 实验设置：历史 xdays 天（以包括自身的前 window 天为当天特征）隔 shift 天预测未来 ydays 天
    parser.add_argument("--xdays", type=int, default=7, help="预测所需历史天数")
    parser.add_argument("--ydays", type=int, default=1, help="预测未来天数")
    parser.add_argument(
        "--window",
        type=int,
        default=-1,
        help="作为特征的历史天数窗口大小，值为-1时和xdays相同",
    )
    parser.add_argument(
        "--shift",
        type=int,
        default=2,
        help="大于 0 则启用隔 shift 天预测。如xdays=7, ydays=2, shift=1 即 0-6 天预测 8-9 天。",
    )
    # # 实验设置：比率
    parser.add_argument(
        "--node-observed-ratio", type=float, default=50, help="观测到的结点百分点"
    )
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
    parser.add_argument(
        "--graph-lambda", type=float, default=None
    )  # 此参数用于覆盖下方设置
    # parser.add_argument("--graph-lambda-0", type=float, default=0.8)
    # parser.add_argument("--graph-lambda-n", type=float, default=0)
    # parser.add_argument("--graph-lambda-epoch-max", type=float, default=-1)
    # parser.add_argument("--graph-lambda-method", choices=graph_lambda_methods, default='cos')
    parser.add_argument("--maml", action="store_true", help="是否启用元学习")
    parser.add_argument(
        "--no-graph", action="store_true", help="是否不启用图学习。默认启用"
    )
    parser.add_argument(
        "--no-virtual-node", action="store_true", help="是否启用虚拟结点，默认启用"
    )

    parser.add_argument(
        "--comp-last",
        action="store_true",
        help="在多天预测的情形下只比对最后一天的结果",
    )
    parser.add_argument("--desc", help="该实验的说明")
    parser.add_argument("--f", help="兼容 jupyter")
    parser.add_argument(
        "--gendata", action="store_true", help="仅生成数据集"
    )  # 仅生成数据集
    return parser


def parse_args(record_log=True, parent_parser=None):
    # logger.info("正在读取参数，请不要修改配置文件")
    args = get_parser(parent_parser).parse_args()
    args = process_args(args, record_log)
    logger.info("参数处理完毕：" + str(args))
    return args


def process_args(args, record_log):

    # 读取 yaml
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 把 args 中的参数加入 cfg
    for key, value in vars(args).items():
        # if key not in cfg: cfg[key] = value # 冲突则跳过
        if value is not None:
            cfg[key] = (
                value  # 冲突且部位不为 None 则覆盖。即可设 args 对应参数为 None 避免覆盖
            )

    args = EasyDict(cfg)

    # # 处理loss正则化项参数 graph_lambda        
    if 'graph_lambda' in args: args["graph_lambda_0"] = args["graph_lambda_n"] = args["graph_lambda"]
    else:
        if args.dataset in cfg["lambda_graph_loss"]:
            args["lambda_graph_loss"] = pd.DataFrame(
                cfg["lambda_graph_loss"][args.dataset][
                    f'arr_{int(cfg["node_observed_ratio"])}'
                ],
                index=cfg["lambda_graph_loss"][args.dataset]["ydays_idx"],
                columns=cfg["lambda_graph_loss"][args.dataset]["country_idx"],
            )
            args['graph_lambda'] = args["lambda_graph_loss"].loc[args.ydays + args.shift, args.country]
        else:
            args["graph_lambda_0"] = 0
    args.seed_dataset = cfg["lambda_graph_loss"][args.dataset]['seed_dataset']

    now = date2str(datetime.now(), "%Y%m%d%H%M%S")

    # args.comp_last 仅支持多天连续预测
    if args.comp_last:
        assert args.shift == 0, f"参数 comp-last ({args.comp_last}) 仅支持多天连续预测."

    # 通过 args.window 的默认值规范 args.window
    args.window = args.xdays if args.window == -1 else args.window

    # subdir 用于后续规范实验结果的最终保存位置
    subdir = f"{args.model}_{args.xdays}_{args.ydays}_w{args.window}_s{args.shift}"
    subdir += f"_{now}"

    # 通过数据集以及 arg.exp 锁定实验结果保存目录
    args.exp = str(args.exp)
    args.result_dir = os.path.join(
        "results",
        args.result_dir,
        "tmp" if args.exp == "" or args.exp == "-1" else f"exp_{args.exp}",
        args.dataset,
        subdir,
    )

    # 实验结果保存目录，初始化 log
    if record_log:
        os.makedirs(args.result_dir, exist_ok=True)
        print("结果目录:", args.result_dir)
        set_logger(
            args.result_dir
            if "," in args.country
            else os.path.join(args.result_dir, args.country)
        )

    # 设置 GPU 设备
    args.device = set_device(args.device)

    # 通过百分点将隐藏结点比例规范化至0-1
    assert 0 < args.node_observed_ratio <= 100 and not args.node_observed_ratio < 1
    args.node_observed_ratio /= 100

    # 通过百分点将训练集划分比例规范化至0-1
    assert 0 < args.train_ratio < 100 and 0 < args.val_ratio < 100
    args.train_ratio /= 100
    args.val_ratio /= 100

    if args.dataset == "dataforgood":
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
            logger.info(f"警告: 无效的 GPU 号 {device_id}, 尝试切换到 0 号 GPU")
            return set_device(0)
    else:
        logger.info("警告: 未检测到可用的GPU, 使用CPU进行计算")
        return torch.device("cpu")
