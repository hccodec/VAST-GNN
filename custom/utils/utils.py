import os, torch, numpy as np, random
from models.LSTM_ONLY import LSTM_MODEL
from models.SAB_GNN import Multiwave_SpecGCN_LSTM
from models.SAB_GNN_case_trained import Multiwave_SpecGCN_LSTM_CASE_TRAINED
from models.SELF_MODEL import SelfModel
from utils.logger import logger
import traceback, functools, yaml
from argparse import ArgumentParser
from utils.custom_datetime import date2str, datetime
from utils.logger import set_logger

font_style = ["", "b", "h", "i", "u", "", "", "r", "t", "d"]
color_code = ["red", "green", "yellow", "blue", "purple", "cyan"]


def _style(style, content):
    assert style in font_style
    _color = f"\033[1;{font_style.index(style) + 31}m"
    if isinstance(content, float):
        return f"{_color}{content:.3f}\033[0m"
    else:
        return f"{_color}{content}\033[0m"


def _color(color, content):
    assert color in color_code
    _color = f"\033[1;{color_code.index(color) + 31}m"
    if isinstance(content, float):
        return f"{_color}{content:.3f}\033[0m"
    else:
        return f"{_color}{content}\033[0m"


def font_bold(content):
    return _style("b", content)


def font_hide(content):
    return _style("h", content)


def font_italics(content):
    return _style("i", content)


def font_underlined(content):
    return _style("u", content)


def font_reversed(content):
    return _style("r", content)


def font_transparent(content):
    return _style("t", content)


def font_deleted(content):
    return _style("d", content)


def font_red(content):
    return _color("red", content)


def font_green(content):
    return _color("green", content)


def font_yellow(content):
    return _color("yellow", content)


def font_blue(content):
    return _color("blue", content)


def font_purple(content):
    return _color("purple", content)


def font_cyan(content):
    return _color("cyan", content)


def set_random_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # 让显卡产生的随机数一致
    torch.cuda.manual_seed_all(
        seed
    )  # 多卡模式下，让所有显卡生成的随机数一致？这个待验证
    np.random.seed(seed)  # numpy产生的随机数一致
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


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


def catch(msg="出现错误，中断训练"):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            if os.getenv("DEBUG_MODE") == "true":
                return f(*args, **kwargs)
            try:
                return f(*args, **kwargs)
            except Exception as e:
                logger.info(msg)
                logger.info(str(e))
                _traceback = traceback.format_exc()
                for line in _traceback.split("\n"):
                    logger.info(str(line))
                return None

        return wrapper

    return decorator


def select_model(args, train_loader):

    shape = [tuple(i.shape) for i in train_loader.dataset[0]]

    specGCN_model_args = {
        "alpha": 0.2,
        "specGCN": {"hid": 6, "out": 4, "dropout": 0.5},
        "shape": shape,
        "lstm": {"hid": 3},
    }
    self_model_args = {
        "dropout": 0.5,
        "gnn": {"hid": 6, "out": 4},
        "lstm": {"hid": [3, 16]},
        "shape": shape,
    }
    lstm_model_args = {
        # 'in': train_loader.dataset[0][2].shape[-1],
        "lstm": {"hid": 128},
        "linear": {"hid": 64},
        "dropout": 0.5,
        # 'out': 32,
        "shape": shape,
    }

    assert args.model in models
    index = models.index(args.model)

    if index == 0:
        model_args = self_model_args
        model = SelfModel(args, model_args).to(args.device)
    elif index == 1:
        model_args = specGCN_model_args
        model = Multiwave_SpecGCN_LSTM(args, model_args).to(args.device)
    elif index == 2:
        specGCN_model_args["shape"] = train_loader.dataset[0][2].shape
        model_args = specGCN_model_args
        model = Multiwave_SpecGCN_LSTM_CASE_TRAINED(args, model_args).to(args.device)
    elif index == 3:
        model_args = lstm_model_args
        model = LSTM_MODEL(args, model_args).to(args.device)
    else:
        raise IndexError("请选择模型：" + ", ".join(models))

    return model, model_args


models = ["selfmodel", "sabgnn", "sabgnn_case_only", "lstm"]


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="cfg/config.yaml", help="配置文件路径"
    )
    parser.add_argument("--data-dir", default="data", help="数据集目录")
    parser.add_argument(
        "--databinfile",
        type=str,
        default="dataset",
        help="处理后的数据集文件名称。其实际文件名为 parser.databinfile_wave_xdays_ydays.bin",
    )
    parser.add_argument(
        "--preprocessed-data-dir",
        default="data_preprocessed",
        help="处理后的数据集目录",
    )
    parser.add_argument("--exp", default="-1", help="实验编号.-1 表示不编号")
    parser.add_argument(
        "--model", default="sabgnn", choices=models, help="设置实验所用模型"
    )
    parser.add_argument("--result-dir", default="results_test", help="")
    parser.add_argument("--seed", default=5, help="随机种子")
    parser.add_argument("--device", default=7, help="GPU号")
    parser.add_argument("--xdays", type=int, default=21, help="预测所需历史天数")
    parser.add_argument("--ydays", type=int, default=7, help="预测未来天数")
    parser.add_argument("--wave", type=int, default=4, choices=[3, 4], help="预测波次")
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
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batchsize", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-min", type=float, default=1e-4)
    parser.add_argument("--lr-scheduler-stepsize", type=float, default=50)
    parser.add_argument("--lr-scheduler-gamma", type=float, default=0.8)
    parser.add_argument("--early-stop-patience", type=float, default=100)
    parser.add_argument(
        "--enable-graph-learner", action="store_true", help="是否启用图学习器"
    )
    parser.add_argument(
        "--train-with-text",
        action="store_true",
        help="是否结合除病例数外的数据进行训练",
    )
    parser.add_argument("--desc", help="该实验的说明")
    parser.add_argument("--f", help="兼容 jupyter")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    now = date2str(datetime.now(), "%Y%m%d%H%M%S")
    
    if args.train_with_text:
        subdir = f"{args.wave}_{args.xdays}_{args.ydays}_{args.model}_text_{now}"
    else:
        subdir = f"{args.wave}_{args.xdays}_{args.ydays}_{args.model}_{now}"
        
    if args.exp == "" or args.exp == "-1":
        args.result_dir = os.path.join("results", args.result_dir, subdir)
    else:
        args.result_dir = os.path.join(
            "results", args.result_dir, f"exp_{args.exp}", subdir
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
