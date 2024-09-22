import os, torch, numpy as np, random
from models.LSTM_ONLY import LSTM_MODEL
from models.MPNN_LSTM import MPNN_LSTM
from models.SAB_GNN import Multiwave_SpecGCN_LSTM
from models.SAB_GNN_case_trained import Multiwave_SpecGCN_LSTM_CASE_TRAINED
# from models.SELF_MODEL import SelfModel
# from models.self.SELF_MODEL_20240715 import SelfModel
# from models.self.SELF_MODEL_20240718 import SelfModel
# from models.self.SELF_MODEL_20240728 import SelfModel
from models.self.SELF_MODEL_20240828 import SelfModel
from models.dynst.model import dynst_extra_info, dynst
from utils.logger import logger
import traceback, functools, yaml
from argparse import ArgumentParser
from utils.custom_datetime import date2str, datetime
from utils.logger import set_logger
from tqdm.auto import tqdm

# 进度条
# l_bar='{desc}...({n_fmt}/{total_fmt} {percentage:3.2f}%)'
# r_bar= '{n_fmt}/{total_fmt}'
# r_bar= '{n_fmt}/{total_fmt} [{rate_fmt}{postfix}]'
# bar_format = f'{l_bar}|{{bar}}|{r_bar}{"{postfix}"} '
def bar_format(show_total):
    if show_total: return '{desc}...|{bar}|({n_fmt}/{total_fmt} {percentage:3.0f}%){postfix}'
    else: return '{desc}...{percentage:3.2f}% {postfix}'

def progress_indicator(*args, show_total=True, **kwargs):
    return tqdm(*args, **kwargs, bar_format=bar_format(show_total))


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
            if os.getenv("DEBUG_MODE") == "true" or os.getenv("DEBUG_MODE") == "1":
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
        "text_fc": {"out": 4},
        "gnn": {"hid": 16, "out": 6},
        "lstm": {"hid": [6, 3]},
        "shape": shape,
    }
    dynst_model_args = {
        "in_dim": shape[0][-1],
        "out_dim": 1,
        "hidden": 64,
        "num_heads": 4,
        "num_layers": 2,
        "graph_layers": 1,
        "dropout": 0.5,
        "device": args.device
    }
    lstm_model_args = {
        # 'in': train_loader.dataset[0][2].shape[-1],
        "lstm": {"hid": 128},
        "linear": {"hid": 64},
        "dropout": 0.5,
        # 'out': 32,
        "shape": shape,
    }
    mpnn_lstm_model_args = dict(nfeat=shape[0][-1], nhid=64, nout=1, # n_nodes=shape[0][1],
                              window=shape[0][0], dropout=0.5)

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
    elif index == 4:
        model_args = dynst_model_args
        model = dynst(*model_args.values(), args.enable_graph_learner).to(args.device)
    elif index == 5:
        model_args = mpnn_lstm_model_args
        model = MPNN_LSTM(*model_args.values()).to(args.device)
    else:
        raise ValueError("请选择模型：" + ", ".join(models))

    return model, model_args


models = ["selfmodel", "sabgnn", "sabgnn_case_only", "lstm", "dynst", "mpnn_lstm"]

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
        "--model", default="dynst", choices=models, help="设置实验所用模型"
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

graph_lambda_methods = ['exp']
def adjust_lambda(epoch, lambda_0, k, method='exp'):
    index = graph_lambda_methods.index(method)
    if index == 0:
        return lambda_0 * np.exp(-k * epoch)
    else:
        raise IndexError("请选择正确的 adjency matrix lambda 策略")

def hits_at_k(A_hat_batch, A_batch, k=10, threshold_ratio=0.1):
    """
    计算带有按比例阈值的 HITS@k
    A_hat_batch: 预测的邻接矩阵，大小为 (batch_size, day, n, n)，包含小数值
    A_batch: 实际的邻接矩阵，大小为 (batch_size, day, n, n)
    k: 考虑前k个最高置信度的边
    threshold_ratio: 用于确定局部阈值的比例 (例如 0.1 表示取前 10% 的值作为阈值)
    
    返回:
        HITS@k 的平均值
    """
    A_hat_batch, A_batch = A_hat_batch.detach().cpu().numpy(), A_batch.detach().cpu().numpy()

    batch_size, day, n, _ = A_hat_batch.shape
    total_hits = 0
    total_edges = 0
    
    for b in range(batch_size):
        for d in range(day):
            A_hat = A_hat_batch[b, d]  # 第 b 个batch的第d天的预测邻接矩阵
            A = A_batch[b, d]          # 第 b 个batch的第d天的实际邻接矩阵
            
            hits = 0
            for i in range(n):
                # 预测矩阵中第i行的边置信度（从i节点出发的所有边）
                preds = A_hat[i]
                
                # 使用局部比例法，设定该节点的阈值
                local_threshold = np.percentile(preds, 100 * (1 - threshold_ratio))
                
                # 大于阈值的边被视为存在
                valid_indices = np.where(preds >= local_threshold)[0]
                
                # 如果 k 大于有效边数量，则调整 k 的值
                top_k_count = min(k, len(valid_indices))
                
                # 找出置信度最高的 top_k_count 个边
                if top_k_count > 0:
                    top_k_indices = preds[valid_indices].argsort()[-top_k_count:][::-1]
                    top_k_indices = valid_indices[top_k_indices]
                else:
                    top_k_indices = []
                
                # 实际矩阵中，第i行的边，看看哪些是真实存在的
                true_edges = np.where(A[i] > 0)[0]
                
                # 计算命中的数量
                hits += len(set(top_k_indices).intersection(set(true_edges)))
            
            # 记录当前batch和天数的命中
            total_hits += hits
            total_edges += (n * k)
    
    # 计算所有批次和天数的平均 HITS@k
    return total_hits / total_edges

def process_batch_data(data, adj_lambda, model, device, observed_ratio):
    casex, casey, mobility, idx, extra_info = data
    casex, casey, mobility, idx, extra_info = random_mask((casex, casey, mobility, idx, extra_info), observed_ratio)
    
    # 处理 extra_info
    if isinstance(model, dynst):
        extra_info = dynst_extra_info(adj_lambda, dataset_extra=extra_info)

    return casex, casey, mobility, idx, extra_info

def random_mask(data, observed_ratio = 0.8):
    assert observed_ratio > 0 and observed_ratio <= 1
    
    casex, casey, mobility, idx, extra_info = data
    (batch_size, num_xdays, num_nodes, num_features), num_ydays = casex.size(), casey.size(1)

    mask = torch.cat([torch.ones(int(num_nodes * observed_ratio)), torch.zeros(num_nodes - int(num_nodes * observed_ratio))])
    mask = mask[torch.randperm(num_nodes)]
    # mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, num_xdays, -1)
    selected_indices = torch.nonzero(mask).squeeze().to(casex.device)
    
    if casex is not None: casex = casex[:, :, selected_indices]
    if casey is not None: casey = casey[:, :, selected_indices]
    if mobility is not None: mobility = mobility[:, :, selected_indices][:, :, :, selected_indices]
    # if idx is not None: idx = idx[selected_indices]
    if extra_info is not None: extra_info = extra_info[:, :, selected_indices][:, :, :, selected_indices]

    return casex, casey, mobility, idx, extra_info
