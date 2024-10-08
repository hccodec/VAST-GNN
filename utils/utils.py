import os, torch, numpy as np, random
from models.LSTM_ONLY import LSTM_MODEL
from models.MPNN_LSTM import MPNN_LSTM
from models.dynst import dynst_extra_info, dynst
from utils.args import models_list, graph_lambda_methods
from utils.logger import logger
import traceback, functools
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

    dynst_model_args = {
        "in_dim": args.window,
        "out_dim": 1,
        "hidden": 32,
        # "hidden": 64,
        "num_heads": 4,
        "num_layers": 2,
        "graph_layers": 1,
        "dropout": 0.5,
        "device": args.device,
        "no_graph_gt": args.no_graph_gt
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

    assert args.model in models_list
    index = models_list.index(args.model)

    if index == 0:
        model_args = lstm_model_args
        model = LSTM_MODEL(args, model_args).to(args.device)
    elif index == 1:
        model_args = dynst_model_args
        model = dynst(*model_args.values()).to(args.device)
    elif index == 2:
        model_args = mpnn_lstm_model_args
        model = MPNN_LSTM(*model_args.values()).to(args.device)
    else:
        raise ValueError("请选择模型：" + ", ".join(models_list))

    return model, model_args

def adjust_lambda(epoch, num_epochs, lambda_0, lambda_n, lambda_epoch_max, method='cos'):
    assert method in graph_lambda_methods
    assert epoch < num_epochs

    if epoch > lambda_epoch_max: epoch = lambda_epoch_max

    res = 1

    index = graph_lambda_methods.index(method)
    if index == 0:
        k = np.log(10 * lambda_0 / num_epochs)
        res = np.exp(-k * epoch / num_epochs)
    elif index == 1:
        res = np.cos(np.pi / 2 / lambda_epoch_max * epoch)
    else:
        raise IndexError("请选择正确的 adjency matrix lambda 策略")
    
    return res * (lambda_0 - lambda_n) + lambda_n

# def process_batch(data, observed_ratio):
#     """
#     考虑到 random_mask 的随机性和 adj_lambda 的动态变化，故在此对每个 batch 都执行一次数据转换操作。
#     Args:
#         data:           数据集 batch
#         adj_lambda:     模型运行时 A_gt 的系数
#         model:          模型 class
#         observed_ratio: 观测到节点的比例
#
#     Returns:
#
#     """
#     x_case, y_case, x_mob, y_mob, idx_dataset = data
#     x_case, y_case, x_mob, y_mob, idx_dataset = random_mask((x_case, y_case, x_mob, y_mob, idx_dataset), observed_ratio)
#
#     return x_case, y_case, x_mob, y_mob, idx_dataset

def random_mask(data, observed_ratio = 0.8):
    assert observed_ratio > 0 and observed_ratio <= 1
    
    x_case, y_case, x_mob, y_mob, idx_dataset = data
    (batch_size, num_xdays, num_nodes, num_features), num_ydays = x_case.size(), y_case.size(1)

    mask = torch.cat([torch.ones(int(num_nodes * observed_ratio)), torch.zeros(num_nodes - int(num_nodes * observed_ratio))])
    mask = mask[torch.randperm(num_nodes)]
    # mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, num_xdays, -1)
    selected_indices = torch.nonzero(mask).squeeze().to(x_case.device)
    
    if x_case is not None: x_case = x_case[:, :, selected_indices]
    if y_case is not None: y_case = y_case[:, :, selected_indices]
    if x_mob is not None: x_mob = x_mob[:, :, selected_indices][:, :, :, selected_indices]
    if y_mob is not None: y_mob = y_mob[:, :, selected_indices][:, :, :, selected_indices]
    # if idx is not None: idx = idx[selected_indices]
    # if extra_info is not None: extra_info = extra_info[:, :, selected_indices][:, :, :, selected_indices]

    return x_case, y_case, x_mob, y_mob, idx_dataset

def get_exp_desc(modelstr, xdays, ydays, window, shift) -> str:
    '''

    Args:
        modelstr:   模型字符串
        xdays:      历史窗口
        ydays:      预测窗口
        window:     历史数据特征窗口
        shift:      预测窗口偏移

    Returns:

    '''
    y_desc = ""
    if shift == 0: y_desc += f" {ydays}"
    elif ydays == 1: y_desc += f"第 {shift + 1}"
    else: y_desc += f"第 {shift + 1}-{shift + ydays}"

    return f"历史 {xdays} 天预测未来{y_desc} 天 ({modelstr}_w{window})"

@torch.no_grad()
def min_max_adj(adj: torch.Tensor, epsilon = 1e-8):
    adj = adj.clone()
    if abs(adj.max() - 1) < epsilon and adj.min() < epsilon:
        _ = 1
    adj = adj * (1 - torch.eye(adj.shape[-2])).to(adj.device) # 去掉自环，避免自环影响数量级
    adj_min = adj.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
    adj_max = adj.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
    # 防止除以 0，设置一个极小值 epsilon
    adj = (adj - adj_min) / (adj_max - adj_min + epsilon)

    return adj

@torch.no_grad()
def rm_self_loops(a): return a * (1 - torch.eye(a.size(-1)).to(a.device))
