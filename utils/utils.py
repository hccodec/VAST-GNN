import os, torch, numpy as np, random
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
    _color = f"\033[{font_style.index(style)}m"
    if isinstance(content, float):
        return f"{_color}{content:.3f}\033[0m"
    else:
        return f"{_color}{content}\033[0m"


def _color(color, content):
    assert color in color_code
    _color = f"\033[1;{color_code.index(color) + 31}m"
    if isinstance(content, float) or isinstance(content, np.float32):
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
    torch.use_deterministic_algorithms(True, warn_only=True)

def adjust_lambda(epoch, num_epochs, lambda_0, lambda_n, lambda_epoch_max, method='cos'):
    from utils.args import graph_lambda_methods
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

# def random_mask(data, observed_ratio = 0.8):
#     assert observed_ratio > 0 and observed_ratio <= 1
    
#     x_case, y_case, x_mob, y_mob, idx_dataset = data
#     (batch_size, num_xdays, num_nodes, num_features), num_ydays = x_case.size(), y_case.size(1)

#     mask = torch.cat([torch.ones(int(num_nodes * observed_ratio)), torch.zeros(num_nodes - int(num_nodes * observed_ratio))])
#     mask = mask[torch.randperm(num_nodes)]
#     # mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, num_xdays, -1)
#     selected_indices = torch.nonzero(mask).squeeze().to(x_case.device)
    
#     if x_case is not None: x_case = x_case[:, :, selected_indices]
#     if y_case is not None: y_case = y_case[:, :, selected_indices]
#     if x_mob is not None: x_mob = x_mob[:, :, selected_indices][:, :, :, selected_indices]
#     if y_mob is not None: y_mob = y_mob[:, :, selected_indices][:, :, :, selected_indices]
#     # if idx is not None: idx = idx[selected_indices]
#     # if extra_info is not None: extra_info = extra_info[:, :, selected_indices][:, :, :, selected_indices]

#     return x_case, y_case, x_mob, y_mob, idx_dataset

def get_exp_desc(modelstr, xdays, ydays, window, shift, node_observed_ratio, language='cn') -> str:
    '''
    Args:
        modelstr: 模型字符串
        xdays: 历史窗口
        ydays: 预测窗口
        window: 历史数据特征窗口
        shift: 预测窗口偏移
        node_observed_ratio: 节点保留比例
        language: 语言 ('cn' 或 'en')

    Returns:
        描述字符串
    '''
    
    # 描述预测天数信息
    y_desc = ""
    if shift == 0:
        y_desc += f" {ydays}" if language == 'cn' else f" {ydays}"
    elif ydays == 1:
        y_desc += f"第 {shift + 1}" if language == 'cn' else f" {shift + 1}"
    else:
        y_desc += f"第 {shift + 1}-{shift + ydays}" if language == 'cn' else f" {shift + 1}-{shift + ydays}"

    # 描述 mask 结点信息
    node_observed_ratio_desc = ""
    if node_observed_ratio < 1:
        node_observed_ratio_desc = f" (图节点保留 {node_observed_ratio * 100:.2f}%)" if language == 'cn' else f" (Graph nodes kept {node_observed_ratio * 100:.2f}%)"
    
    # 生成描述
    if language == 'cn':
        desc = f"历史 {xdays} 天预测未来{y_desc} 天 ({modelstr}_w{window}){node_observed_ratio_desc}"
    else:
        desc = f"Predict {y_desc} days using past {xdays} days ({modelstr}_w{window}){node_observed_ratio_desc}"
    
    return desc

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

def rm_self_loops(a): return a * (1 - torch.eye(a.size(-1)).to(a.device))

def scale_adj(adj_target, adj_source):
    '''
    将 adj_target 缩放到 adj_source 所在尺度
    '''
    # 计算 μ 和 σ
    mean_adj_target = adj_target.mean(axis=(-2, -1), keepdims=True)
    std_adj_target = adj_target.std(axis=(-2, -1), keepdims=True)
    mean_adj_source = adj_source.mean(axis=(-2, -1), keepdims=True)
    std_adj_source = adj_source.std(axis=(-2, -1), keepdims=True)

    # # # 按 μ 和 σ 缩放
    # # adj_target = (adj_target - mean_adj_target) / std_adj_target * std_adj_source_no_diag + mean_adj_source_no_diag
    # # adj_target = rm_self_loops(adj_target)

    # 按 μ 缩放
    adj_target = adj_target * mean_adj_source / mean_adj_target
    
    # # 均做拉普拉斯变换
    # adj_target = getLaplaceMat(adj_target)
    # adj_source_no_diag = getLaplaceMat(adj_source_no_diag)

    #######################
    return adj_target

def getLaplaceMat(adj):
    shape = adj.shape
    adj = adj.flatten(0, -3)
    batch_size, m, _ = adj.size()
    i_mat = torch.eye(m).to(adj.device)
    i_mat = i_mat.unsqueeze(0)
    o_mat = torch.ones(m).to(adj.device)
    o_mat = o_mat.unsqueeze(0)
    i_mat = i_mat.expand(batch_size, m, m)
    o_mat = o_mat.expand(batch_size, m, m)
    adj = torch.where(adj > 0, o_mat, adj)
    '''
    d_mat = torch.bmm(adj, adj.permute(0, 2, 1))
    d_mat = torch.where(i_mat>0, d_mat, i_mat)
    print('d_mat version 1', d_mat)
    '''
    d_mat_in = torch.sum(adj, dim=1)
    d_mat_out = torch.sum(adj, dim=2)
    d_mat = torch.sum(adj, dim=2)  # attention: dim=2
    d_mat = d_mat.unsqueeze(2)
    d_mat = d_mat + 1e-12
    # d_mat = torch.pow(d_mat, -0.5) if is 1/2
    d_mat = torch.pow(d_mat, -1)
    d_mat = d_mat.expand(d_mat.shape[0], d_mat.shape[1], d_mat.shape[1])
    d_mat = i_mat * d_mat

    # laplace_mat = d_mat * adj * d_mat
    laplace_mat = torch.bmm(d_mat, adj)
    # laplace_mat = torch.bmm(laplace_mat, d_mat)
    laplace_mat = laplace_mat.reshape(shape)
    return laplace_mat

def get_country(country, meta_data):
    country_codes, country_names = meta_data["country_codes"], meta_data["country_names"]
    country_code, country_name = "", ""
    if country in country_codes:
        country_code = country
        country_name = country_names[country_codes.index(country_code)]
    elif country in country_names:
        country_name = country
        country_code = country_codes[country_names.index(country_name)]
    return country_code, country_name

def matplotlib_chinese():

    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    import matplotlib as mpl

    font_path = "/home/hbj/HYWenHei-85W.ttf"
    font_prop = fm.FontProperties(fname=font_path)
    font_name = font_prop.get_name()

    mpl.rcParams['font.family'] = font_name
    mpl.rcParams['font.sans-serif'] = [font_name]
    mpl.rcParams['axes.unicode_minus'] = False

