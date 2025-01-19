"""
dataforgood """

import os, re
import numpy as np
import torch, random
from torch.utils.data import DataLoader, TensorDataset
import pickle
from utils.logger import logger
from utils.utils import progress_indicator
import networkx as nx
import pandas as pd

import numpy as np
from typing import Tuple

# def generate_sim_data(
#     num_nodes: int,
#     num_dates: int,
#     a: float = 0.074,
#     b: float = 7.130,
#     c: float = 0.01,
#     gamma: float = 0.05,
#     delta_t: float = 1.0,
#     max_cases: int = 1000,
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     模拟传染病动力学数据。
    
#     参数:
#         num_nodes (int): 节点数量。
#         num_dates (int): 时间步数。
#         a (float): 自增长参数。
#         b (float): 传播系数。
#         c (float): 增长抑制系数。
#         gamma (float): 邻接矩阵调整系数。
#         delta_t (float): 时间步长。
#         max_cases (int): 病例数上限。

#     返回:
#         Tuple[np.ndarray, np.ndarray]: 累积病例数和邻接矩阵序列。
#     """
#     def sigmoid(x: np.ndarray) -> np.ndarray:
#         return 1 / (1 + np.exp(-x))
    
#     # 动力学方程：增长量 = \( a \cdot x + b \cdot \text{传播项} - c \cdot x^2 \)，受病例数上限限制
#     def compute_growth(x: np.ndarray, adj: np.ndarray) -> np.ndarray:
#         propagation = np.sum(adj * sigmoid(x.T - x), axis=1, keepdims=True)
#         growth_rate = a * x + b * propagation - c * x**2
#         growth_rate = np.maximum(growth_rate, 0)  # 确保非负
#         growth_limit_factor = 1 - sigmoid((x - max_cases) / 100)
#         return growth_rate * growth_limit_factor

#     # 更新邻接矩阵规则：基于病例数差异调整权重
#     def update_adjacency(adj: np.ndarray, x: np.ndarray) -> np.ndarray:
#         x_diff = x - x.T
#         adj += gamma * np.exp(-x_diff**2 / 2)
#         np.fill_diagonal(adj, 0)
#         return np.clip(adj, 0, 1)

#     # 初始化病例数和邻接矩阵
#     growth = np.zeros((num_dates, num_nodes, 1))
#     cases = np.zeros((num_dates, num_nodes, 1))
#     adjs = np.random.rand(num_dates, num_nodes, num_nodes)
#     np.fill_diagonal(adjs[0], 0)
    
#     # 设置初始值
#     growth[0] = np.random.randint(1, 10, size=(num_nodes, 1))
#     cases[0] = growth[0]

#     # 动力学迭代
#     for i_date in range(1, num_dates):
#         x_t = cases[i_date - 1]
#         adj_t = adjs[i_date - 1]
        
#         # 计算增长量和更新病例数
#         growth_rate = compute_growth(x_t, adj_t)
#         growth[i_date] = delta_t * growth_rate
#         cases[i_date] = cases[i_date - 1] + growth[i_date]
        
#         # 更新邻接矩阵
#         adjs[i_date] = update_adjacency(adj_t, x_t)

#     return cases, adjs

def generate_sim_data(num_nodes, num_dates, a=0.074, b=0.013, c=0.01, gamma=0.05, delta_t=1.0, max_cases=1000):
    delta_t = 1.0  # 时间步长
    # a, b = 0.074, 7.130  # 动力学参数
    # a, b = 0.074, 0.013  # 动力学参数


    # 初始化节点和时间
    nodes = [f"R{i:03}" for i in range(num_nodes)]
    dates = [f"D{i:03}" for i in range(num_dates)]

    # 初始化病例数和邻接矩阵
    cases = np.zeros((num_dates, num_nodes, 1))  # 病例数，形状为 (时间, 节点, 1)
    adjs = np.tile(np.random.rand(num_nodes, num_nodes), (num_dates, 1, 1))  # 随机生成的邻接矩阵

    # 设置初始病例数
    cases[0] = np.random.randint(0, 100, size=(num_nodes, 1))

    # 定义动力学公式 dx/dt
    def compute_dx_dt(x, adj, a, b):
        """
        计算 dx/dt = a * x + b * sum(adj * sigmoid(x_j - x_i))
        """
        # 计算 sigmoid(x_j - x_i)
        x_diff = x.T - x  # 广播减法
        sigmoid = 1 / (1 + np.exp(-x_diff))
        
        # 计算加权求和部分
        weighted_sum = np.sum(adj * sigmoid, axis=1, keepdims=True)
        
        # 返回 dx/dt
        return a * x + b * weighted_sum

    # 动力学迭代
    for i_date in range(1, num_dates):
        # 取前一天的病例数和邻接矩阵
        x_t = cases[i_date - 1]
        adj_t = adjs[i_date - 1]
        
        # 计算 dx/dt
        dx = compute_dx_dt(x_t, adj_t, a, b)
        
        # 使用欧拉法更新病例数
        cases[i_date] = x_t + delta_t * dx

    return cases, adjs, nodes

# meta_info = {"name": country_names, "code": country_codes}
pattern_graph_file = re.compile("(.*)_(.*).csv")

def load_data(dataset_cache_dir, data_dir, dataset, batch_size,
              xdays, ydays, window_size, shift,
              train_ratio, val_ratio, node_observed_ratio,
              enable_cache = True):
    '''
    为了缓存，此阶段将读取全部国家并缓存
    '''
    # 通过 args.dataset 锁定数据集目录
    dataset_dir = f"{data_dir}/{dataset}"
    databinfile = os.path.join(dataset_cache_dir, f"{dataset}_x{xdays}_y{ydays}_w{window_size}_s{shift}" +\
        ("" if int(node_observed_ratio) == 100 else f"_m{int(node_observed_ratio * 100)}") + ".bin")

    if enable_cache and os.path.exists(databinfile):
        with open(databinfile, 'rb') as f:
            meta_data = pickle.load(f)
        logger.info(f'已从数据文件 [{databinfile}] 读取 sim 数据集')
    else:
        #从数据集里读取数据

        country_names = [f"SIM{i}" for i in range(5)]
        country_codes = [f"S{i}" for i in range(5)]
        
        meta_data = {"country_names": country_names, "country_codes": country_codes, "data": {}, "regions": {}, "selected_indices": {}}

        for i in range(len(country_names)):
            logger.info(f"正在读取国家 {country_names[i]:{max([len(n) for n in country_names])}s} 的数据...")
            data, (nodes, selected_indices) = _load_data(window_size, dataset_dir, node_observed_ratio, country_names[i])
            dataloaders = split_dataset(xdays, ydays, shift, train_ratio, val_ratio, batch_size, *data)
            meta_data["data"][country_names[i]] = (dataloaders, data)
            meta_data["regions"][country_names[i]] = nodes
            meta_data["selected_indices"][country_names[i]] = selected_indices

        if enable_cache and not os.path.exists(databinfile):
            os.makedirs(dataset_cache_dir, exist_ok=True)
            with open(databinfile, 'wb') as f:
                pickle.dump(meta_data, f)

    meta_info = [list(k[0].shape[:2]) for k in map(lambda x: x[1], meta_data['data'].values())]
    meta_info = pd.DataFrame(meta_info, columns=['Days', 'Regions'], index=meta_data['country_names'])
    print(meta_info)

    return meta_data


def _load_data(window_size, data_dir, node_observed_ratio, country_name: str, policy: str = "clip"):
    """
    按国家从文件加载数据集
    """
    assert policy in ['clip', 'pad']


    # region 生成 cases 和 adjs 数据
    #############################################################################################
    import numpy as np

    # 参数设置
    num_nodes = random.randint(30, 100)  # 节点数
    num_dates = random.randint(60, 100)  # 时间步数

    cases, adjs, nodes = generate_sim_data(num_nodes, num_dates)

    #############################################################################################
    # endregion

    # 根据 window_size 拼接特征，并根据策略对 adjs 和 cases 进行裁剪或补零
    if policy == "clip":
        features = np.stack([cases[i : i + window_size].squeeze(-1).T for i in range(len(cases) - window_size + 1)])
        cases, adjs = cases[-features.shape[0]:], adjs[-features.shape[0]:]
    elif policy == 'pad':
        cases_pad = np.pad(cases, ((6, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)
        features = np.stack([cases_pad[i : i + window_size].squeeze(-1).T for i in range(len(dates))])

    features, cases, adjs = torch.tensor(features), torch.tensor(cases), torch.tensor(adjs)

    # random_mask
    # num_nodes = len(nodes)
    mask = torch.cat([torch.ones(int(num_nodes * node_observed_ratio)), torch.zeros(num_nodes - int(num_nodes * node_observed_ratio))])
    mask = mask[torch.randperm(num_nodes)]
    selected_indices = torch.nonzero(mask).squeeze().numpy()
    if selected_indices.shape == ():
        selected_indices = selected_indices.reshape(-1,)
    features = features[:, selected_indices]
    cases = cases[:, selected_indices]
    adjs = adjs[:, selected_indices][:, :, selected_indices]

    return (features, cases, adjs), (nodes, selected_indices)


# 分割数据集
def split_dataset(xdays, ydays, shift, train_ratio, val_ratio, batch_size, features, cases, adjs):

    num_days, num_nodes, _ = features.shape

    # indices generation for train/val/test
    train_indices, val_indices, test_indices = generate_indices(num_days - xdays - ydays - shift + 1, train_ratio, val_ratio)

    # extract formatted data with xdays/yays as well as shift
    x_case = torch.stack([features[i : i + xdays] for i in range(num_days - xdays - ydays - shift + 1)]).to(torch.float32)
    x_mob = torch.stack([adjs[i : i + xdays] for i in range(num_days - xdays - ydays - shift + 1)]).to(torch.float32)
    y_case = torch.stack([cases[i : i + ydays] for i in range(xdays + shift, num_days - ydays + 1)]).to(torch.float32)
    y_mob = torch.stack([adjs[i : i + ydays] for i in range(xdays + shift, num_days - ydays + 1)]).to(torch.float32)

    train_data = (x_case[train_indices], y_case[train_indices], x_mob[train_indices], y_mob[train_indices], torch.tensor(train_indices))
    val_data = (x_case[val_indices], y_case[val_indices], x_mob[val_indices], y_mob[val_indices], torch.tensor(val_indices))
    test_data = (x_case[test_indices], y_case[test_indices], x_mob[test_indices], y_mob[test_indices], torch.tensor(test_indices))

    train_loader = DataLoader(TensorDataset(*train_data), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(*val_data), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(*test_data), batch_size=batch_size)

    return train_loader, val_loader, test_loader


def generate_indices(n, train_ratio, val_ratio):

    if hasattr(n, '__len__'): n = len(n)
    assert train_ratio + val_ratio < 1 and isinstance(n, int)

    n_train, n_validation = int(n * train_ratio), int(n * val_ratio)
    n_test = n - n_train - n_validation

    # backwards even index
    train_indices, val_indices, test_indices = [], [], []
    for i in list(range(n_train + n_validation)):
        # 从后往前，将索引为偶数的当验证集
        if i < n_train - n_validation or i % 2:
            train_indices.append(i)
        else:
            val_indices.append(i)
    test_indices = n - 1 - np.arange(n_test)
    train_indices, val_indices, test_indices = sorted(train_indices), sorted(val_indices), sorted(test_indices)

    return train_indices, val_indices, test_indices
