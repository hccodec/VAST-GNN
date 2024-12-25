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
        logger.info(f'已从数据文件 [{databinfile}] 读取 dataforgood 数据集')
    else:
        #从数据集里读取数据

        country_names = ["Japan"]
        country_codes = ["JP"]
        
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
    
    cases = pd.read_csv("data/Others/japan.txt")
    num_dates, num_nodes = cases.shape
    cases = cases.to_numpy().reshape(num_dates, num_nodes, 1)
    # 取前 56 天
    num_dates = 56
    cases = cases[:num_dates]
    adjs = pd.read_csv("data/Others/japan-adj.txt", header=None)
    adjs = np.stack([adjs.to_numpy()] * num_dates, axis=0)

    nodes = [f"R{i}" for i in range(num_nodes)]
    dates = [f"D{i}" for i in range(num_dates)]

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
