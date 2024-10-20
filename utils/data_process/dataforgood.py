"""
dataforgood jieguo 
"""

import os, re
import numpy as np
import torch, random
from torch.utils.data import DataLoader, TensorDataset
import pickle
from utils.logger import logger
from utils.utils import progress_indicator
import networkx as nx
import pandas as pd

# meta_info = {"name": country_names, "code": country_codes}
pattern_graph_file = re.compile("(.*)_(.*).csv")

def load_data(args, enable_cache = True):
    preprocessed_data_dir = args.preprocessed_data_dir
    data_dir, databinfile, batch_size = args.data_dir, args.databinfile, args.batch_size
    xdays, ydays, window_size, shift = args.xdays, args.ydays, args.window, args.shift
    train_ratio, val_ratio = args.train_ratio, args.val_ratio
    node_observed_ratio = args.node_observed_ratio

    if enable_cache and os.path.exists(databinfile):
        with open(databinfile, 'rb') as f:
            meta_data = pickle.load(f)
        logger.info(f'已从数据文件 [{databinfile}] 读取 dataforgood 数据集')
    else:
        #从数据集里读取数据
        meta_data = {}

        country_names = [d for d in os.listdir(data_dir)
                         if os.path.isdir(os.path.join(data_dir, d))]
        country_codes = list(map(lambda x: x if x is None else x.groups()[0],
                                 [pattern_graph_file.search(os.listdir(os.path.join(data_dir, d, "graphs"))[0])
                                  for d in os.listdir(data_dir)
                                  if os.path.isdir(os.path.join(data_dir, d))]))

        for i in range(len(country_names)):
            logger.info(f"正在读取国家 {country_names[i]:{max([len(n) for n in country_names])}s} 的数据...")
            data = _load_data(window_size, data_dir, node_observed_ratio, country_names[i])
            dataloaders = split_dataset(xdays, ydays, shift, train_ratio, val_ratio, batch_size, *data)
            meta_data[country_names[i]] = (dataloaders, data)

        meta_data = {"country_names": country_names, "country_codes": country_codes, "data": meta_data}

        if enable_cache and not os.path.exists(databinfile):
            os.makedirs(preprocessed_data_dir, exist_ok=True)
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
    data_country_path = os.path.join(data_dir, country_name)

    Gs, dates, nodes = [], [], []

    # 读图，并提取图中包含的结点 nodes 和日期 dates
    for i_date, graph_file in enumerate(os.listdir(os.path.join(data_country_path, "graphs"))):
        graph = pd.read_csv(os.path.join(data_country_path, "graphs", graph_file), header=None)

        country_code, date = pattern_graph_file.search(graph_file).groups()
        dates.append(date)

        _nodes = sorted(set(list(graph[0]) + list(graph[1])))

        G = nx.DiGraph()
        G.add_nodes_from(_nodes)
        for row in graph.iterrows():
            G.add_edge(row[1][0], row[1][1], weight=row[1][2])
        Gs.append(G)

        assert not len(nodes) or nodes == _nodes  # 所有图的结点都应相同
        nodes = _nodes

    adjs = np.stack([nx.adjacency_matrix(G).toarray() for G in Gs])

    # 读标签，本实验为病例数
    labels = pd.read_csv(os.path.join(data_country_path, f"{country_name.lower()}_labels.csv")).set_index("name")
    labels = labels.loc[nodes, dates]

    cases = np.expand_dims(labels.to_numpy().T, -1)

    # 根据 window_size 拼接特征，并根据策略对 adjs 和 cases 进行裁剪或补零
    if policy == "clip":
        features = np.stack([cases[i : i + window_size].squeeze(-1).T for i in range(len(cases) - window_size + 1)])
        cases, adjs = cases[-features.shape[0]:], adjs[-features.shape[0]:]
    elif policy == 'pad':
        cases_pad = np.pad(cases, ((6, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)
        features = np.stack([cases_pad[i : i + window_size].squeeze(-1).T for i in range(len(dates))])

    features, cases, adjs = torch.tensor(features), torch.tensor(cases), torch.tensor(adjs)

    # random_mask
    num_nodes = len(nodes)
    mask = torch.cat([torch.ones(int(num_nodes * node_observed_ratio)), torch.zeros(num_nodes - int(num_nodes * node_observed_ratio))])
    mask = mask[torch.randperm(num_nodes)]
    selected_indices = torch.nonzero(mask).squeeze()
    features = features[:, selected_indices]
    cases = cases[:, selected_indices]
    adjs = adjs[:, selected_indices][:, :, selected_indices]

    return features, cases, adjs


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
