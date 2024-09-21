"""
dataforgood jieguo 
"""

from utils.custom_datetime import DateRange, date2str, str2date
import geopandas as gpd, os, json, math, copy
import numpy as np
import torch, random
from torch.utils.data import DataLoader, TensorDataset
import pickle
from tqdm.auto import tqdm
from utils.logger import logger
from utils.utils import progress_indicator
import networkx as nx
import pandas as pd

data_dir = "data/dataforgood"
country_codes = ["IT", "ES", "EN", "FR"]
country_names = ["Italy", "Spain", "England", "France"]
country_start_dates = ["0224", "0312", "0313", "0310"]
country_end_date = "0512"
# window_size = 21

meta_info = {"name": country_names, "code": country_codes}

def load_data(args, enable_cache = True):
    databinfile = os.path.join(args.preprocessed_data_dir, args.databinfile)
    if enable_cache and os.path.exists(databinfile):
        with open(databinfile, 'rb') as f:
            meta_data = pickle.load(f)
        logger.info('已从数据文件读取 dataforgood 数据集')
        return meta_data

    meta_data = {}
    for i in range(len(country_names)):
        data = _load_data(i, args.window)
        dataloaders, data_indices = split_dataset(args, *data)
        meta_data[country_names[i]] = (dataloaders, data, data_indices)

    if not os.path.exists(databinfile):
        os.makedirs(args.preprocessed_data_dir, exist_ok=True)
        with open(databinfile, 'wb') as f:
            pickle.dump(meta_data, f)

    return meta_data


def _load_data(i_country: int, window_size: int):
    """
    按国家从文件加载数据集
    """

    logger.info(f"正在读取国家 {country_names[i_country]:{max([len(n) for n in country_names])}s} 的数据...")

    dates = [
        date2str(date, "%Y-%m-%d")
        for date in DateRange(
            f"2020{country_start_dates[i_country]}",
            str2date(f"2020{country_end_date}") + 1,
        )
    ]

    labels = pd.read_csv(
        os.path.join(
            data_dir,
            country_names[i_country],
            f"{country_names[i_country].lower()}_labels.csv",
        )
    ).set_index("name")

    Gs = []
    nodes_pre = []

    for i_dates in range(len(dates)):
        graph = pd.read_csv(
            os.path.join(
                data_dir,
                country_names[i_country],
                "graphs",
                "{}_{}.csv".format(country_codes[i_country], dates[i_dates]),
            ),
            header=None,
        )
        G = nx.DiGraph()
        nodes = sorted(set(list(graph[0]) + list(graph[1])))
        G.add_nodes_from(nodes)
        for row in graph.iterrows():
            G.add_edge(row[1][0], row[1][1], weight=row[1][2])
        Gs.append(G)
        assert not len(nodes_pre) or nodes_pre == nodes  # 所有图的结点都相同
        nodes_pre = nodes

    labels = labels.loc[nodes, dates]  # not needed

    adjs = np.stack([nx.adjacency_matrix(G).toarray() for G in Gs])  # A

    H = np.zeros((len(dates), len(nodes), window_size))  # X
    y = np.zeros((len(dates), len(nodes), 1))  # y

    for i_date in range(len(dates)):
        for i_node in range(len(nodes)):
            H[i_date, i_node, max(0, window_size - i_date) :] = labels.iloc[
                i_node, max(0, i_date - window_size) : i_date
            ]
            y[i_date, i_node] = labels.iloc[i_node, i_date]

    # train_loader, validation_loader, test_loader, train_origin, validation_origin, test_origin, train_indices, validation_indices, test_indices, data_origin, date_all
    H, y, adjs = torch.tensor(H), torch.tensor(y), torch.tensor(adjs)

    return H, y, adjs


# 分割数据集
def split_dataset(args, 
    H, y, adjs, batchsize=8, test_sample=15, shift=1, sep=10
):
    train_indices, val_indices, test_indices = [], [], []
    for i in range(min(args.window - 1, test_sample - sep), test_sample):
        if i < test_sample - sep or (i - test_sample + sep) & 1:
            train_indices.append(i)
        else:
            val_indices.append(i)
    test_indices.append(test_sample)

    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    test_indices = np.array(test_indices)

    x_days, y_days = args.xdays, args.ydays
    X = torch.zeros_like(H).unsqueeze(1).repeat(1, x_days, 1, 1)
    A = torch.zeros_like(adjs).unsqueeze(1).repeat(1, x_days, 1, 1)
    Y = torch.zeros_like(y).unsqueeze(1).repeat(1, y_days, 1, 1)

    for i_date in range(H.shape[0]):
        for i_xdays in range(x_days):
            _i_date = i_date + i_xdays - x_days + 1
            if _i_date < 0: _i_date = 0
            X[i_date, i_xdays] = H[_i_date]
            A[i_date, i_xdays] = adjs[_i_date]
        for i_ydays in range(y_days):
            _i_date = i_date + 1 + i_ydays
            if _i_date >= Y.size(0): _i_date = Y.size(0) - 1
            Y[i_date, i_ydays] = y[_i_date]

    y_indices = lambda indices: np.where(
        (test_sample > 0) & (indices + shift >= test_sample), indices, indices + shift
    )
    # (
    #     (indices + shift)
    #     if (test_sample > 0 and indices + shift >= test_sample)
    #     else indices
    # )

    train_data = (X[train_indices], Y[y_indices(train_indices)], A[train_indices], torch.tensor(train_indices), torch.empty(train_indices.shape))
    val_data = (X[val_indices], Y[y_indices(val_indices)], A[val_indices], torch.tensor(val_indices), torch.empty(val_indices.shape))
    test_data = (X[test_indices], Y[y_indices(test_indices)], A[test_indices], torch.tensor(test_indices), torch.empty(test_indices.shape))

    train_loader = DataLoader(
        TensorDataset(*train_data), batch_size=batchsize, shuffle=True
    )
    val_loader = DataLoader(TensorDataset(*val_data), batch_size=batchsize)
    test_loader = DataLoader(TensorDataset(*test_data), batch_size=batchsize)

    return (train_loader, val_loader, test_loader), (train_indices, val_indices, test_indices)
