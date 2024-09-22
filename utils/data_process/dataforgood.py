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

def load_data(args, enable_cache = False):
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
    H, y, adjs, batchsize=8
):
    num_days, num_nodes, _ = H.shape
    # indices generation for train/val/test
    train_indices, val_indices, test_indices, get_y_indices = generate_indices(args, num_days, num_nodes)

    # extract formatted data with xdays/yays
    x_days, y_days = args.xdays, args.ydays
    X = torch.zeros_like(H).unsqueeze(1).repeat(1, x_days, 1, 1)
    A = torch.zeros_like(adjs).unsqueeze(1).repeat(1, x_days, 1, 1)
    Y = torch.zeros_like(y).unsqueeze(1).repeat(1, y_days, 1, 1)
    A_Y = torch.zeros_like(adjs).unsqueeze(1).repeat(1, y_days, 1, 1)

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
            A_Y[i_date, i_ydays] = adjs[_i_date]

    train_data = (X[train_indices], Y[get_y_indices(train_indices)], A[train_indices], torch.tensor(train_indices), A_Y[train_indices])
    val_data = (X[val_indices], Y[get_y_indices(val_indices)], A[val_indices], torch.tensor(val_indices), A_Y[val_indices])
    test_data = (X[test_indices], Y[get_y_indices(test_indices)], A[test_indices], torch.tensor(test_indices), A_Y[test_indices])

    train_loader = DataLoader(
        TensorDataset(*train_data), batch_size=batchsize, shuffle=True
    )
    val_loader = DataLoader(TensorDataset(*val_data), batch_size=batchsize)
    test_loader = DataLoader(TensorDataset(*test_data), batch_size=batchsize)

    return (train_loader, val_loader, test_loader), (train_indices, val_indices, test_indices)

def generate_indices(args, num_days, num_nodes, test_sample=15, sep=10, shift=1):
    
    train_indices, val_indices, test_indices, get_y_indices = [], [], [], lambda x : x

    if args.model == 'mpnn_lstm':
        for i in range(min(args.window - 1, test_sample - sep), test_sample):
            if i < test_sample - sep or (i - test_sample + sep) & 1:
                train_indices.append(i)
            else:
                val_indices.append(i)
        test_indices.append(test_sample)

        train_indices, val_indices, test_indices = map(np.array, [train_indices, val_indices, test_indices])
        get_y_indices = lambda indices: np.where(
            (test_sample > 0) and (indices + shift >= test_sample), indices, indices + shift
        )
    else:
        for i in range(min(args.window - 1, test_sample - sep), test_sample):
            if i < test_sample - sep or (i - test_sample + sep) & 1:
                train_indices.append(i)
            else:
                val_indices.append(i)
        test_indices.append(test_sample)

        train_indices, val_indices, test_indices = map(np.array, [train_indices, val_indices, test_indices])
        get_y_indices = lambda indices: np.where(
            (test_sample > 0) and (indices + shift >= test_sample), indices, indices + shift
        )
        

    return train_indices, val_indices,test_indices, get_y_indices
    
def split_dataset_old(args, data_origin, date_all, mode=2):

    # 先生成对应索引，再生成
    n = len(date_all) - args.xdays - args.ydays + 1
    train_ratio, validation_ratio = args.trainratio, args.validateratio

    if hasattr(n, '__len__'): n = len(n)
    assert train_ratio + validation_ratio < 1 and isinstance(n, int)

    n_train, n_validation = int(n * train_ratio), int(n * validation_ratio)
    n_test = n - n_train - n_validation
    
    train_indices, validation_indices, test_indices = [], [], []
    if mode == 0:
        # sequential
        train_indices = range(n_train)
        validation_indices = np.arange(n_validation) + train_indices[-1] + 1
    elif mode == 1:
        # modulo 9
        train_validate = list(range(n_train + n_validation))
        for i in train_validate:
            if i % 9: train_indices.append(i)
            else: validation_indices.append(i)
    elif mode == 2:
        # backwards even index
        train_validate = list(range(n_train + n_validation))
        for i in train_validate:
            # 从后往前，将索引为偶数的当验证集
            if i < n_train - n_validation or i % 2: train_indices.append(i)
            else: validation_indices.append(i)
    test_indices = n - 1 - np.arange(n_test)
    train_indices, validation_indices, test_indices = sorted(train_indices), sorted(validation_indices), sorted(test_indices)

    # 打乱训练集顺序
    random.shuffle(train_indices, random=random.random)

    train_origin = [_data[train_indices] for _data in data_origin]
    validation_origin = [_data[validation_indices] for _data in data_origin]
    test_origin = [_data[test_indices] for _data in data_origin]

    train_data, validation_data, test_data = TensorDataset(*train_origin), TensorDataset(*validation_origin), TensorDataset(*test_origin)
    train_loader, validation_loader, test_loader = DataLoader(train_data, args.batchsize), DataLoader(validation_data, args.batchsize), DataLoader(test_data, args.batchsize)

    return (
        train_loader, validation_loader, test_loader,
        train_origin, validation_origin, test_origin,
        train_indices, validation_indices, test_indices
    )