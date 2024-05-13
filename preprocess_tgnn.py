import torch
import networkx as nx
import numpy as np
import scipy.sparse as sp
import pandas as pd
from tqdm.auto import tqdm

from utils.datetime import *

import os


def generate_features(
    graphs: list,
    labels: list,
    dates: list,
    window=7,
    country_name: str = None,
    scaled=False,
):
    features = list()

    # --- one hot encoded the region
    qbar = tqdm(
        graphs,
        desc=f"Generating features"
        + (f" for {country_name}" if country_name is not None else ""),
        leave=False,
    )
    for idx, G in enumerate(qbar):
        #  Features = population, coordinates, d past cases, one hot region
        H = np.zeros([G.number_of_nodes(), window])  # +3+n_departments])#])#])
        if scaled:
            mu = labels.loc[:, dates[:idx]].mean(1)
            std = labels.loc[:, dates[:idx]].std(1) + 1

        _v_start, H_start = max(0, idx - window), max(0, window - idx)

        # region [slide window explanation]
        # H 和 _v 各用到了滑窗，下图是其简单理解 ($=window, &=序号, #=滑窗中的元素, _=滑窗外的元素)
        # H and _v each utilize a sliding window. Below is a simple representation:
        # idx |  0123456  |  01234...         -1 |
        # 0   |  _______  |  ___________________ |
        # 1   |  ______$  |  &__________________ |
        # 2   |  _____#$  |  #&_________________ |
        # 3   |  ____##$  |  ##&________________ |
        # 4   |  ___###$  |  ###&_______________ |
        # 5   |  __####$  |  ####&______________ |
        # 6   |  _#####$  |  #####&_____________ |
        # 7   |  ######$  |  ######&____________ |
        # 8   |  ######$  |  _######&___________ |
        # 9   |  ######$  |  __######&__________ |
        #                 ...                    |
        # -3  |  ######$  |  __________######&__ |
        # -2  |  ######$  |  ___________######&_ |
        # -1  |  ######$  |  ____________######& |
        # endregion

        for i, node in enumerate(G.nodes()):
            # ---- Past cases
            _v = labels.loc[
                node, dates[_v_start:idx]
            ]  # labels in the sliding window of the node
            if scaled:
                _v = (_v - mu[node]) / std[node]
            H[i, H_start:window] = _v  # slide window (end=window)

        features.append(H)

    return features

def analyze_generated_batches(
    graphs: list,
    features: list,
    y: list,
    indicies: list,
    graph_window: int,
    shift: int,
    batch_size: int,
    device,
    test_sample: int,
):
    N = len(indicies)  # 数据集的数据个数
    n_nodes = graphs[0].shape[0]  # IT: 105 个行政区域
    step = n_nodes * graph_window

    for i in range(0, N, batch_size):
        n_nodes_batch = (min(i + batch_size, N) - i) * graph_window * n_nodes

        print(f'i = {i:2d}, n_node_batch = {n_nodes_batch}, step = {step}', end=', ')
        # adj_tmp = list()
        # features_tmp = np.zeros((n_nodes_batch, features[0].shape[1]))
        print("features_tmp.shape =", (n_nodes_batch, features[0].shape[1]), end=', ')
        # y_tmp = np.zeros((min(i + batch_size, N) - i) * n_nodes)
        print("y_tmp.shape =", (min(i + batch_size, N) - i) * n_nodes, end='')

        # fill the input for each batch
        for e1, j in enumerate(range(i, min(i + batch_size, N))):
            idx = indicies[j]
            print(f'\n  idx=indicies[{j:2d}]={idx:2d}', end=', ')

            if test_sample <= 0:
                _i = idx + shift
            else:
                # --- val is by construction less than test sample
                _i = (idx + shift) if idx + shift < test_sample else idx

            # y_tmp[(n_nodes * e1) : (n_nodes * (e1 + 1))] = y[_i]
            print('y_tmp [%3d, %3d)' % ((n_nodes * e1) , (n_nodes * (e1 + 1))), f'<= y[{_i:2d}] |', end='')

            # E.g. feature[10] containes the previous 7 cases of y[10]
            for e2, k in enumerate(range(idx - graph_window + 1, idx + 1)):

                # adj_tmp.append(graphs[k - 1].T)
                print(f'\n    <adj_tmp <= graphs[{k:2d}].T', end=', ')
                # each feature has a size of n_nodes
                # features_tmp[
                #     (e1 * step + e2 * n_nodes) : (e1 * step + (e2 + 1) * n_nodes), :
                # ] = features[k]
                print('features_tmp [%3d, %3d)' % ((e1 * step + e2 * n_nodes), (e1 * step + (e2 + 1) * n_nodes)), f'新增 features[{k}]>', end='')

        print()

def judge_batches(new, old) -> bool:
    res = True
    res0 = [i.to_dense() == j.to_dense() for i, j in zip(new[0], old[0])]
    res = res and all([i.all() for i in res0])
    res1 = [i == j for i, j in zip(new[1], old[1])]
    res = res and all([i.all() for i in res1])
    res2 = [i == j for i, j in zip(new[2], old[2])]
    res = res and all([i.all() for i in res2])
    return res


def generate_batches(
    graphs: list,
    features: list,
    y: list,
    indicies: list,
    graph_window: int,
    shift: int,
    batch_size: int,
    device: torch.device,
    test_sample: int,
):
    """
    Generate batches for graphs for MPNN

    Parameters:
    graphs (list): List of graphs
    features (list): List of features
    y (list): List of targets
    indicies (list): List of indices (trian, val, test)
    graph_window (int): Graph window size
    shift (int): Shift size
    batch_size (int): Batch size
    device (torch.device): Device to use
    test_sample (int): Test sample


    Returns:
    adj_lst (list): List with block adjacency matrices, where its smaller adjacency is a graph in the batch
    features_lst (list): The features are a list with length=number of batches.
                          Each feature matrix has size (window, n_nodes * batch_size),
                          so second column has all values of the nodes in the 1st batch, then 2nd batch etc.
    y_lst (list): List of labels

    """

    dataset_len = len(indicies)
    n_nodes = graphs[0].shape[0]
    step = n_nodes * graph_window

    adj_lst = list()
    features_lst = list()
    y_lst = list()

    for i in range(0, dataset_len, batch_size):
        n_nodes_batch = (min(i + batch_size, dataset_len) - i) * graph_window * n_nodes

        adj_tmp = list()
        features_tmp = np.zeros((n_nodes_batch, features[0].shape[1]))
        y_tmp = np.zeros((min(i + batch_size, dataset_len) - i) * n_nodes)

        # fill the input for each batch
        for e1, j in enumerate(range(i, min(i + batch_size, dataset_len))):
            idx = indicies[j]

            # E.g. feature[10] containes the previous 7 cases of y[10]
            for e2, k in enumerate(range(idx - graph_window + 1, idx + 1)):

                adj_tmp.append(graphs[k].T) # 如果是第一天，就用它本身，防止数据泄露
                # each feature has a size of n_nodes
                features_tmp[
                    (e1 * step + e2 * n_nodes) : (e1 * step + (e2 + 1) * n_nodes), :
                ] = features[k]

            if test_sample <= 0:
                _v = y[idx + shift]
            else:
                # --- val is by construction less than test sample
                _v = y[(idx + shift) if idx + shift < test_sample else idx]

            y_tmp[(n_nodes * e1) : (n_nodes * (e1 + 1))] = _v

        adj_lst.append(adj_list_to_torch_sparse_tensor(adj_tmp).to(device))
        features_lst.append(torch.FloatTensor(features_tmp).to(device))
        y_lst.append(torch.FloatTensor(y_tmp).to(device))

    return adj_lst, features_lst, y_lst


def generate_batches_lstm(
    n_nodes: int,
    y: list,
    idx: list,
    window: int,
    shift: int,
    batch_size: int,
    device: torch.device,
    test_sample: int,
):
    """
    Generate batches for the LSTM, no graphs are needed in this case

    Parameters:
    n_nodes (int): Number of nodes
    y (list): List of targets
    idx (list): List of indices (trian, val, test)
    window (int): Window size
    shift (int): Shift size
    batch_size (int): Batch size
    device (torch.device): Device to use
    test_sample (int): Test sample

    Returns:
    adj_fake (list): A dummy list of empty adjacency matrices for the model's placeholders. Adj is not used in pure LSTM
    features_lst (list): The features are a list with length=number of batches.
                          Each feature matrix has size (window, n_nodes * batch_size),
                          so second column has all values of the nodes in the 1st batch, then 2nd batch etc.
    y_lst (list): List of labels
    """
    N = len(idx)
    features_lst = list()
    y_lst = list()
    adj_fake = list()

    for i in range(0, N, batch_size):
        n_nodes_batch = (min(i + batch_size, N) - i) * n_nodes * 1

        step = n_nodes * 1

        adj_tmp = list()
        features_tmp = np.zeros((window, n_nodes_batch))  #

        y_tmp = np.zeros((min(i + batch_size, N) - i) * n_nodes)

        for e1, j in enumerate(range(i, min(i + batch_size, N))):
            val = idx[j]

            # keep the past information from val-window until val-1
            for e2, k in enumerate(range(val - window, val)):

                if k == 0:
                    features_tmp[e2, (e1 * step) : (e1 * step + n_nodes)] = np.zeros(
                        [n_nodes]
                    )  # features#[k]
                else:
                    features_tmp[e2, (e1 * step) : (e1 * step + n_nodes)] = np.array(
                        y[k]
                    )  # .reshape([n_nodes,1])#

            if test_sample > 0:
                # val is by construction less than test sample
                if val + shift < test_sample:
                    y_tmp[(n_nodes * e1) : (n_nodes * (e1 + 1))] = y[val + shift]
                else:
                    y_tmp[(n_nodes * e1) : (n_nodes * (e1 + 1))] = y[val]

            else:

                y_tmp[(n_nodes * e1) : (n_nodes * (e1 + 1))] = y[val + shift]

        adj_fake.append(0)

        features_lst.append(torch.FloatTensor(features_tmp).to(device))
        y_lst.append(torch.FloatTensor(y_tmp).to(device))

    return adj_fake, features_lst, y_lst


def adj_list_to_torch_sparse_tensor(matrix: list):
    """
    Convert a scipy sparse matrix to a torch sparse tensor.
    matrix (list): List of numpy arrays whose shape is (n_nodes * n_nodes)
    """
    sparse_mx = sp.block_diag(matrix)
    sparse_mx = sparse_mx.tocoo().astype(np.float32)

    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    
    return torch.sparse_coo_tensor(indices, values, shape)


class AverageMeter(object):
    """
    Compute and store the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def read_meta_datasets(window, config):

    # 从文件中取
    import pickle

    data_file = "dataset_tgnn.bin"

    if os.path.exists(data_file):
        print("从 bin 文件中读取数据集")
        with open(data_file, "rb") as f:
            return pickle.load(f)

    """
    Read the datasets and create the features, labels and graphs

    Parameters:
    window (int): Window size
    config (dict): Configuration dictionary
    country_keys (list): List of country keys


    Returns:
    meta_labs (list): List of labels
    meta_graphs (list): List of graphs
    meta_features (list): List of features

    """
    # if config is None:
    #     # 读取配置文件
    #     import json
    #     with open('config.json', 'r') as config_file:
    #         config = json.load(config_file)

    country_codes = list(config["country_idx"].keys())
    country_names = list(config["countries"])

    cwd = os.getcwd()

    os.chdir("data/tgnn_data")

    tqdm.bar_format = "{desc}: {percentage:3.0f}%|{bar}| {n}/{total}"
    bar_desc = lambda i: f"Reading country {country_names[i]}"

    meta_labs, meta_graphs, meta_features, meta_y = [], [], [], []
    # --------------------------------------------------------------
    for country_idx, country in enumerate(country_names):
        print(f"Preparing {country} dataset")
        os.chdir(country)

        sdate = date(
            2020,
            config["country_start_month"][country_idx],
            config["country_start_day"][country_idx],
        )
        edate = date(
            2020,
            config["country_end_month"][country_idx],
            config["country_end_day"][country_idx],
        )
        delta = edate - sdate
        dates = [sdate + i for i in range(delta.days + 1)]
        dates = [str(date) for date in dates]

        Gs = generate_graphs(dates, country_codes[country_idx])

        labels = pd.read_csv(config["country_labels"][country_idx])
        if "id" in labels:
            del labels["id"]
        labels = labels.set_index("name")
        labels = labels.loc[list(Gs[0].nodes()), dates]

        gs_adj = [nx.adjacency_matrix(kgs).toarray().T for kgs in Gs]

        features = generate_features(Gs, labels, dates, window, country)

        y = list()
        for i, G in enumerate(Gs):
            y.append(list())
            for node in G.nodes():
                y[i].append(labels.loc[node, dates[i]])

        meta_labs.append(labels)
        meta_graphs.append(gs_adj)
        meta_features.append(features)
        meta_y.append(y)

        os.chdir("..")
    # --------------------------------------------------------------
    os.chdir(cwd)

    # 存到文件中
    with open(data_file, "wb") as f:
        print(f"将读到的数据存入文件 {data_file} 中")
        pickle.dump((meta_labs, meta_graphs, meta_features, meta_y), f)

    return meta_labs, meta_graphs, meta_features, meta_y

def generate_graphs(dates, country):
    """
    Generate graphs for a country at a specific date

    Parameters:
    dates (list): List of dates
    country (str): Country code

    Returns:
    Gs (list): List of networx graphs
    """
    Gs = []
    qbar = tqdm(dates, desc=f"Generating graphs for {country}", leave=False)
    for date in qbar:
        d = pd.read_csv("graphs/" + country + "_" + date + ".csv", header=None)
        G = nx.DiGraph()
        nodes = set(d[0].unique()).union(set(d[1].unique()))
        G.add_nodes_from(nodes)

        for row in d.iterrows():
            G.add_edge(row[1][0], row[1][1], weight=row[1][2])
        Gs.append(G)

    return Gs


# def generate_graphs_by_day(dates, country):
#     return generate_graphs(dates, country)

# def generate_graphs_britain(dates):
#     return generate_graphs(dates, "EN")
