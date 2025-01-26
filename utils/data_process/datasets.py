"""
dataforgood """

import os, re
import numpy as np
import torch, random
from torch.utils.data import DataLoader, TensorDataset, Dataset
import pickle
from utils.logger import logger
from utils.utils import progress_indicator
import networkx as nx
import pandas as pd


class Datasets:

    def __init__(
        self,
        dataset_cache_dir,
        data_dir,
        dataset,
        batch_size,
        xdays,
        ydays,
        window_size,
        shift,
        train_ratio,
        val_ratio,
        node_observed_ratio,
        enable_cache=True,
    ):

        self.pattern_graph_file = re.compile("(.*)_(.*).csv")
        self.dataset_cache_dir = dataset_cache_dir
        self.data_dir = data_dir
        self.dataset = dataset
        self.batch_size = batch_size
        self.xdays = xdays
        self.ydays = ydays
        self.window_size = window_size
        self.shift = shift
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.node_observed_ratio = node_observed_ratio
        self.enable_cache = enable_cache

    def load_data(self):
        #   , dataset_cache_dir, data_dir, dataset, batch_size,
        # xdays, ydays, window_size, shift,
        # train_ratio, val_ratio, node_observed_ratio,
        # enable_cache = True):
        """
        为了缓存，此阶段将读取全部国家并缓存
        """
        # 通过 args.dataset 锁定数据集目录
        databinfile = os.path.join(
            self.dataset_cache_dir,
            f"{self.dataset}_x{self.xdays}_y{self.ydays}_w{self.window_size}_s{self.shift}"
            + (
                ""
                if int(self.node_observed_ratio) == 100
                else f"_m{int(self.node_observed_ratio * 100)}"
            )
            + ".bin",
        )

        if self.dataset == "dataforgood":
            dataset_dir = f"{self.data_dir}/{self.dataset}"
            country_names = [
                d
                for d in os.listdir(dataset_dir)
                if os.path.isdir(os.path.join(dataset_dir, d))
            ]
            country_codes = list(
                map(
                    lambda x: x if x is None else x.groups()[0],
                    [
                        self.pattern_graph_file.search(
                            [
                                f
                                for f in os.listdir(
                                    os.path.join(dataset_dir, d, "graphs")
                                )
                                if f.endswith(".csv")
                            ][0]
                        )
                        for d in os.listdir(dataset_dir)
                        if os.path.isdir(os.path.join(dataset_dir, d))
                    ],
                )
            )
        elif self.dataset == "japan":
            country_names = ["Japan"]
            country_codes = ["JP"]
        elif self.dataset == "sim":
            country_names = [f"SIM{i}" for i in range(5)]
            country_codes = [f"S{i}" for i in range(5)]
        elif self.dataset == "flunet":
            country_names = ["global"]
            country_codes = ["GL"]
        else:
            raise NotImplementedError("")
        self.country_names = country_names
        self.country_codes = country_codes

        # self.dataset_dir = f"{self.data_dir}/{self.dataset}"
        # country_names = [d for d in os.listdir(dataset_dir)
        #                 if os.path.isdir(os.path.join(dataset_dir, d))]
        # country_codes = list(map(lambda x: x if x is None else x.groups()[0],
        #                         [self.pattern_graph_file.search([f for f in os.listdir(os.path.join(dataset_dir, d, "graphs")) if f.endswith(".csv")][0])
        #                         for d in os.listdir(dataset_dir)
        #                         if os.path.isdir(os.path.join(dataset_dir, d))]))

        if self.enable_cache and os.path.exists(databinfile):
            with open(databinfile, "rb") as f:
                meta_data = pickle.load(f)
            logger.info(f"已从数据文件 [{databinfile}] 读取 {self.dataset} 数据集")
        else:
            # 从数据集里读取数据
            meta_data = {
                "country_names": country_names,
                "country_codes": country_codes,
                "data": {},
                "regions": {},
                "selected_indices": {},
                "dates": {},
            }

            for i in range(len(country_names)):
                logger.info(
                    f"正在读取国家 {country_names[i]:{max([len(n) for n in country_names])}s} 的数据..."
                )
                data, (nodes, dates, selected_indices) = self._load_data(
                    country_names[i]
                )
                dataloaders = self.split_dataset(*data)
                meta_data["data"][country_names[i]] = (dataloaders, data)
                meta_data["regions"][country_names[i]] = nodes
                meta_data["dates"][country_names[i]] = dates
                meta_data["selected_indices"][country_names[i]] = selected_indices

            if self.enable_cache and not os.path.exists(databinfile):
                os.makedirs(self.dataset_cache_dir, exist_ok=True)
                with open(databinfile, "wb") as f:
                    pickle.dump(meta_data, f)

        meta_info = [
            list(k[0].shape[:2])
            for k in map(lambda x: x[1], meta_data["data"].values())
        ]
        meta_info = pd.DataFrame(
            meta_info, columns=["Days", "Regions"], index=meta_data["country_names"]
        )
        print(meta_info)

        return meta_data

    def _load_data(self, country_name: str, policy: str = "clip"):
        """
        按国家从文件加载数据集
        """
        assert policy in ["clip", "pad"]

        cases, adjs, nodes, dates = self.read_data(country_name)

        # data_country_path = os.path.join(data_dir, country_name)

        # Gs, dates, nodes_graph = [], [], []

        # label_file = [i for i in os.listdir(data_country_path) if i.endswith("_labels.csv")]
        # assert len(label_file) == 1
        # label_file = label_file[0]
        # # 读标签，本实验为病例数
        # labels = pd.read_csv(os.path.join(data_country_path, label_file)).set_index("name")

        # nodes_label = list(labels.index)

        # # 读图，并提取图中包含的结点 nodes 和日期 dates
        # for i_date, graph_file in enumerate(os.listdir(os.path.join(data_country_path, "graphs"))):
        #     if not graph_file.endswith(".csv"): continue
        #     graph = pd.read_csv(os.path.join(data_country_path, "graphs", graph_file), header=None)

        #     country_code, date = self.pattern_graph_file.search(graph_file).groups()
        #     dates.append(date)

        #     nodes = sorted(set(list(graph[0]) + list(graph[1])))
        #     assert not len(nodes_graph) or nodes_graph == nodes  # 所有图的结点都应相同
        #     if nodes_graph == []: nodes_graph = nodes

        #     nodes = sorted(set(nodes_graph) & set(nodes_label))

        #     G = nx.DiGraph()
        #     G.add_nodes_from(nodes)
        #     for row in graph.iterrows():
        #         G.add_edge(row[1][0], row[1][1], weight=row[1][2])
        #     Gs.append(G)

        # adjs = np.stack([nx.adjacency_matrix(G).toarray() for G in Gs])

        # labels = labels.loc[nodes, dates]

        # cases = np.expand_dims(labels.to_numpy().T, -1)

        # 根据 window_size 拼接特征，并根据策略对 adjs 和 cases 进行裁剪或补零
        if policy == "clip":
            features = np.stack(
                [
                    cases[i : i + self.window_size].squeeze(-1).T
                    for i in range(len(cases) - self.window_size + 1)
                ]
            )
            cases, adjs = cases[-features.shape[0] :], adjs[-features.shape[0] :]
        elif policy == "pad":
            cases_pad = np.pad(
                cases, ((6, 0), (0, 0), (0, 0)), mode="constant", constant_values=0
            )
            features = np.stack(
                [
                    cases_pad[i : i + self.window_size].squeeze(-1).T
                    for i in range(len(dates))
                ]
            )

        features, cases, adjs = (
            torch.tensor(features),
            torch.tensor(cases),
            torch.tensor(adjs),
        )

        # random_mask
        num_nodes = len(nodes)
        mask = torch.cat(
            [
                torch.ones(int(num_nodes * self.node_observed_ratio)),
                torch.zeros(num_nodes - int(num_nodes * self.node_observed_ratio)),
            ]
        )
        mask = mask[torch.randperm(num_nodes)]
        selected_indices = torch.nonzero(mask).squeeze().numpy()
        if selected_indices.shape == ():
            selected_indices = selected_indices.reshape(
                -1,
            )
        features = features[:, selected_indices]
        cases = cases[:, selected_indices]
        adjs = adjs[:, selected_indices][:, :, selected_indices]

        return (features, cases, adjs), (nodes, dates, selected_indices)

    # 分割数据集
    def split_dataset(self, features, cases, adjs):

        num_days, num_nodes, _ = features.shape

        # indices generation for train/val/test
        train_indices, val_indices, test_indices = self.generate_indices(
            num_days - self.xdays - self.ydays - self.shift + 1
        )

        # extract formatted data with self.xdays/yays as well as self.shift
        x_case = torch.stack(
            [
                features[i : i + self.xdays]
                for i in range(num_days - self.xdays - self.ydays - self.shift + 1)
            ]
        ).to(torch.float32)
        x_mob = torch.stack(
            [
                adjs[i : i + self.xdays]
                for i in range(num_days - self.xdays - self.ydays - self.shift + 1)
            ]
        ).to(torch.float32)
        y_case = torch.stack(
            [
                cases[i : i + self.ydays]
                for i in range(self.xdays + self.shift, num_days - self.ydays + 1)
            ]
        ).to(torch.float32)
        y_mob = torch.stack(
            [
                adjs[i : i + self.ydays]
                for i in range(self.xdays + self.shift, num_days - self.ydays + 1)
            ]
        ).to(torch.float32)

        train_data = (
            x_case[train_indices],
            y_case[train_indices],
            x_mob[train_indices],
            y_mob[train_indices],
            torch.tensor(train_indices),
        )
        val_data = (
            x_case[val_indices],
            y_case[val_indices],
            x_mob[val_indices],
            y_mob[val_indices],
            torch.tensor(val_indices),
        )
        test_data = (
            x_case[test_indices],
            y_case[test_indices],
            x_mob[test_indices],
            y_mob[test_indices],
            torch.tensor(test_indices),
        )

        train_loader = DataLoader(
            TensorDataset(*train_data), batch_size=self.batch_size, shuffle=True
        )
        val_loader = DataLoader(TensorDataset(*val_data), batch_size=self.batch_size)
        test_loader = DataLoader(TensorDataset(*test_data), batch_size=self.batch_size)

        return train_loader, val_loader, test_loader

    def generate_indices(self, n):

        if hasattr(n, "__len__"):
            n = len(n)
        assert self.train_ratio + self.val_ratio < 1 and isinstance(n, int)

        n_train, n_validation = int(n * self.train_ratio), int(n * self.val_ratio)
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
        train_indices, val_indices, test_indices = (
            sorted(train_indices),
            sorted(val_indices),
            sorted(test_indices),
        )

        return train_indices, val_indices, test_indices

    def read_data(self, country_name):

        if self.dataset == "dataforgood":

            # region read_data dataforgood
            data_country_path = os.path.join(self.data_dir, country_name)

            Gs, dates, nodes_graph = [], [], []

            label_file = [
                i for i in os.listdir(data_country_path) if i.endswith("_labels.csv")
            ]
            assert len(label_file) == 1
            label_file = label_file[0]
            # 读标签，本实验为病例数
            labels = pd.read_csv(os.path.join(data_country_path, label_file)).set_index(
                "name"
            )

            nodes_label = list(labels.index)

            # 读图，并提取图中包含的结点 nodes 和日期 dates
            for i_date, graph_file in enumerate(
                os.listdir(os.path.join(data_country_path, "graphs"))
            ):
                if not graph_file.endswith(".csv"):
                    continue
                graph = pd.read_csv(
                    os.path.join(data_country_path, "graphs", graph_file), header=None
                )

                country_code, date = self.pattern_graph_file.search(graph_file).groups()
                dates.append(date)

                nodes = sorted(set(list(graph[0]) + list(graph[1])))
                assert (
                    not len(nodes_graph) or nodes_graph == nodes
                )  # 所有图的结点都应相同
                if nodes_graph == []:
                    nodes_graph = nodes

                nodes = sorted(set(nodes_graph) & set(nodes_label))

                G = nx.DiGraph()
                G.add_nodes_from(nodes)
                for row in graph.iterrows():
                    G.add_edge(row[1][0], row[1][1], weight=row[1][2])
                Gs.append(G)

            adjs = np.stack([nx.adjacency_matrix(G).toarray() for G in Gs])

            labels = labels.loc[nodes, dates]

            cases = np.expand_dims(labels.to_numpy().T, -1)

            # endregion

        elif self.dataset == "japan":

            # region read_data japan_prefectures

            cases = pd.read_csv("data/Others/japan.txt")
            num_dates, num_nodes = cases.shape
            cases = cases.to_numpy().reshape(num_dates, num_nodes, 1)
            # 取前 100 天
            num_dates = 100
            cases = cases[:num_dates]
            adjs = pd.read_csv("data/Others/japan-adj.txt", header=None)
            adjs = np.stack([adjs.to_numpy()] * num_dates, axis=0)

            nodes = [f"R{i}" for i in range(num_nodes)]
            dates = [f"D{i}" for i in range(num_dates)]

            # endregion

        elif self.dataset == "sim":

            # region read_data sim

            import numpy as np

            # 参数设置
            num_nodes = random.randint(30, 100)  # 节点数
            num_dates = random.randint(60, 100)  # 时间步数

            cases, adjs, nodes, dates = generate_sim_data(num_nodes, num_dates)

            # endregion

        elif self.dataset == 'flunet':

            # region read_data flunet

            
            pass

            #endregion

        return cases, adjs, nodes, dates


def generate_sim_data(
    num_nodes,
    num_dates,
    a=0.074,
    b=0.013,
    c=0.01,
    gamma=0.05,
    delta_t=1.0,
    max_cases=1000,
):
    delta_t = 1.0  # 时间步长
    # a, b = 0.074, 7.130  # 动力学参数
    # a, b = 0.074, 0.013  # 动力学参数

    # 初始化节点和时间
    nodes = [f"R{i:03}" for i in range(num_nodes)]
    dates = [f"D{i:03}" for i in range(num_dates)]

    # 初始化病例数和邻接矩阵
    cases = np.zeros((num_dates, num_nodes, 1))  # 病例数，形状为 (时间, 节点, 1)
    adjs = np.tile(
        np.random.rand(num_nodes, num_nodes), (num_dates, 1, 1)
    )  # 随机生成的邻接矩阵

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

    return cases, adjs, nodes, dates
