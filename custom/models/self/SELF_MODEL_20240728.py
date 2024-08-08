"""
该模型在 20240718 模型基础上，更灵活适配外部数据
注意：参数 train-with-text 失去作用
"""

import torch, math
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TopKPooling
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data, DataLoader
from models.GraphLearner import GraphLearner
from models.SAB_GNN import SpecGCN


class ConvLayers(nn.Module):
    def __init__(
        self,
        in_c,
        out_c,
        kernel_size,
        zones,
        pooling_size,
        dilation_factor,
        needPooling=True,
    ):
        super().__init__()
        self.zones = zones
        self.needPooling = needPooling
        self.conv = nn.Conv2d(
            in_c, out_c, kernel_size=(kernel_size, 1), dilation=(dilation_factor, 1)
        )
        self.batchnorm = nn.BatchNorm2d(out_c)
        if self.needPooling:
            self.pooling = nn.AdaptiveAvgPool2d((pooling_size, zones))

    def forward(self, x):
        shape = x.shape
        x = self.conv(x)
        x = self.batchnorm(x)
        if self.needPooling:
            x = self.pooling(x)
        x = x.view(x.shape[0], -1, self.zones)


class GraphEncoder(nn.Module):
    def __init__(self, layers_config, pooling_ratio=[0.8, 0.5], dropout=0.5, device="cpu"):
        super(GraphEncoder, self).__init__()
        self.device = device
        self.layers = nn.ModuleList()
        self.poolings = nn.ModuleList()

        # Building the network according to the config
        for i in range(len(layers_config) - 1):
            self.layers.append(GCNConv(layers_config[i], layers_config[i + 1]))
            self.poolings.append(TopKPooling(layers_config[i + 1], pooling_ratio[i]))
        self.dropout = dropout

    def forward(self, x, A):
        z = []
        for i_batch in range(x.size(0)):
            _x, _A = x[i_batch], A[i_batch]
            edge_index, edge_weight = dense_to_sparse(_A)

            num_nodes = _x.size(0)
            batch = torch.zeros(num_nodes, dtype=torch.long).to(self.device)

            original_features = _x.clone()  # Save the original node features
            original_num_nodes = num_nodes
            # Initialize the mask to keep track of nodes that are retained after pooling
            node_mask = torch.ones(num_nodes, dtype=torch.bool).to(self.device)

            for layer, pool in zip(self.layers, self.poolings):
                _x = layer(_x, edge_index, edge_weight)
                _x = F.relu(_x)
                _x = F.dropout(_x, self.dropout, training=self.training)
                _x, edge_index, edge_weight, batch, perm, _ = pool(
                    _x, edge_index, edge_weight, batch
                )
                node_mask = node_mask[perm]

            # Recover original node features
            expanded_x = torch.zeros(original_num_nodes, _x.size(1)).to(self.device)
            expanded_x[:_x.size(0)] = _x  # Place the features of retained nodes

            unretained_indices = torch.nonzero(~node_mask, as_tuple=False).squeeze()
            if unretained_indices.numel() > 0:  # Ensure there are unretained nodes
                # Fill in missing features with the mean of retained nodes' features
                expanded_x[unretained_indices] = original_features.mean(dim=0)  # Simple mean interpolation

            z.append(expanded_x.unsqueeze(0))
        z = torch.cat(z, dim=0)
        return z


class GraphDecoder(nn.Module):
    def __init__(self, latent_dim):
        super(GraphDecoder, self).__init__()
        self.latent_dim = latent_dim

    def forward(self, z):
        A = torch.sigmoid(torch.matmul(z, z.t()))
        return A


class SelfModel(nn.Module):
    def __init__(self, args, self_model_args):
        super().__init__()

        self.x_days, self.num_zones, self.feature_dim = self_model_args["shape"][0]
        self.y_days = self_model_args["shape"][1][0]
        # self.text_feature_dim = self_model_args["shape"][3][2]

        self.train_with_text = args.train_with_extrainfo

        # if args.enable_graph_learner: self.graph_learner = GraphLearner(self.text_feature_dim)
        if args.enable_graph_learner:
            self.graph_learner = GraphLearner(self.feature_dim)

        self.graph_encoder = GraphEncoder(
            # self.text_feature_dim,
            [
                self.feature_dim,
                self_model_args["gnn"]["hid"],
                self_model_args["gnn"]["out"],
            ],
            device=args.device,
        )

        # if self.train_with_text:
        #     # self.text_graph_encoder = GraphEncoder(
        #     #     self.text_feature_dim,
        #     #     self_model_args["gnn"]["hid"],
        #     #     self_model_args["gnn"]["out"],
        #     # )
        #     self.text_fc = nn.Linear(
        #         self.text_feature_dim, self_model_args["text_fc"]["out"]
        #     )

        # self.graph_decoder = GraphDecoder()
        self.lstm = nn.LSTM(
            input_size=self_model_args["gnn"]["out"] + 1,
            hidden_size=self_model_args["lstm"]["hid"][0],
            batch_first=True,
        )
        # self.lstm2 = nn.LSTM(
        #     input_size=self_model_args["lstm"]["hid"][0],
        #     hidden_size=self_model_args["lstm"]["hid"][1],
        #     batch_first=True,
        # )
        # self.conv1 = GCNConv(self.in_c, self.hid_c)
        # self.conv2 = GCNConv(self.hid_c, self.out_c)
        self.fc = nn.Linear(self_model_args["lstm"]["hid"][0], self.y_days)

        self.social_recovery_lambda = nn.Parameter(
            torch.empty(self.num_zones), requires_grad=True
        )
        nn.init.normal_(self.social_recovery_lambda.data, mean=0.05, std=0.0)
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)  # best
            else:
                stdv = 1.0 / math.sqrt(p.size(0))
                p.data.uniform_(-stdv, stdv)

    def forward(self, X, A, extra_info=None, idx=None):
        """
        前向传播函数，用于处理序列数据和图结构数据，通过LSTM和图学习组件进行预测。

        参数:
        X: 序列数据，形状为(batch_size, seq_len, feature_size)。
        A: 图结构数据，表示每一步的邻接矩阵，形状为(batch_size, seq_len, node_num, node_num)。
        extra_info: 可选的额外信息输入，目前未使用。
        idx: 可选的索引信息，目前未使用。

        返回:
        如果self.graph_learner存在，则返回(out, adj_output)元组，
        out是预测结果，adj_output是学习到的图结构；
        否则只返回预测结果out。
        """
        # 获取批次大小
        batch_size = X.size(0)

        # 将输入数据转换为浮点类型，以适应后续计算
        A = A.float()
        X = X.float()

        # 初始化用于存储LSTM输入和调整后邻接矩阵的列表
        lstm_input = []
        adj_output = []

        # 遍历序列中的每一天，处理图结构和序列数据
        for day in range(X.size(1)):
            # 提取当前天的序列数据和邻接矩阵
            x = X[:, day, :, :]
            adj = A[:, day, :, :]

            # 如果定义了graph_learner，则通过它学习调整邻接矩阵
            if hasattr(self, "graph_learner"):
                adj_hat = self.graph_learner(x)
                z = self.graph_encoder(x, adj_hat)
                adj_output.append(adj_hat.unsqueeze(1))
            # 否则直接使用原始邻接矩阵进行图编码
            else:
                z = self.graph_encoder(x, adj)

            # 将图编码结果与序列数据相结合，作为LSTM的输入
            z = torch.cat((z, x[:, :, -1].unsqueeze(-1)), -1)

            lstm_input.append(z.unsqueeze(1))

        # 将LSTM的输入按时间步合并
        lstm_input = torch.cat(lstm_input, 1)
        # 如有定义graph_learner，将调整后的邻接矩阵合并
        if hasattr(self, "graph_learner"):
            adj_output = torch.cat(adj_output, 1)

        # 重新排列LSTM输入的形状，以适应LSTM的要求
        lstm_input = lstm_input.transpose(1, 2).flatten(0, 1)
        # 运行LSTM，获取输出和隐藏状态
        lstm_out, (hc1, cn1) = self.lstm(lstm_input)

        # 提取最后一个时间步的LSTM输出，作为后续处理的输入
        lstm_out = lstm_out[:, -1, :].unsqueeze(-2)
        lstm_out2 = lstm_out

        # 经过第二层LSTM（目前未使用），提取最后一个时间步的输出
        # lstm_out2, (hc2, cn2) = self.lstm2(lstm_out)

        lstm_out2 = lstm_out2[:, -1, :]

        # 将LSTM的最后输出通过全连接层，得到最终的预测结果
        out = self.fc(lstm_out2)

        # 将预测结果重塑为预期的形状
        out = out.view(batch_size, self.y_days, self.num_zones)

        # 根据是否存在graph_learner，返回不同的结果
        if hasattr(self, "graph_learner"):
            return out, adj_output
        else:
            return out


class LSTMs(nn.Module):
    def __init__(self, in_c, hid_cs, out_c):
        super().__init__(self)
        assert hasattr(hid_cs, "__len__") and len(hid_cs) >= 1
        self.lstms = nn.ModuleList()
        self.lstms.append(
            nn.LSTM(
                input_size=in_c,
                hidden_size=hid_cs[0],
                batch_first=True,
            )
        )
        for i_hid_c in range(len(hid_cs[1:-1])):
            self.lstms.append(
                nn.LSTM(
                    input_size=hid_cs[i_hid_c],
                    hidden_size=hid_cs[i_hid_c + 1],
                    batch_first=True,
                )
            )
        self.lstms.append(
            nn.LSTM(
                input_size=hid_cs[-1],
                hidden_size=out_c,
                batch_first=True,
            )
        )

    def forward(self, x):
        for lstm in self.lstms:
            x, _ = lstm(x)
        return x
