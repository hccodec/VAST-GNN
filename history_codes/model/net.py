import torch, math
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv
import torch.nn.functional as F
from torch_geometric.data import Data
from torch.utils.data import DataLoader

from torch_geometric.nn import GCNConv

class GraphLearner(nn.Module):
    def __init__(self, hidden_dim, tanhalpha=1):
        super().__init__()
        self.hid = hidden_dim
        self.linear1 = nn.Linear(self.hid, self.hid)
        self.linear2 = nn.Linear(self.hid, self.hid)
        self.alpha = tanhalpha

    def forward(self, embedding):
        # embedding [batchsize, hidden_dim]
        nodevec1 = self.linear1(embedding)
        nodevec2 = self.linear2(embedding)
        nodevec1 = self.alpha * nodevec1
        nodevec2 = self.alpha * nodevec2
        nodevec1 = torch.tanh(nodevec1)
        nodevec2 = torch.tanh(nodevec2)
        
        adj = torch.bmm(nodevec1, nodevec2.permute(0, 2, 1))-torch.bmm(nodevec2, nodevec1.permute(0, 2, 1))
        adj = self.alpha * adj
        adj = torch.relu(torch.tanh(adj))
        return adj


class GNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GNNLayer, self).__init__()
        self.gcn = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.gcn(x, edge_index)
        return F.relu(x)


class TemporalGraphLearner(nn.Module):
    def __init__(
        self, num_nodes, in_channels, hidden_channels, out_channels, num_layers, seq_len
    ):
        super(TemporalGraphLearner, self).__init__()
        self.num_nodes = num_nodes
        self.seq_len = seq_len

        self.gnn_layers = nn.ModuleList(
            [
                (GNNLayer(in_channels if i == 0 else hidden_channels, hidden_channels))
                for i in range(num_layers)
            ]
        )
        self.lstm = nn.LSTM(hidden_channels, hidden_channels, batch_first=True)
        self.fc = nn.Linear(hidden_channels, out_channels)

    def forward(self, edge_index, x_seq):
        batch_size = x_seq.size(0)

        # Encode each time step with GNN
        gnn_out_seq = []
        for t in range(self.seq_len):
            x_t = x_seq[:, t, :]  # Shape: [batch_size, num_nodes, in_channels]
            for gnn in self.gnn_layers:
                x_t = gnn(x_t, edge_index)
            gnn_out_seq.append(
                x_t.unsqueeze(1)
            )  # Shape: [batch_size, 1, num_nodes, hidden_channels]

        gnn_out_seq = torch.cat(
            gnn_out_seq, dim=1
        )  # Shape: [batch_size, seq_len, num_nodes, hidden_channels]
        gnn_out_seq = gnn_out_seq.view(
            batch_size * self.num_nodes, self.seq_len, -1
        )  # Shape: [batch_size * num_nodes, seq_len, hidden_channels]

        # Encode with LSTM
        lstm_out, _ = self.lstm(
            gnn_out_seq
        )  # Shape: [batch_size * num_nodes, seq_len, hidden_channels]
        lstm_out = lstm_out[
            :, -1, :
        ]  # Take the last output (many-to-one) Shape: [batch_size * num_nodes, hidden_channels]

        # Predict future features
        out = self.fc(lstm_out)  # Shape: [batch_size * num_nodes, out_channels]
        out = out.view(
            batch_size, self.num_nodes, -1
        )  # Shape: [batch_size, num_nodes, out_channels]

        return out


# endregion


# Pandemic TGNN 对应模型
class MPNN_LSTM(nn.Module):
    def __init__(
        self,
        nfeat: int,
        nhid: int,
        nout: int,
        n_nodes: int,
        window: int,
        dropout: float,
    ):
        """
        Parameters:
        nfeat (int): Number of features
        nhid (int): Hidden size
        nout (int): Number of output features
        n_nodes (int): Number of nodes
        window (int): Window size
        dropout (float): Dropout rate

        Returns:
        x (torch.Tensor): Output of the model
        """
        super(MPNN_LSTM, self).__init__()
        self.window = window
        self.n_nodes = n_nodes
        self.nhid = nhid
        self.nfeat = nfeat

        # 定义GCNConv层
        self.conv1 = GCNConv(nfeat, nhid)
        self.conv2 = GCNConv(nhid, nhid)

        # 定义Batch Normalization层
        self.bn1 = nn.BatchNorm1d(nhid)
        self.bn2 = nn.BatchNorm1d(nhid)

        # 定义LSTM层
        self.rnn1 = nn.LSTM(2 * nhid, nhid, 1)
        self.rnn2 = nn.LSTM(nhid, nhid, 1)

        # 定义全连接层
        self.fc1 = nn.Linear(2 * nhid + window * nfeat, nhid)
        self.fc2 = nn.Linear(nhid, nout)

        # 定义Dropout层和ReLU激活函数
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, adj, x):
        lst = list()

        # 获取稀疏邻接矩阵的权重和索引
        weight = adj.coalesce().values()
        adj = adj.coalesce().indices()

        # 处理时间窗口数据
        skip = x.view(-1, self.window, self.n_nodes, self.nfeat)
        skip = torch.transpose(skip, 1, 2).reshape(-1, self.window, self.nfeat)

        # 第一层GCNConv
        x = self.relu(self.conv1(x, adj, edge_weight=weight))
        x = self.bn1(x)
        x = self.dropout(x)
        lst.append(x)

        # 第二层GCNConv

        x = self.relu(self.conv2(x, adj, edge_weight=weight))
        x = self.bn2(x)
        x = self.dropout(x)
        lst.append(x)

        # 将所有层的输出进行拼接
        x = torch.cat(lst, dim=1)

        # reshape to (seq_len, batch_size , hidden) to fit the lstm
        # 重新整形以适应LSTM输入形状
        x = x.view(-1, self.window, self.n_nodes, x.size(-1))
        x = torch.transpose(x, 0, 1)
        x = x.contiguous().view(self.window, -1, x.size(-1))

        # 第一个LSTM层
        x, (hn1, cn1) = self.rnn1(x)
        # 第二个LSTM层
        out2, (hn2, cn2) = self.rnn2(x)

        hn1 = hn1.squeeze()
        hn2 = hn2.squeeze()

        # use the hidden states of both rnns
        # 使用两个LSTM层的隐藏状态
        x = torch.cat([hn1, hn2], dim=1)
        skip = skip.reshape(skip.size(0), -1)

        x = torch.cat([x, skip], dim=1)

        # 第一个全连接层
        x = self.relu(self.fc1(x))
        x = self.dropout(x)

        # 第二个全连接层
        x = self.relu(self.fc2(x)).squeeze()
        x = x.view(-1)

        return x


#######################################################################


class SpecGCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.5, concat=True):
        super(SpecGCNLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = output_dim
        self.dropout = dropout
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(input_dim, output_dim)))
        self.W.requires_grad = True
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

    def forward(self, feature, D_n_A_D_n):
        # feature.shape: (N,input_dim)
        feature_new = torch.mm(feature.float(), self.W)
        feature_new = F.dropout(feature_new, self.dropout, training=self.training)
        H = torch.mm(D_n_A_D_n.float(), feature_new)
        return H


class SpecGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, dropout=0.5):
        super(SpecGCN, self).__init__()
        self.dropout = dropout
        # self.layer1 = SpecGCNLayer(input_dim, hidden_dim, dropout=dropout, concat=True)
        # self.out_feature = SpecGCNLayer(hidden_dim, out_dim, dropout=dropout, concat=True)
        self.layer1 = SpecGCNLayer(input_dim, out_dim, dropout=dropout, concat=True)
        # self.layer2 = SpecGCNLayer(out_dim, out_dim, dropout=dropout, concat=True)

    def compute_D_n_A_D_n(self, adjs):
        N = adjs.size()[0]
        tilde_A = adjs + torch.eye(N).cuda()
        tilde_D_n = torch.diag(torch.pow(tilde_A.sum(-1).float(), -0.5))
        D_n_A_D_n = torch.mm(tilde_D_n, torch.mm(tilde_A, tilde_D_n))
        return D_n_A_D_n

    def forward(self, x, adjs):
        D_n_A_D_n = self.compute_D_n_A_D_n(adjs)
        # x1 = self.layer1(x, D_n_A_D_n)
        # x1 = F.dropout(x1, self.dropout, training=self.training)
        # x2 = self.out_feature(x1, D_n_A_D_n)
        # x_out = F.relu(x2)
        x1 = self.layer1(x, D_n_A_D_n)
        x1 = F.dropout(x1, self.dropout, training=self.training)
        # x1 = self.layer2(x1, D_n_A_D_n)
        # x1 = F.dropout(x1, self.dropout, training=self.training)
        x_out = F.relu(x1)
        return x_out


class MULTIWAVE_SpecGCN_LSTM(nn.Module):
    def __init__(
        self,
        x_days,
        y_days,
        input_dim_1, # 特征维度
        hidden_dim_1, # specGCN 隐藏层
        out_dim_1, # specGNN 输出层
        hidden_dim_2, # LSTM 输出层
        dropout_1,
        N,
        device="cpu",
    ):
        super(MULTIWAVE_SpecGCN_LSTM, self).__init__()
        self.device = device
        self.N = N
        self.x_days = x_days
        self.y_days = y_days
        self.specGCN = SpecGCN(input_dim_1, hidden_dim_1, out_dim_1, dropout_1).to(
            device
        )
        self.graph_learner = GraphLearner(input_dim_1).to(device)
        self.graph_learner_params = nn.Parameter(torch.FloatTensor(self.N, self.N), requires_grad=True)

        self.lstm = nn.LSTM(
            batch_first=True,
            input_size=out_dim_1 + 1,
            hidden_size=hidden_dim_2,
            num_layers=2,
            bidirectional=False,
        )
        # self.lstm = nn.LSTM(batch_first=True, input_size=N*out_dim_1+N, hidden_size=N*hidden_dim_2,num_layers=1, bidirectional=False)
        self.out_dim_1 = out_dim_1  # out dimension of GNN
        self.hidden_dim_2 = hidden_dim_2  # out dimension of LSTM
        self.fc1 = nn.Linear(self.hidden_dim_2 + self.x_days, 1)
        self.v = torch.nn.Parameter(torch.empty(23))
        self.v.requires_grad = True
        torch.nn.init.normal_(self.v.data, mean=0.05, std=0.000)

    def run_specGCN_lstm(self, input_record, reconstruct_graph=False):
        N = self.N
        n_batch = len(input_record)
        x_days, y_days = round(len(input_record[0][0]) / 2), len(input_record[0][1])

        # Step1: calculate the SpecGCN output
        # lstm_input_batch = torch.zeros((N, n_batch, self.x_days, self.out_dim_1+1))
        lstm_input_batch = torch.zeros(
            (N, n_batch, self.x_days, self.out_dim_1 + 1), device=self.device
        )
        adj_output = torch.zeros(
            (n_batch, self.x_days, self.N, self.N), device=self.device
        )
        for batch in range(n_batch):
            # mobility
            # torch.tensor(input_record[batch][0][2*i]).float()     #(N,N)
            # text
            # torch.tensor(input_record[batch][0][2*i+1]).float()   #(N,text_dimension)
            # infection
            # torch.tensor(input_record[batch][2][i]).float()       #N
            
            for day in range(x_days):
                x = torch.tensor(input_record[batch][0][2 * day + 1]).float().to(self.device)  # (N, text_dimension)
                x_infection = torch.tensor(input_record[batch][2][day]).float().unsqueeze(-1).to(self.device)  # (N, 1)
                day_order = input_record[batch][3]  # v6
                adj_real = torch.tensor(input_record[batch][0][2 * day]).float()  # (N, N)

                if reconstruct_graph:
                    adj_degree = adj_real.sum(-1).unsqueeze(-1)
                    mat_degree = torch.mm(adj_degree, adj_degree.permute(1, 0))
                    mat_degree = torch.sigmoid(torch.mul(self.graph_learner_params, mat_degree.to(self.device)))
                    adj_real_spatial = torch.mul(mat_degree, adj_real.to(self.device))
                    adj = self.graph_learner(x.unsqueeze(0)).squeeze()

                    adj = adj + adj_real_spatial

                    adj_output[batch][day] = adj
                else:
                    adj = adj_real
                    adj = adj.to(self.device)

                specGCN_out = self.specGCN(x, adj)  # (N, out_dim1)
                specGCN_out = specGCN_out.mul(
                    torch.unsqueeze(
                        torch.exp(self.v * self.v * float(day_order)), dim=1
                    ).repeat(1, self.out_dim_1)
                )

                specGCN_out = torch.cat(
                    [specGCN_out, x_infection], dim=1
                )  # (N, out_dim1+1)
                for zone in range(N):
                    lstm_input_batch[zone][batch][day] = specGCN_out[zone]
        # lstm_input_batch.shape = [N, batch, x_days, out_dim_1+1]

        # Step2: calculate the LSTM outputs
        y_output_batch = torch.zeros((n_batch, self.y_days, self.N))
        for zone in range(N):
            lstm_input_x1 = lstm_input_batch[zone]  # [batch, x_days, out_dim_1+1]

            # lstm_input_x2 = torch.mean(lstm_input_x1, dim=1)
            # lstm_input_x2 = lstm_input_x2.view(
            #     (lstm_input_x2.size()[0], 1, lstm_input_x2.size()[1])
            # )  ##[batch, 1, out_dim_1+1]
            # lstm_input_x2 = lstm_input_x2.repeat(
            #     1, y_days, 1
            # )  ##[batch, y_days, out_dim_1+1]
            lstm_input_x2 = torch.mean(lstm_input_x1, dim=1).unsqueeze(1).repeat(1, y_days, 1)

            lstm_input_x1_output, (hc, cn) = self.lstm(lstm_input_x1)  # [batch, x_days, hidden_dim_2]
            lstm_output, (hc1, cn1) = self.lstm(lstm_input_x2, (hc, cn))  # [batch, y_days, hidden_dim_2]

            infection_tensor = (
                torch.tensor(
                    [
                        [
                            input_record[batch][2][x_day_c][zone]
                            for x_day_c in range(x_days)
                        ]
                        for batch in range(n_batch)
                    ]
                )
                .float()
                .to(self.device)
            )
            # infection_tensor = infection_tensor.view(
            #     (infection_tensor.size()[0], 1, infection_tensor.size()[1])
            # )
            # infection_tensor = infection_tensor.repeat(
            #     1, y_days, 1
            # )
            infection_tensor = infection_tensor.unsqueeze(dim=1).repeat(1, y_days, 1)  # [batch, y_days, x_days]

            lstm_output = torch.cat([lstm_output, infection_tensor], dim=2)

            fc_output = self.fc1(lstm_output)  # [batch, y_days, 1]
            fc_output = F.relu(fc_output)  # [batch, y_days, 1]
            if zone == 0:
                y_output_batch = fc_output
            else:
                y_output_batch = torch.cat([y_output_batch, fc_output], dim=2)
        return adj_output, y_output_batch  # y_output_batch.shape = (batch, y_day, N)
