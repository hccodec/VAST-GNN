import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from utils.utils import getLaplaceMat, rm_self_loops

# region TCN
class TCN(nn.Module):
    def __init__(self, n_in: int, n_hid: int, n_out: int, num_layers: int = 2, do_prob: float=0.):
        """
        Args:
            n_in: input dimension
            n_hid: dimension of hidden layers
            n_out: output dimension
        """
        super(TCN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_in, n_hid, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm1d(n_hid),
            nn.Dropout(do_prob),
            nn.MaxPool1d(kernel_size=2, stride=None, padding=0,
                         dilation=1, return_indices=False,
                         ceil_mode=False),
            nn.Conv1d(n_hid, n_hid, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm1d(n_hid),
        )
        self.out = nn.Conv1d(n_hid, n_out, kernel_size=1)
        self.att = nn.Conv1d(n_hid, 1, kernel_size=1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs):
        x = self.cnn(inputs.float())
        pred = self.out(x)
        # return pred
        attention = self.att(x).softmax(2)
        edge_prob = (pred * attention).mean(dim=2)
        return edge_prob
# endregion
# from models.layers.TCN import TCN

class DynGraphEncoder(nn.Module):
    def  __init__(self, in_dim, hidden, num_heads, tcn_layers, lstm_layers, dropout, device):
        super().__init__()
        # self.dropout = dropout
        self.tcn = TCN(1, hidden, hidden, tcn_layers, dropout).to(device)
        self.hidden = hidden
        self.num_heads = 4
        self.global_attention = nn.MultiheadAttention(embed_dim = hidden, num_heads=num_heads)
        self.lstm = nn.LSTM(self.hidden * 2, self.hidden, lstm_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(self.hidden, 1)
        self.device = device

    def forward(self, x, gt):
        batch_size, obs_len, num_nodes, feature_len = x.size()
        pred_len = gt.size(1)

        # 根据 x 将 y 也拼接成窗口形式，使 y 像 x 一样能作为特征支持递归预测
        _seq = torch.cat([x[:, 0, :, :-1], x[:,:,:,-1].transpose(1, 2), gt.squeeze(-1).transpose(1, 2)], dim=-1)
        total_features = torch.zeros((batch_size, obs_len + pred_len, num_nodes, feature_len)).to(self.device)
        for i in range(obs_len + pred_len): total_features[:, i] = _seq[:, :, i : i + feature_len]

        # tcn 卷积
        total_features = total_features.flatten(0, -2).unsqueeze(1)
        x_tcn = self.tcn(total_features)
        x_tcn = x_tcn.reshape(batch_size, obs_len + pred_len, num_nodes, self.hidden)
        # 注意力计算 (最后将所有边的特征进行两两拼接)
        x_tcn = x_tcn.permute(2, 0, 1, 3).reshape(num_nodes, -1, self.hidden)
        x_global, attn_weights = self.global_attention(x_tcn, x_tcn, x_tcn)
        x_global = x_global.reshape(num_nodes, batch_size, obs_len + pred_len, self.hidden).permute(1, 2, 0, 3)
        edge_features = torch.cat([x_global.unsqueeze(3).expand(-1, -1, -1, num_nodes, -1),
                                   x_global.unsqueeze(2).expand(-1, -1, num_nodes, -1, -1)], dim=-1)
        
        # lstm
        edge_features = edge_features.permute(0, 2, 3, 1, 4).flatten(0, -3) # 这是真的 bug 2024年10月20日10点41分
        edge_features, _ = self.lstm(edge_features)
        edge_features = edge_features.reshape(batch_size, num_nodes, num_nodes, (obs_len + pred_len), self.hidden)\
            .permute(0, 3, 1, 2, 4) # 这是真的 bug 2024年10月20日10点41分

        edge_features = torch.sigmoid(self.fc(edge_features)).squeeze(-1)

        # mask = torch.eye(num_nodes).unsqueeze(0).unsqueeze(0).repeat(batch_size, (obs_len + pred_len), 1, 1).to(self.device)
        # edge_features = edge_features * (1 - mask)
        edge_features = rm_self_loops(edge_features)

        return edge_features


class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.act = nn.ELU()
        nn.init.xavier_uniform_(self.weight)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
            stdv = 1. / math.sqrt(self.bias.size(0))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, feature, adj):
        support = torch.matmul(feature, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return self.act(output + self.bias)
        else:
            return self.act(output)

class Decoder(nn.Module):
    def __init__(self, in_dim, out_dim, hidden, window_size, tcn_layers, graph_layers, dropout, device, no_virtual_node = True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden = hidden
        self.graph_layers = graph_layers
        self.no_virtual_node = no_virtual_node

        self.tcn = TCN(1, hidden, hidden, tcn_layers, dropout).to(device)
        self.tcn_mlp = nn.Linear(window_size, hidden)
        # self.tcn_out_fc = nn.Linear(hidden * 2, hidden)
        
        # 这里第一层 GNN 的 in_features 随上面 TCN 的拼接策略更改
        self.GNNBlocks = nn.ModuleList((GraphConvLayer(in_features=hidden * 2, out_features=hidden),
            *[GraphConvLayer(in_features=hidden, out_features=hidden) for i in range(graph_layers - 1)]
            ))
        self.fc = nn.Linear(self.GNNBlocks[0].in_features + sum([l.out_features for l in self.GNNBlocks]), hidden)

        self.gru_cell = nn.GRUCell(hidden, hidden)
        # self.out = nn.Linear(hidden, out_dim)
        self.out = nn.Sequential(
            nn.Linear(hidden + window_size, hidden),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden, out_dim),
            nn.ReLU()
        )
        self.device = device
    
    def TCN_Module(self, current_x):
        batch_size, num_nodes, _ = current_x.shape
        # TCN
        # 以下 TCN 模块输出方式三选一
        # 1. 仅 TCN
        x_tcn = current_x.flatten(0, -2).unsqueeze(1)
        x_tcn = self.tcn(x_tcn)
        x_tcn = x_tcn.reshape(batch_size, num_nodes, -1)
        x_tcn_out = x_tcn
        # 2. 仅 MLP
        x_tcn_mlp = self.tcn_mlp(current_x)
        x_tcn_out = x_tcn_mlp
        # 3.拼接 TCN 和 MLP
        # x_tcn_out = self.tcn_out_fc(torch.cat((x_tcn, x_tcn_mlp), dim = -1))
        # x_tcn_out = torch.cat((x_tcn, current_x), dim = -1)
        x_tcn_out = torch.cat((x_tcn, x_tcn_mlp), dim = -1)
        return x_tcn_out

    def GNN_Module(self, x_tcn_out, adj):

        if self.no_virtual_node:
            x_tcn_new = x_tcn_out
            adj_new = adj
        else:
            batch_size = x_tcn_out.shape[0]
            # 添加缺失节点
            x_virtual = x_tcn_out.mean(dim=1, keepdim=True)
            x_tcn_new = torch.cat([x_tcn_out, x_virtual], dim=1)
            
            missing_adj_row = adj.mean(dim=1, keepdim=True)
            missing_adj_col = torch.cat([adj.mean(dim=2, keepdim=True), torch.zeros(batch_size, 1, 1).to(self.device)], dim=1)

            adj_new = torch.cat([adj, missing_adj_row], dim=1) 
            adj_new = torch.cat([adj_new, missing_adj_col], dim=2)
            
        # 图卷积
        node_state = [x_tcn_new]
        for layer in self.GNNBlocks: node_state.append(layer(node_state[-1], adj_new))
        node_state = torch.cat(node_state, dim=-1)
        
        if not self.no_virtual_node:
            # 移除缺失节点
            node_state = node_state[:, :-1]
        return node_state

    def forward(self, x, gt, adj_t, use_predict = False):
        batch_size, obs_len, num_nodes, pred_len = x.size(0), x.size(1), x.size(2), gt.size(1)

        _seq = torch.cat([x[:, 0, :, :-1], x[:, :, :, -1].transpose(1, 2), gt.squeeze(-1).transpose(1, 2)], dim=-1)

        current_x = x[:, 0]
        gru_hidden = torch.zeros(batch_size * num_nodes, self.hidden).to(self.device)
        predict_list = []
        for i in range(obs_len + pred_len - 1):
            # TCN
            x_tcn_out = self.TCN_Module(current_x)
            # GNN
            node_state = self.GNN_Module(x_tcn_out, adj_t[:, i])
            # GRU
            node_state = node_state.flatten(0, -2)
            node_state = self.fc(node_state)
            gru_hidden = self.gru_cell(node_state, gru_hidden)
            # predict = self.out(gru_hidden)

            # Output
            predict = self.out(torch.cat((gru_hidden, current_x.flatten(0, -2)), dim=-1))
            predict = predict.reshape(batch_size, num_nodes, -1)

            # 
            if i < obs_len - 1:
                current_x = _seq[:, :, i + 1 : i + obs_len + 1]
            else:
                if use_predict: current_x = torch.cat((current_x[:, :, 1:], predict), dim=-1)
                else: current_x = _seq[:, :, i + 1 : i + obs_len + 1]
                predict_list.append(predict)

        predict_list = torch.stack(predict_list, dim=1)
        return predict_list


class dynst_extra_info():
    def __init__(self, lambda_A, dataset_extra = None):
        self.lambda_A = lambda_A
        self.dataset_extra = dataset_extra

    # def __init__(self, epoch, max_epochs, lambda_range = [0.1, 1.0], dataset_extra = None):
    #     assert hasattr(lambda_range, "__len__") and len(lambda_range == 2) and lambda_range[1] - lambda_range[0] > 0

    #     self.lambda_range = lambda_range
    #     self.epoch = epoch
    #     self.max_epochs = max_epochs
    #     self.dataset_extra = dataset_extra

    # def get_lambda(self):
    #     return self.lambda_range[1] - (self.lambda_range[1] - self.lambda_range[0]) * (self.epoch / self.max_epochs)

class dynst(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_enc, hidden_dec, window_size, num_heads, tcn_layers, lstm_layers, graph_layers, dropout = 0, device = torch.device('cpu'),
                 no_graph = False, no_virtual_node = False):
        super().__init__()
        # self.in_dim = in_dim
        # self.out_dim = out_dim
        # self.hidden_enc = hidden_enc
        # self.hidden_dec = hidden_dec
        # self.num_heads = num_heads
        # self.tcn_layers = tcn_layers
        # self.graph_layers = graph_layers

        self.device = device
        self.no_graph = no_graph
        self.no_virtual_node = no_virtual_node

        self.enc = DynGraphEncoder(in_dim, hidden_enc, num_heads, tcn_layers, lstm_layers, dropout, device).to(device)
        self.dec = Decoder(in_dim, out_dim, hidden_dec, window_size, tcn_layers, graph_layers, dropout, device, no_virtual_node).to(device)

    def forward(self, X, y, A, A_y, adj_lambda):
        
        if self.no_graph:
            # # 处理真实矩阵
            adj_gt = torch.cat((A, A_y), dim=1)
            adj_gt = rm_self_loops(adj_gt)
            # adj_gt = getLaplaceMat(adj_gt)
            adj_enc = adj_gt
        else:
            adj_enc = self.enc(X, y) # enc 输出的图结构，不可更改，用于返回值
            # adj_enc_laplaced = getLaplaceMat(adj_enc)

        y_hat = self.dec(X, y, adj_enc, not self.training)
        return y_hat, adj_enc
