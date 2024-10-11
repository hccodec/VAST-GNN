import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class CNN(nn.Module):
    def __init__(self, n_in: int, n_hid: int, n_out: int, do_prob: float=0.):
        """
        Args:
            n_in: input dimension
            n_hid: dimension of hidden layers
            n_out: output dimension
        """
        super(CNN, self).__init__()
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
            nn.BatchNorm1d(n_hid)
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
        attention = self.att(x).softmax(2)
        edge_prob = (pred * attention).mean(dim=2)
        return edge_prob


class DynGraphEncoder(nn.Module):
    def  __init__(self, in_dim, hidden, num_heads, num_layers, dropout, device):
        super().__init__()
        # self.dropout = dropout
        self.tcn = CNN(1, hidden, hidden, dropout).to(device)
        self.hidden = hidden
        self.num_heads = 4
        self.global_attention = nn.MultiheadAttention(embed_dim = hidden, num_heads=num_heads)
        self.lstm = nn.LSTM(self.hidden * 2, self.hidden, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(self.hidden, 1)
        self.device = device

    def forward(self, x, gt):
        batch_size, obs_len, num_nodes, feature_len = x.size()
        pred_len = gt.size(1)

        # seq = torch.cat((x, gt), dim=1).squeeze(-1)
        _seq = torch.cat([x[:, 0, :, :-1], x[:,:,:,-1].transpose(1, 2), gt.squeeze(-1).transpose(1, 2)], dim=-1)
        adjust_x = torch.zeros((batch_size, obs_len + pred_len, num_nodes, feature_len)).to(self.device)

        # for i in range(pred_len): adjust_x[:, i] = _seq[:, :, i : i + obs_len].permute(0, 2, 1).contiguous()
        for i in range(obs_len + pred_len):
            adjust_x[:, i] = _seq[:, :, i : i + feature_len]

        adjust_x = adjust_x.flatten(0, -2).unsqueeze(1)
        x_tcn = self.tcn(adjust_x)
        x_tcn = x_tcn.reshape(batch_size, pred_len + obs_len, num_nodes, self.hidden)
        x_tcn = x_tcn.permute(2, 0, 1, 3).reshape(num_nodes, batch_size * (pred_len + obs_len), self.hidden)
        x_global, attn_weights = self.global_attention(x_tcn, x_tcn, x_tcn)
        x_global = x_global.reshape(num_nodes, batch_size, (pred_len + obs_len), self.hidden).permute(1, 2, 0, 3)
        edge_features = torch.cat([x_global.unsqueeze(3).expand(-1, -1, -1, num_nodes, -1),
                                   x_global.unsqueeze(2).expand(-1, -1, num_nodes, -1, -1)], dim=-1)
        # lstm
        edge_features = edge_features.reshape(-1, (pred_len + obs_len), 2 * self.hidden)
        edge_features, _ = self.lstm(edge_features)
        edge_features = edge_features.reshape(batch_size, (pred_len + obs_len), num_nodes, num_nodes, self.hidden)

        edge_features = torch.sigmoid(self.fc(edge_features))

        mask = torch.eye(num_nodes).unsqueeze(0).unsqueeze(0).repeat(batch_size, (pred_len + obs_len), 1, 1).to(self.device)
        edge_features = edge_features.squeeze(-1) * (1 - mask)

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


def getLaplaceMat(adj):
    batch_size, m, _ = adj.size()
    i_mat = torch.eye(m).to(adj.device)
    i_mat = i_mat.unsqueeze(0)
    o_mat = torch.ones(m).to(adj.device)
    o_mat = o_mat.unsqueeze(0)
    i_mat = i_mat.expand(batch_size, m, m)
    o_mat = o_mat.expand(batch_size, m, m)
    adj = torch.where(adj > 0, o_mat, adj)
    '''
    d_mat = torch.bmm(adj, adj.permute(0, 2, 1))
    d_mat = torch.where(i_mat>0, d_mat, i_mat)
    print('d_mat version 1', d_mat)
    '''
    d_mat_in = torch.sum(adj, dim=1)
    d_mat_out = torch.sum(adj, dim=2)
    d_mat = torch.sum(adj, dim=2)  # attention: dim=2
    d_mat = d_mat.unsqueeze(2)
    d_mat = d_mat + 1e-12
    # d_mat = torch.pow(d_mat, -0.5) if is 1/2
    d_mat = torch.pow(d_mat, -1)
    d_mat = d_mat.expand(d_mat.shape[0], d_mat.shape[1], d_mat.shape[1])
    d_mat = i_mat * d_mat

    # laplace_mat = d_mat * adj * d_mat
    laplace_mat = torch.bmm(d_mat, adj)
    # laplace_mat = torch.bmm(laplace_mat, d_mat)
    return laplace_mat


class Decoder(nn.Module):
    def __init__(self, in_dim, out_dim, hidden, graph_layers, dropout, device):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden = hidden
        self.graph_layers = graph_layers
        self.tcn = CNN(1, hidden, hidden, dropout).to(device)
        self.GNNBlocks = nn.ModuleList(
            [GraphConvLayer(in_features=hidden, out_features=hidden) for i in range(graph_layers)])
        self.fc = nn.Linear(hidden * (graph_layers + 1), hidden)
        self.gru_cell = nn.GRUCell(hidden, hidden)
        self.out = nn.Linear(hidden, out_dim)
        self.device = device

    def forward(self, x, gt, adj_t, use_predict = False):
        batch_size, obs_len, num_nodes, pred_len = x.size(0), x.size(1), x.size(2), gt.size(1)

        _seq = torch.cat([x[:, 0, :, :-1], x[:, :, :, -1].transpose(1, 2), gt.squeeze(-1).transpose(1, 2)], dim=-1)

        current_x = x[:, 0]
        gru_hidden = torch.zeros(batch_size * num_nodes, self.hidden).to(self.device)
        predict_list = []
        for i in range(obs_len + pred_len):
            # current_x = current_x.permute(0, 2, 3, 1).contiguous()
            x_tcn = self.tcn(current_x.flatten(0, -2).unsqueeze(1))
            x_tcn = x_tcn.reshape(batch_size, num_nodes, -1)
            adj = adj_t[:, i]
            node_state_list = [x_tcn]
            node_state = x_tcn

            # TODO: 以 105 pooling 初始化，神经网络 源节点

            # GNN
            for layer in self.GNNBlocks:
                node_state = layer(node_state, adj)
                node_state_list.append(node_state)
            node_state = torch.cat(node_state_list, dim=-1)
            node_state = node_state.flatten(0, -2)

            node_state = self.fc(node_state)

            gru_hidden = self.gru_cell(node_state, gru_hidden)
            predict = self.out(gru_hidden)
            predict = predict.reshape(batch_size, num_nodes, -1)

            if i < obs_len:
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
    def __init__(self, in_dim, out_dim, hidden_enc, hidden_dec, num_heads, num_layers, graph_layers, dropout = 0, device = torch.device('cpu'), no_graph_gt = False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_enc = hidden_enc
        self.hidden_dec = hidden_dec
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.graph_layers = graph_layers

        self.device = device
        self.no_graph_gt = no_graph_gt

        self.enc = DynGraphEncoder(in_dim, hidden_enc, num_heads, num_layers, dropout, device).to(device)
        self.dec = Decoder(in_dim, out_dim, hidden_dec, graph_layers, dropout, device).to(device)

    def forward(self, X, y, A, A_y, adj_lambda):

        adj_output = self.enc(X, y) # enc 输出的图结构，不可更改，用于返回值

        adj_enc = adj_output # 使用 adj_enc 传入 dec

        # 求图结构 gt 和 enc_output 的线性结果
        if self.training and not self.no_graph_gt and adj_lambda is not None:
            adj_gt = torch.cat((A, A_y), dim=1)

            # （√）adj_enc 的 mu sigma 和 adj_gt 一致（√）
            # (已解决，在外部对 adj_enc 按照 mu 执行 adj_enc -> adj_gt 的尺度缩放)

            adj_enc = getLaplaceMat(adj_enc.flatten(0, 1)).reshape(adj_output.shape)
            adj_gt = getLaplaceMat(adj_gt.flatten(0, 1)).reshape(adj_output.shape)

            adj_enc = (1 - adj_lambda) * adj_enc + adj_lambda * adj_gt

            # 此处已针对具体实验进行临时更改：只传入 adj_output 和只传入 adj_gt （均仅在下一句执行Laplace归一化）
            # adj_enc = getLaplaceMat(adj_enc.flatten(0, 1)).reshape(adj_output.shape)
            # adj_enc = getLaplaceMat(adj_output.flatten(0, 1)).reshape(adj_output.shape)
            # adj_enc = getLaplaceMat(adj_gt.flatten(0, 1)).reshape(adj_output.shape)

        y_hat = self.dec(X, y, adj_enc.float(), not self.training)
        return y_hat, adj_output
