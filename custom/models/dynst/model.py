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
        x = self.cnn(inputs)
        pred = self.out(x)
        attention = self.att(x).softmax(2)
        edge_prob = (pred * attention).mean(dim=2)
        return edge_prob

class DynGraphEncoder(nn.Module):
    def  __init__(self, num_nodes, in_dim, hidden, num_heads, num_layers, dropout):
        super().__init__()
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.tcn = CNN(in_dim, hidden,hidden)
        self.hidden = hidden
        self.num_heads = 4
        self.global_attention = nn.MultiheadAttention(embed_dim = hidden, num_heads=num_heads)
        self.lstm = nn.LSTM(self.hidden * 2, self.hidden, num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden, 1)

    ## 32,28,47,1
    ## (32,14,47,1)  (32,7,47,1)
    ## 32 7 47 14
    def forward(self, x, gt, adp_g=None):
        batchsize = x.size(0)
        obs_len = x.size(1)
        pred_len = gt.size(1)
        seq = torch.cat((x, gt), dim=1).squeeze()
        adjust_x = torch.zeros((batchsize, pred_len, self.num_nodes, obs_len))

        for i in range(pred_len):
            adjust_x[:, i] = seq[:, i:i + obs_len].permute(0, 2, 1).contiguous()
        # (32,7,47,14)
        adjust_x = adjust_x.reshape(-1, obs_len).unsqueeze(1)
        x_tcn = self.tcn(adjust_x)
        x_tcn = x_tcn.reshape(batchsize, pred_len, self.num_nodes, self.hidden)
        x_tcn = x_tcn.permute(2, 0, 1, 3).reshape(self.num_nodes, batchsize * pred_len, self.hidden)
        x_global, attn_weights = self.global_attention(x_tcn, x_tcn, x_tcn)
        x_global = x_global.reshape(self.num_nodes, batchsize, pred_len, self.hidden).permute(1, 2, 0, 3)
        edge_features = torch.cat([x_global.unsqueeze(3).expand(-1, -1, -1, self.num_nodes, -1),
                                   x_global.unsqueeze(2).expand(-1, -1, self.num_nodes, -1, -1)], dim=-1)
        # lstm
        edge_features = edge_features.reshape(-1, pred_len, 2 * self.hidden)
        edge_features, _ = self.lstm(edge_features)
        edge_features = edge_features.reshape(batchsize, pred_len, self.num_nodes, self.num_nodes, self.hidden)

        edge_features = torch.sigmoid(self.fc(edge_features))

        mask = torch.eye(self.num_nodes).unsqueeze(0).unsqueeze(0).unsqueeze(-1).repeat(batchsize, pred_len, 1, 1, 1)
        edge_features = edge_features * (1 - mask)

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
    def __init__(self, in_dim, out_dim, hidden, graph_layers, dropout):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden = hidden
        self.graph_layers = graph_layers
        self.tcn = CNN(in_dim, hidden, hidden)
        self.GNNBlocks = nn.ModuleList(
            [GraphConvLayer(in_features=hidden, out_features=hidden) for i in range(graph_layers)])
        self.fc = nn.Linear(hidden * (graph_layers + 1), hidden)
        self.gru_cell = nn.GRUCell(hidden, hidden)
        self.out = nn.Linear(hidden, out_dim)

    def forward(self, x, gt, adj_t, use_predict=False):
        batchsize = x.size(0)
        obs_len = x.size(1)
        num_nodes = x.size(2)
        pred_len = gt.size(1)
        seq = torch.cat((x, gt), dim=1).squeeze()
        current_x = x
        gru_hidden = torch.zeros(batchsize * num_nodes, self.hidden)
        predict_list = []
        for i in range(pred_len):
            print(current_x.shape)
            current_x = current_x.permute(0, 2, 3, 1).contiguous()
            current_x = current_x.reshape(-1, 1, obs_len)
            x_tcn = self.tcn(current_x)
            x_tcn = x_tcn.reshape(batchsize, num_nodes, -1)
            adj = adj_t[:, i].squeeze()
            laplace_adj = getLaplaceMat(adj)
            node_state_list = [x_tcn]
            node_state = x_tcn
            for layer in self.GNNBlocks:
                node_state = layer(node_state, laplace_adj)
                node_state_list.append(node_state)
            node_state = torch.cat(node_state_list, dim=-1)
            node_state = node_state.reshape(-1, node_state.size(-1))

            node_state = self.fc(node_state)

            gru_hidden = self.gru_cell(node_state, gru_hidden)
            predict = self.out(gru_hidden)
            predict = predict.reshape(batchsize, num_nodes, -1)

            if use_predict:
                current_x = torch.cat((current_x[:, 1:], predict), dim=1).unsqueeze(-1)
            else:
                current_x = seq[:, i + 1: i + obs_len + 1].unsqueeze(-1)
            predict_list.append(predict)
        predict_list = torch.stack(predict_list, dim=1)
        return predict_list


