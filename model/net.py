import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch.utils.data import DataLoader

class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.gnn_layer = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    # def forward(self, x):
    #     out, _ = self.gnn_layer(x)
    #     out = self.fc(out[:, -1, :])
    #     return out

    def forward(self, adj, nodes):

        # features = torch.cat((adj, nodes), dim=-1)
        extended_nodes = nodes.repeat(
            1,
            nodes.shape[0] // nodes.shape[1])
        features = torch.matmul(adj, extended_nodes)
        out, _ = self.gnn_layer(features)
        out = self.fc(out.unsqueeze(0)[:, -1, :])
        return out

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTM, self).__init__()
        self.lstm_layer = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm_layer(x)
        out = self.fc(out.unsqueeze(0)[:, -1, :])
        return out

def compute_loss(
        pred_graph, target_graph, pred_links, target_links,
        graph_loss_weight=0.5
):
    criterion = nn.MSELoss()

    target_graph = target_graph.to_dense()
    target_links = target_links.to_dense()
    
    graph_loss = criterion(pred_graph.squeeze(0), target_graph.mean(dim=1))
    link_loss = criterion(pred_links, target_links)

    total_loss = graph_loss_weight * graph_loss + \
        (1 - graph_loss_weight) * link_loss
    
    return total_loss


# 定义 GNN+LSTM 模型
class GNNLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNLSTM, self).__init__()
        
        # GNN 层
        self.gcn_conv = GCNConv(input_dim, hidden_dim)
        
        # LSTM 层
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        # 输出层
        self.fc = nn.Linear(hidden_dim, output_dim)

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        
    def forward(self, edge_index: torch.Tensor, x: torch.Tensor):

        weight = edge_index.coalesce().values()
        adj = edge_index.coalesce().indices()

        # GNN 前向传播
        x = self.gcn_conv(x, adj, edge_weight=weight)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.dropout(x)
        
        # LSTM 前向传播
        x = x.unsqueeze(0)  # 添加时间维度
        x, _ = self.lstm(x)
        x = x.squeeze(0)  # 移除时间维度
        
        # 输出层前向传播
        x = self.fc(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x