import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import scipy.sparse as sp
import numpy as np

class LSTM_MODEL(nn.Module):
    def __init__(self, args, lstm_args):
        super().__init__()

        self.train_with_text = args.train_with_extrainfo
        text_dim = lstm_args['shape'][1][-1]
        lstm_input_dim = (1 + text_dim) if self.train_with_text else args.xdays

        self.lstm = nn.LSTM(lstm_input_dim, lstm_args['lstm']['hid'], num_layers=2, batch_first=True)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(lstm_args['lstm']['hid'], lstm_args['linear']['hid'])
        self.fc_out = nn.Linear(lstm_args['linear']['hid'], args.ydays)

    def forward(self, X, y, A, extra_info=None, idx=None):
        X = X.float()
        
        x = X[:,-1].float()
        batch_size, num_zones, _ = x.shape
        if self.train_with_text: 
            extra_info = extra_info.float()
            x = torch.cat([x.unsqueeze(-1), extra_info.transpose(1, 2)], -1)
        x = x.flatten(0, 1)
        x, (hc, cn) = self.lstm(x)
        # x = x.flatten(-2, -1)
        if self.train_with_text: x = x[:, -1]
        x = self.relu(self.fc(x))
        x = self.fc_out(x)
        x = x.view(batch_size, num_zones, -1).transpose(-2, -1)
        return x


class MPNN_LSTM(nn.Module):
    def __init__(self, nfeat: int,
                 nhid: int,
                 nout: int,
                 n_nodes: int,
                 window: int,
                 dropout: float):
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

    def forward(self, X, y, A, extra_info=None, idx=None):
        x, adj = X.view(-1, self.nfeat).float(), to_sparse(A)
        
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
        x = x.view(-1, self.window, self.n_nodes, x.size(1))
        x = torch.transpose(x, 0, 1)
        x = x.contiguous().view(self.window, -1, x.size(3))

        # 第一个LSTM层
        x, (hn1, cn1) = self.rnn1(x)
        # 第二个LSTM层
        out2, (hn2, cn2) = self.rnn2(x)

        # use the hidden states of both rnns
        # 使用两个LSTM层的隐藏状态
        x = torch.cat([hn1[0, :, :], hn2[0, :, :]], dim=1)
        skip = skip.reshape(skip.size(0), -1)

        x = torch.cat([x, skip], dim=1)

        # 第一个全连接层
        x = self.relu(self.fc1(x))
        x = self.dropout(x)

        # 第二个全连接层
        x = self.relu(self.fc2(x)).squeeze()
        x = x.view(-1)

        x = x.reshape(-1, 1, self.n_nodes, 1)
        return x

        
# def sparse_mx_to_torch_sparse_tensor(_tensor):
#     """
#     Convert a scipy sparse matrix to a torch sparse tensor.
#     """
#     sparse_mx = sp.block_diag(_tensor)
#     sparse_mx = sparse_mx.tocoo().astype(np.float32)
#     indices = torch.from_numpy(
#         np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
#     values = torch.from_numpy(sparse_mx.data)
#     shape = torch.Size(sparse_mx.shape)
#     return torch.sparse.FloatTensor(indices, values, shape)


def to_sparse(batch_tensor):
    """
    Convert a batch of dense tensors to a sparse block diagonal tensor.

    batch_tensor: A torch.Tensor of shape (batch_size, time_steps, N, N)
                  where N is the size of each 2D matrix.
    Returns:
        A sparse block diagonal torch.sparse.FloatTensor.
    """
    batch_size, time_steps, N, _ = batch_tensor.shape

    # Flatten the batch and time dimension to loop over the matrices
    num_blocks = batch_size * time_steps
    block_size = N

    indices_list = []
    values_list = []

    for i in range(num_blocks):
        # Get the current matrix
        matrix = batch_tensor.view(num_blocks, N, N)[i]
        
        # Get non-zero indices and values
        indices = matrix.nonzero(as_tuple=False).t()
        values = matrix[indices[0], indices[1]]

        # Shift the indices to the correct block diagonal position
        row_shift = (i * block_size)
        indices_shifted = indices + row_shift

        # Store the shifted indices and values
        indices_list.append(indices_shifted)
        values_list.append(values)

    # Concatenate all indices and values across the blocks
    all_indices = torch.cat(indices_list, dim=1)
    all_values = torch.cat(values_list)

    # Define the overall shape of the sparse block diagonal matrix
    total_size = num_blocks * block_size
    shape = torch.Size([total_size, total_size])

    # Create the sparse tensor
    sparse_tensor = torch.sparse_coo_tensor(all_indices, all_values, shape).float()

    return sparse_tensor
