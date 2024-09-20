import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GraphLearner(nn.Module):
    def __init__(self, hid, tan_alpha=1):
        super(GraphLearner, self).__init__()
        self.hid = hid
        self.linear1 = nn.Linear(self.hid, self.hid)
        self.linear2 = nn.Linear(self.hid, self.hid)
        self.alpha = tan_alpha


    def linear_layer(self, linear, x):
        return torch.tanh(self.alpha * linear(x))

    def forward(self, x):
        x1, x2 = self.linear_layer(self.linear1, x), self.linear_layer(self.linear2, x)
        x1_T, x2_T = x1.permute((0, 2, 1)), x2.permute((0, 2, 1))
        A = torch.bmm(x1, x2_T) - torch.bmm(x2, x1_T)
        A = self.alpha * A
        A = torch.relu(torch.tanh(A))
        return A