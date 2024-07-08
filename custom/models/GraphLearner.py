import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv

class GraphLearner(nn.Module):
    def __init__(self, hid, tan_alpha=1):
        super(GraphLearner, self).__init__()
        self.hid = hid
        self.linear1 = nn.Linear(self.hid, self.hid)
        self.linear2 = nn.Linear(self.hid, self.hid)
        self.alpha = tan_alpha

        self.linear_layer = lambda linear, x: torch.tanh(self.alpha * linear(x))

    def forward(self, x):
        x1, x2 = self.linear_layer(self.linear1, x), self.linear_layer(self.linear2, x)
        x1_T, x2_T = x1.permute((0, 2, 1)), x2.permute((0, 2, 1))
        A = torch.bmm(x1, x2_T) - torch.bmm(x2, x1_T)
        A = self.alpha * A
        A = torch.relu(torch.tanh(A))
        return A

class GraphEncoder(nn.Module):
    def __init__(self, in_c, out_c):
        super(GraphEncoder, self).__init__()
        self.conv1 = GCNConv(in_c, 16)
        self.conv2 = GCNConv(16, out_c)
    
    def forward(self, x, A):
        x = self.conv1(x, A)
        x = F.relu(x)
        x = self.conv2(x, A)
        return x

class GraphDecoder(nn.Module):
    def __int__(self, latent_dim):
        super(GraphDecoder, self).__init__()
        self.latent_dim = latent_dim

    def forwaerd(self, z):
        A = torch.sigmoid(torch.matmul(z, z.t()))
        return A


