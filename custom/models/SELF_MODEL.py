import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from models.GraphLearner import GraphLearner

class SelfModel(nn.Module):
    def __init__(self, args, self_model_args):
        super().__init__()

        self.x_days, self.y_days = args.xdays, args.ydays
        self.num_zones, self.feature_dim = self_model_args['shape']

        self.in_c, self.hid_c, self.out_c = self_model_args['in'], self_model_args['hid'], self_model_args['out']

        self.lstm = nn.LSTM(input_size=self.x_days, hidden_size=self.hid_c, batch_first=True)
        self.conv1 = GCNConv(self.in_c, self.hid_c)
        self.conv2 = GCNConv(self.hid_c, self.out_c)
        self.fc = nn.Linear(self.out_c * self.x_days, self.y_days * self.num_zones)
        self.graph_learner = GraphLearner()

    def forward(self, mobility, text, casex, idx):
        batch_size = casex.size(0)

        mobility = mobility.float()
        text = text.float()
        casex = casex.float()
        
