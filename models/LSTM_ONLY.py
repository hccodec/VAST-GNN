import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class LSTM_MODEL(nn.Module):
    def __init__(self, args, lstm_args):
        super().__init__()

        lstm_input_dim = args.xdays

        self.lstm = nn.LSTM(lstm_input_dim, lstm_args['lstm']['hid'], num_layers=2, batch_first=True)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(lstm_args['lstm']['hid'], lstm_args['linear']['hid'])
        self.fc_out = nn.Linear(lstm_args['linear']['hid'], args.ydays)

    def forward(self, X, y, A, A_y, extra_info=None, idx=None):
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

