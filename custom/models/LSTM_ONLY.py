import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class LSTM_MODEL(nn.Module):
    def __init__(self, args, lstm_args):
        super().__init__()

        self.with_text = lstm_args['with_text']
        text_dim = lstm_args['shape'][-1]
        lstm_input_dim = (1 + text_dim) if self.with_text else args.xdays

        self.lstm = nn.LSTM(lstm_input_dim, lstm_args['lstm']['hid'], num_layers=2, batch_first=True)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(lstm_args['lstm']['hid'], lstm_args['linear']['hid'])
        self.fc_out = nn.Linear(lstm_args['linear']['hid'], args.ydays)
        
    def forward(self, mobility, text, casex, idx):
        casex = casex.float()
        text = text.float()
        
        x = casex[:,-1].float()
        batch_size, num_zones, _ = x.shape
        if self.with_text: 
            x = torch.cat([x.unsqueeze(-1), text.transpose(1, 2)], -1)
        x = x.flatten(0, 1)
        x, (hc, cn) = self.lstm(x)
        # x = x.flatten(-2, -1)
        if self.with_text: x = x[:, -1]
        x = self.relu(self.fc(x))
        x = self.fc_out(x)
        x = x.view(batch_size, num_zones, -1).transpose(-2, -1)
        return x

