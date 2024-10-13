import torch.nn as nn
import math

class TCN(nn.Module):
    def __init__(self, n_in: int, n_hid: int, n_out: int, tcn_layers = 2, do_prob: float=0.):
        super(TCN, self).__init__()
        
        # More convolutional layers and residual connections
        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv1d(n_in, n_hid, kernel_size=2, stride=1, padding=0))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm1d(n_hid))
        self.layers.append(nn.Dropout(do_prob))
        
        # Add more convolutional layers
        for _ in range(tcn_layers - 1):  # num_layers is a parameter you set
            self.layers.append(nn.Conv1d(n_hid, n_hid, kernel_size=2, stride=1, padding=0))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.BatchNorm1d(n_hid))
        
        self.out = nn.Conv1d(n_hid, n_out, kernel_size=1)
        self.att = nn.Conv1d(n_hid, 1, kernel_size=1)
        self.init_weights()

    def forward(self, inputs):
        x = inputs.float()
        residual = x
        
        for layer in self.layers:
            x = layer(x)
        
        # Add residual connection
        x += residual
        
        pred = self.out(x)
        attention = self.att(x).softmax(2)
        edge_prob = (pred * attention).mean(dim=2)
        return edge_prob

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()