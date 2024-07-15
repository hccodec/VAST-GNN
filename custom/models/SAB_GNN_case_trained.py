import torch
from torch import nn
import torch.nn.functional as F
from models.GraphLearner import GraphLearner

class SpecGCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.5, concat=True):
        super(SpecGCNLayer, self).__init__()
        # self.input_dim, self.hidden_dim = input_dim, output_dim
        self.dropout, self.concat = dropout, concat

        self.W = nn.Parameter(torch.empty((input_dim, output_dim)), requires_grad=True)
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

    def forward(self, feature, DnADn):
        feature_new = torch.mm(feature.float(), self.W)
        feature_new = F.dropout(feature_new, self.dropout, training=self.training)
        H = torch.mm(DnADn.float(), feature_new)
        return H


class SpecGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout, device="cpu"):
        super(SpecGCN, self).__init__()
        self.dropout = dropout
        self.layer1 = SpecGCNLayer(
            input_dim, output_dim, dropout=dropout, concat=True
        ).to(device)
        self.device = device

    def compute_DnADn(self, A):
        tilde_A = A + torch.eye(A.size(0)).to(self.device)
        tilde_D = torch.diag(torch.pow(tilde_A.sum(-1).float(), -0.5))
        DnADn = torch.mm(tilde_D, torch.mm(tilde_A, tilde_D))
        return DnADn

    def forward(self, x, A):
        DnADn = self.compute_DnADn(A)
        x1 = self.layer1(x, DnADn)
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x_out = F.relu(x1)
        return x_out


class Multiwave_SpecGCN_LSTM_CASE_TRAINED(nn.Module):

    def __init__(self, args, model_args):
        super(Multiwave_SpecGCN_LSTM_CASE_TRAINED, self).__init__()
        self.device = args.device

        self.x_days = args.xdays
        self.y_days = args.ydays
        _, self.N, self.input_dim = model_args["shape"][2]

        self.lstm_input_dim, self.lstm_output_dim = model_args["specGCN"]["out"], model_args["lstm"]["hid"]

        self.specGCN = SpecGCN(
            self.input_dim, *model_args["specGCN"].values(), device=self.device
        ).to(self.device)
        self.lstm = nn.LSTM(
            batch_first=True,
            input_size=self.lstm_input_dim,
            hidden_size=self.lstm_output_dim,
            num_layers=2,
            bidirectional=False,
        ).to(self.device)
        self.fc1 = nn.Linear(self.lstm_output_dim, 1).to(self.device)

        self.v = torch.nn.Parameter(torch.empty(23), requires_grad=True)
        torch.nn.init.normal_(self.v.data, mean=0.05, std=0.)

        if args.enable_graph_learner:
            self.graph_learner = GraphLearner(self.input_dim).to(self.device)
        

    def forward(self, mobility, text, casex, idx):
        batch_size = mobility.size(0)

        lstm_input = torch.empty((batch_size, self.x_days, self.N, self.lstm_input_dim)).to(self.device)
        lstm_output = torch.empty((self.N, batch_size, self.y_days)).to(self.device)

        for batch in range(mobility.size(0)):
            # day = idx[batch]
            for i in range(self.x_days):
                adj = mobility[batch][i].float()
                # x = text[batch][i].float()
                case = casex[batch][i].float()
                # specGCN_out = self.specGCN(x, adj)
                specGCN_out = self.specGCN(case, adj)

                # social_recovery_vec = torch.exp(float(day) * self.v ** 2).unsqueeze(-1).repeat(1, specGCN_out.size(-1)).to(self.device)

                # specGCN_out = specGCN_out.mul(social_recovery_vec)
                # specGCN_out = torch.cat([specGCN_out, case[:,-1].unsqueeze(-1)], -1)
                lstm_input[batch][i] = specGCN_out

        lstm_input = lstm_input.permute(2, 0, 1, 3) # N, batch_size, self.x_days, self.lstm_input_dim

        for zone in range(self.N):
            x1 = lstm_input[zone]
            x2 = torch.mean(x1, 1).unsqueeze(1).repeat(1, self.y_days, 1)
            x1_out, (hc, cn) = self.lstm(x1)
            out, (hc1, cn1) = self.lstm(x2, (hc, cn))

            # cases = casex[:,:,zone,-1].unsqueeze(1).repeat(1, self.y_days, 1).float()
            # out = torch.cat([out, cases], -1)
            out_fc = F.relu(self.fc1(out))

            lstm_output[zone] = out_fc.squeeze(-1).unsqueeze(0)

        lstm_output = lstm_output.permute(1, 2, 0) # batch_size, self.y_days, self.N

        return lstm_output
