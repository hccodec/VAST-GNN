import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data, DataLoader
from models.GraphLearner import GraphLearner
from models.SAB_GNN import SpecGCN


class GraphEncoder(nn.Module):
    def __init__(self, in_c, hid_c, out_c, dropout=0.5, device="cpu"):
        super(GraphEncoder, self).__init__()
        self.conv1 = GCNConv(in_c, hid_c)
        self.conv2 = GCNConv(hid_c, out_c)
        self.dropout = dropout
        self.specGCN = SpecGCN(in_c, hid_c, out_c, dropout, device=device)

    def forward(self, x, A):
        z = []
        for batch in range(x.size(0)):
            _x, _A = x[batch], A[batch]
            edge_index, edge_weight = dense_to_sparse(_A)

            _x = self.conv1(_x, edge_index, edge_weight)
            _x = F.relu(_x)
            _x = F.dropout(_x, self.dropout, training=self.training)

            _x = self.conv2(_x, edge_index, edge_weight)
            _x = F.dropout(_x, self.dropout, training=self.training)
            # _x = self.specGCN(x[batch], A[batch])

            z.append(_x.unsqueeze(0))
        z = torch.cat(z)
        return z


class GraphDecoder(nn.Module):
    def __init__(self, latent_dim):
        super(GraphDecoder, self).__init__()
        self.latent_dim = latent_dim

    def forward(self, z):
        A = torch.sigmoid(torch.matmul(z, z.t()))
        return A


class SelfModel(nn.Module):
    def __init__(self, args, self_model_args):
        super().__init__()

        self.x_days, self.num_zones, self.feature_dim = self_model_args["shape"][2]
        self.text_feature_dim = self_model_args["shape"][1][2]
        self.y_days = self_model_args["shape"][3][0]

        self.train_with_text = args.train_with_extrainfo

        if args.enable_graph_learner: self.graph_learner = GraphLearner(self.feature_dim)

        self.graph_encoder = GraphEncoder(
            self.feature_dim,
            self_model_args["gnn"]["hid"],
            self_model_args["gnn"]["out"],
            device=args.device
        )

        if self.train_with_text:
            # self.text_graph_encoder = GraphEncoder(
            #     self.text_feature_dim,
            #     self_model_args["gnn"]["hid"],
            #     self_model_args["gnn"]["out"],
            # )
            self.text_fc = nn.Linear(
                self.text_feature_dim, self_model_args["gnn"]["out"]
            )

        # self.graph_decoder = GraphDecoder()
        self.lstm = nn.LSTM(
            input_size=(
                (self_model_args["gnn"]["out"] * 2)
                if self.train_with_text
                else self_model_args["gnn"]["out"]
            ),
            hidden_size=self_model_args["lstm"]["hid"][0],
            batch_first=True,
        )
        self.lstm2 = nn.LSTM(
            input_size=self_model_args["lstm"]["hid"][0],
            hidden_size=self_model_args["lstm"]["hid"][1],
            batch_first=True,
        )
        # self.conv1 = GCNConv(self.in_c, self.hid_c)
        # self.conv2 = GCNConv(self.hid_c, self.out_c)
        self.fc = nn.Linear(self_model_args["lstm"]["hid"][1], self.y_days)

        self.social_recovery_lambda = nn.Parameter(
            torch.empty(self.num_zones), requires_grad=True
        )
        nn.init.normal_(self.social_recovery_lambda.data, mean=0.05, std=0.0)
        
    def forward(self, X, A, extra_info=None, idx=None):
        batch_size = X.size(0)

        A = A.float()
        extra_info = extra_info.float()
        X = X.float()
        idx = idx.float()

        lstm_input = []
        adj_output = []
        for day in range(X.size(1)):
            x = X[:, day, :, :]
            adj = A[:, day, :, :]
            x_text = extra_info[:, day, :, :]

            if hasattr(self, 'graph_learner'):
                adj_hat = self.graph_learner(x)
                z = self.graph_encoder(x, adj_hat)
                adj_output.append(adj_hat.unsqueeze(1))
            else:
                z = self.graph_encoder(x, adj)

            if self.train_with_text:
                z_text = self.text_fc(x_text)
                # z_text = self.text_graph_encoder(x_text, adj)
                # social_recovery = []
                # for i in idx:
                #     _social_recovery = torch.exp(i * self.social_recovery_lambda ** 2).unsqueeze(1).repeat(1, z_text.size(-1)).unsqueeze(0)
                #     social_recovery.append(_social_recovery)
                # social_recovery = torch.cat(social_recovery, 0)
                social_recovery = (
                    torch.exp(idx.unsqueeze(-1) * self.social_recovery_lambda**2)
                    .unsqueeze(-1)
                    .repeat(1, 1, z_text.size(-1))
                )
                z_text = z_text.mul(social_recovery)
                z = torch.cat([z, z_text], 2)
            lstm_input.append(z.unsqueeze(1))

        lstm_input = torch.cat(lstm_input, 1)
        if hasattr(self, 'graph_learner'): adj_output = torch.cat(adj_output, 1)

        lstm_input = lstm_input.transpose(1, 2).flatten(0, 1)
        lstm_out, (hc1, cn1) = self.lstm(lstm_input)

        lstm_out = lstm_out[:, -1, :].unsqueeze(-2)
        lstm_out2, (hc2, cn2) = self.lstm2(lstm_out)

        lstm_out2 = lstm_out2[:, -1, :]

        out = self.fc(lstm_out2)

        out = out.view(batch_size, self.y_days, self.num_zones)
        
        if hasattr(self, 'graph_learner'): return out, adj_output
        else: return out
