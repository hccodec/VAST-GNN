import torch, math
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv
import torch.nn.functional as F
from torch_geometric.data import Data
from torch.utils.data import DataLoader

class GraphEncoder(nn.Module):
    def __init__(self, num_nodes, input_dim, hidden_dim, num_heads):
        super(GraphEncoder, self).__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.conv1 = GATConv(input_dim, hidden_dim, heads=num_heads, dropout=0.6)
        # self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=0.6)

        self.linear = nn.Linear(hidden_dim * num_heads, input_dim)
    
    def forward(self, x, edge_index):

        weight = edge_index.coalesce().values()
        edge_index = edge_index.coalesce().indices()

        x = F.dropout(x, p=0.6, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        # x = self.conv2(x, edge_index)

        x = self.linear(x)

        return x
    
class GraphReconstruction(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphReconstruction, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.fc = nn.Linear(input_dim, output_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = F.relu(self.fc(x))
        return x

#######################################################################

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
    try: target_links = target_links.to_dense()
    except: pass
    
    graph_loss = criterion(pred_graph.squeeze(0), target_graph.mean(dim=1))
    link_loss = criterion(pred_links, target_links)

    total_loss = graph_loss_weight * graph_loss + \
        (1 - graph_loss_weight) * link_loss
    
    return total_loss

#######################################################################

# 定义 GNN+LSTM 模型
class GNNLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_nodes, num_heads):
        super(GNNLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_nodes = num_nodes
        self.num_heads = num_heads

        self.gnn = GCNConv(input_size, hidden_size, heads=num_heads)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_nodes * num_nodes)
    
    def forward(self, x, edge_index, batch):

        weight = edge_index.coalesce().values()
        edge_index = edge_index.coalesce().indices()

        x = self.gnn(x, edge_index, weight)
        x = torch.split(x, self.num_nodes, dim=0)
        x = torch.stack(x, dim=00)
        lstm_out, _ = self.lstm(x)
        lstm_out_last = lstm_out[:, -1, :]
        output = self.fc(lstm_out_last)
        n = int(math.sqrt(output.shape[1]))
        return output.reshape(n, -1)
    
 
class SpecGCNLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.5, concat=True):
        super(SpecGCNLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.concat = concat
        
        self.W = nn.Parameter(torch.empty(size=(input_dim, hidden_dim)))
        self.W.requires_grad = True
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
    
    def forward(self, feature, D_n_A_D_n):
        #feature.shape: (N,input_dim)
        feature_new = torch.mm(feature.float(), self.W)
        feature_new  = F.dropout(feature_new, self.dropout, training=self.training)
        H = torch.mm(D_n_A_D_n.float(), feature_new)
        return H
class SpecGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, dropout=0.5):
        super(SpecGCN, self).__init__()
        self.dropout = dropout
        #self.layer1 = SpecGCNLayer(input_dim, hidden_dim, dropout=dropout, concat=True)
        #self.out_feature = SpecGCNLayer(hidden_dim, out_dim, dropout=dropout, concat=True)
        self.layer1 = SpecGCNLayer(input_dim, out_dim, dropout=dropout, concat=True)
        #self.layer2 = SpecGCNLayer(out_dim, out_dim, dropout=dropout, concat=True)
        
    def compute_D_n_A_D_n(self, adjs):
        N =  adjs.size()[0]   
        tilde_A = adjs + torch.eye(N).cuda()
        tilde_D_n = torch.diag(torch.pow(tilde_A.sum(-1).float(), -0.5))
        D_n_A_D_n = torch.mm(tilde_D_n, torch.mm(tilde_A, tilde_D_n))
        return D_n_A_D_n 
    
    def forward(self, x, adjs):
        D_n_A_D_n = self.compute_D_n_A_D_n(adjs)
        #x1 = self.layer1(x, D_n_A_D_n)
        #x1 = F.dropout(x1, self.dropout, training=self.training)
        #x2 = self.out_feature(x1, D_n_A_D_n)
        #x_out = F.relu(x2)   
        x1 = self.layer1(x, D_n_A_D_n)
        x1 = F.dropout(x1, self.dropout, training=self.training)
        #x1 = self.layer2(x1, D_n_A_D_n)
        #x1 = F.dropout(x1, self.dropout, training=self.training)
        x_out = F.relu(x1)
        return x_out
    
class MULTIWAVE_SpecGCN_LSTM(nn.Module):
    def __init__(self, x_days, y_days, input_dim_1, hidden_dim_1, out_dim_1, hidden_dim_2, dropout_1, N, device='cpu'):
        super (MULTIWAVE_SpecGCN_LSTM, self).__init__()
        self.device = device
        self.N = N
        self.x_days = x_days
        self.y_days = y_days
        self.specGCN = SpecGCN(input_dim_1, hidden_dim_1, out_dim_1, dropout_1).to(device)
        
        self.lstm = nn.LSTM(batch_first=True, input_size=out_dim_1+1, hidden_size=hidden_dim_2,num_layers=2, bidirectional=False)
        #self.lstm = nn.LSTM(batch_first=True, input_size=N*out_dim_1+N, hidden_size=N*hidden_dim_2,num_layers=1, bidirectional=False)     
        self.out_dim_1 = out_dim_1             #out dimension of GNN
        self.hidden_dim_2 = hidden_dim_2       #out dimension of LSTM
        self.fc1 = nn.Linear(self.hidden_dim_2 + self.x_days, 1)
        self.v = torch.nn.Parameter(torch.empty(23))
        self.v.requires_grad = True
        torch.nn.init.normal_(self.v.data, mean=0.05, std=0.000)
        
    def run_specGCN_lstm(self, input_record):
        N = self.N
        n_batch = len(input_record)
        x_days, y_days = round(len(input_record[0][0])/2), len(input_record[0][1])
        
        #Step1: calculate the SpecGCN output        
        #lstm_input_batch = torch.zeros((N, n_batch, self.x_days, self.out_dim_1+1))  
        lstm_input_batch = torch.zeros((N, n_batch, self.x_days, self.out_dim_1+1), device=self.device)
        for batch in range(n_batch):
            #mobility
            #torch.tensor(input_record[batch][0][2*i]).float()     #(N,N)
            #text
            #torch.tensor(input_record[batch][0][2*i+1]).float()   #(N,text_dimension)
            #infection
            #torch.tensor(input_record[batch][2][i]).float()       #N
            for i in range(x_days):
                x  = torch.tensor(input_record[batch][0][2*i+1]).float()   #(N, text_dimension)
                adj = torch.tensor(input_record[batch][0][2*i]).float()    #(N, N)
                x_infection = torch.tensor(input_record[batch][2][i]).float() #N
                x_infection = x_infection.reshape((x_infection.size()[0],1)) #(N, 1)
                day_order =  input_record[batch][3]                           #v6
                x = x.to(self.device)
                adj = adj.to(self.device)
                x_infection = x_infection.to(self.device)
                specGCN_out = self.specGCN(x, adj)                        #(N, out_dim1)
                specGCN_out = specGCN_out.mul(torch.unsqueeze(torch.exp(self.v*self.v*float(day_order)),dim=1).repeat(1,self.out_dim_1))
                
                specGCN_out = torch.cat([specGCN_out, x_infection], dim=1) #(N, out_dim1+1) 
                for j in range(N):
                    lstm_input_batch[j][batch][i] = specGCN_out[j]    
        #lstm_input_batch.shape = [N, batch, x_days, out_dim_1+1]
        
        #Step2: calculate the LSTM outputs
        y_output_batch = torch.zeros((n_batch, self.y_days, self.N))
        for j in range(N):
            lstm_input_x1 = lstm_input_batch[j]                   #[batch, x_days, out_dim_1+1]
            lstm_input_x2 = torch.mean(lstm_input_x1, dim=1)
            lstm_input_x2 = lstm_input_x2.view((lstm_input_x2.size()[0], 1, lstm_input_x2.size()[1])) ##[batch, 1, out_dim_1+1]
            lstm_input_x2 = lstm_input_x2.repeat(1, y_days, 1)  ##[batch, y_days, out_dim_1+1]
            
            lstm_input_x1_output, (hc,cn) = self.lstm(lstm_input_x1)          #[batch, x_days, hidden_dim_2]
            lstm_output, (hc1,cn1) = self.lstm(lstm_input_x2, (hc,cn))        #[batch, y_days, hidden_dim_2]
            
            infection_tensor = torch.tensor([[input_record[batch][2][x_day_c][j] for x_day_c in range(x_days)] for batch in range(n_batch)]).float().to(self.device)
            infection_tensor = infection_tensor.view((infection_tensor.size()[0], 1, infection_tensor.size()[1]))
            infection_tensor = infection_tensor.repeat(1, y_days, 1)  #[batch, y_days, x_days]
            
            lstm_output = torch.cat([lstm_output, infection_tensor], dim=2)
            
            
            fc_output = self.fc1(lstm_output)   #[batch, y_days, 1]
            fc_output = F.relu(fc_output)    #[batch, y_days, 1]
            if j == 0:
                y_output_batch = fc_output
            else:
                y_output_batch = torch.cat([y_output_batch, fc_output], dim=2)
        return y_output_batch      #y_output_batch.shape = (batch, y_day, N)
