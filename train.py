# -*- encoding: utf-8 -*-
'''
@File    :   train.py
@Time    :   2024/04/12 16:30:37
@Author  :   HCCODEC
'''

import torch
from torch import nn
from preprocess_tgnn import generate_new_batches, AverageMeter
import time
from math import *
from model.net import *

def generate_known_links(adjacency_matrix):
    num_nodes = len(adjacency_matrix)
    known_links = []

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adjacency_matrix[i][j]:
                known_links.append((i, j))

    return known_links

def run_model(graphs: list,
                features: list,
                y: list,
                idx_train: list,
                idx_val: list,
                graph_window: int,
                shift: int,
                batch_size: int,
                device: torch.device,
                test_sample: int, fw):
    adj_train, features_train, y_train = generate_new_batches(
        graphs, features, y, idx_train, graph_window, shift, batch_size, device, test_sample)
    adj_val, features_val, y_val = generate_new_batches(graphs, features, y, idx_val, 1, shift, batch_size, device, test_sample)
    adj_test, features_test, y_test = generate_new_batches(graphs, features, y, [test_sample], 1, shift, batch_size, device, -1)
    
    ############################################# 定义模型  START
    # model = nn.Sequential(
    #     nn.Linear(1, 128), nn.ReLU(),
    #     nn.Linear(128, 256), nn.ReLU(),
    #     nn.Linear(256, 4))

    # model = GNNLSTM(7, 64, 1).cuda()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)

    # num_nodes = adj_train[0].shape[0]
    # gnn_model = GNN(input_dim=num_nodes, hidden_dim=64, output_dim=num_nodes).cuda()
    # lstm_model = LSTM(input_dim=num_nodes, hidden_dim=64, output_dim=1).cuda()
    # gnn_optimizer = torch.optim.Adam(gnn_model.parameters(), lr=1e-3)
    # lstm_optimizer = torch.optim.Adam(lstm_model.parameters(), lr=1e-3)

    # num_nodes, input_dim = features_train[0].shape
    # hidden_dim, num_heads = 64, 4
    # encoder = GraphEncoder(num_nodes, input_dim, hidden_dim, num_heads).cuda()
    # decoder = GraphReconstruction(input_dim, hidden_dim, num_nodes).cuda()
    # # decoder = GraphReconstruction(hidden_dim * num_heads, hidden_dim, num_nodes).cuda()
    # encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    # decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)
    # criterion_encoder = nn.MSELoss()
    # criterion_decoder = nn.MSELoss()

    num_nodes, input_size = features_train[0].shape
    hidden_size, num_layers, num_heads = 32, 2, 4
    model = GNNLSTM(input_size, hidden_size, num_layers, num_nodes, num_heads)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    ############################################# 定义模型 FINISH
    
    n_train_batches = ceil(len(idx_train)/batch_size)

    val_among_epochs, train_among_epochs = [], []

    model = model.cuda()


    for epoch in range(10000):
        start = time.time()
        
        model.train()

        train_loss = AverageMeter()

        for batch in range(n_train_batches):
            optimizer.zero_grad()
            
            x, edge_index = features_train[batch], adj_train[batch]            

            output = model(x, edge_index, batch)

            edge_index = edge_index.to_dense()
            loss = criterion(output, edge_index.reshape(output.shape))

            # loss = torch.nn.functional.mse_loss(
            #     output, y_train[batch].reshape(output.shape)
            #     )

            # graph_representation = gnn_model(adj_train[batch], features_train[batch])
            # predicted_links = lstm_model(graph_representation)

            # loss = compute_loss(
            #     graph_representation, adj_train[batch],
            #     predicted_links, adj_train[batch])


            # encoded_features = encoder(x, edge_index)
            # reconstructed_features = decoder(encoded_features)

            # edge_index = edge_index.to_dense()

            # loss = compute_loss(
            #                 encoded_features, adj_train[batch],
            #                 reconstructed_features, adj_train[batch])

            
            loss.backward()

            optimizer.step()

            train_loss.update(loss.data.item(), output.size(0))
        
        model.eval()


        x, edge_index = features_val[batch], adj_val[batch]            

        output = model(x, edge_index, batch)
        val_loss = criterion(output, edge_index)
    
        # graph_representation = gnn_model(adj_val[0], features_val[0])
        # predicted_links = lstm_model(graph_representation)
        # # 创建目标链路张量，1表示链路存在，0表示链路不存在
        # target_links = generate_known_links(adj_train[batch])
        # val_loss = compute_loss(
        #     graph_representation, adj_train[batch],
        #     predicted_links, target_links)
        # val_loss = float(val_loss.cpu().numpy())

        # output = model(adj_val[0], features_val[0])
        # val_loss = torch.nn.functional.mse_loss(
        #     output, y_val[0].reshape(output.shape)
        #     ).detach().cpu().numpy()
        # val_loss = float(val_loss)

        msg = f"Epoch: {epoch + 1:3d}, train_loss = {train_loss.avg:.5f}, val_loss = {val_loss:.5f}, time = {time.time() - start:.5f}"
        fw.write(f"{epoch + 1},{train_loss.avg:.5f},{val_loss:.5f},{time.time() - start:.5f}\n")
        if not epoch % 50:
            print(msg)

        if len(train_among_epochs) and abs(train_among_epochs[-1] - train_loss.avg) < 1e-3:
            print('Little improvement by further training, early stop.')
            break

        train_among_epochs.append(train_loss.avg)    
        val_among_epochs.append(val_loss)

    return train_among_epochs, val_among_epochs, model

def train():
    pass

if __name__ == "__main__":
    train()