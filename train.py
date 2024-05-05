# -*- encoding: utf-8 -*-
'''
@File    :   train.py
@Time    :   2024/04/12 16:30:37
@Author  :   HCCODEC
'''

import torch, random
from torch import nn
import time, copy, numpy as np
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

def run_tgnn_model(graphs: list,
                features: list,
                y: list,
                idx_train: list,
                idx_val: list,
                graph_window: int,
                shift: int,
                batch_size: int,
                device: torch.device,
                test_sample: int, fw):
    
    from preprocess_tgnn import generate_new_batches, AverageMeter

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

def run_tokyo_model(train_x_y, vali_data, test_data, device):

    r = random.random

    def normalize_column_one(input_matrix):
        column_sum = np.sum(input_matrix, axis=0)
        row_num, column_num = len(input_matrix), len(input_matrix[0])
        for i in range(row_num):
            for j in range(column_num):
                input_matrix[i][j] = input_matrix[i][j]*1.0/column_sum[j]
        return input_matrix

    def convertAdj(x_batch):
        #x_batch：(n_batch, 0/1, 2*i+1)
        x_batch_new = copy.copy(x_batch)
        n_batch = len(x_batch)
        days = round(len(x_batch[0][0])/2)
        for i in range(n_batch):
            for j in range(days):
                mobility_matrix = x_batch[i][0][2*j]
                x_batch_new[i][0][2*j] = normalize_column_one(mobility_matrix)   #20210818
        return x_batch_new

    def validate_test_process(trained_model, vali_test_data):
        criterion = nn.MSELoss()
        vali_test_y = [vali_test_data[i][1] for i in range(len(vali_test_data))]
        y_real = torch.tensor(vali_test_y)
        vali_test_x = [vali_test_data[i] for i in range(len(vali_test_data))]
        vali_test_x = convertAdj(vali_test_x)
        y_hat = trained_model.run_specGCN_lstm(vali_test_x)                                  ###Attention              
        loss = criterion(y_hat.float(), y_real.float())
        return loss, y_hat, y_real 
    
    def train_epoch_option(model, opt, criterion, trainX_c, trainY_c, batch_size):  
        model.train()
        losses = []
        batch_num = 0
        for beg_i in range(0, len(trainX_c), batch_size):
            batch_num += 1
            if batch_num % 16 ==0:
                print ("batch_num: ", batch_num, "total batch number: ", int(len(trainX_c)/batch_size))
            x_batch = trainX_c[beg_i:beg_i+batch_size]        
            y_batch = torch.tensor(trainY_c[beg_i:beg_i+batch_size])   
            opt.zero_grad()
            x_batch = convertAdj(x_batch)   #conduct the column normalization
            y_hat = model.run_specGCN_lstm(x_batch)                          ###Attention
            loss = criterion(y_hat.float(), y_batch.float()) #MSE loss
            #opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.data.numpy())
        return sum(losses)/float(len(losses)), model
    
    def train_process(train_data, lr, num_epochs, net, criterion, bs, vali_data, test_data):
        opt = torch.optim.Adam(net.parameters(), lr, betas = (0.9,0.999), weight_decay=0) 
        train_y = [train_data[i][1] for i in range(len(train_data))]
        e_losses = list()
        e_losses_vali = list()
        e_losses_test = list()
        time00 = time.time()
        for e in range(num_epochs):
            time1 = time.time()
            print ("current epoch: ",e, "total epoch: ", num_epochs)
            number_list = list(range(len(train_data)))       
            random.shuffle(number_list, random = r)
            trainX_sample = [train_data[number_list[j]] for j in range(len(number_list))]
            trainY_sample = [train_y[number_list[j]] for j in range(len(number_list))]
            loss, net =  train_epoch_option(net, opt, criterion, trainX_sample, trainY_sample, bs)  
            print ("train loss", loss*infection_normalize_ratio*infection_normalize_ratio)
            e_losses.append(loss*infection_normalize_ratio*infection_normalize_ratio)
            
            loss_vali, y_hat_vali, y_real_vali = validate_test_process(net, vali_data) 
            loss_test, y_hat_test, y_real_test = validate_test_process(net, test_data)
            e_losses_vali.append(float(loss_vali)*infection_normalize_ratio*infection_normalize_ratio)
            e_losses_test.append(float(loss_test)*infection_normalize_ratio*infection_normalize_ratio)
            
            print ("validate loss", float(loss_vali)*infection_normalize_ratio*infection_normalize_ratio)
            print ("test loss", float(loss_test)*infection_normalize_ratio*infection_normalize_ratio)
            # if e>=2 and (e+1)%10 ==0:
            #     visual_loss(e_losses, e_losses_vali, e_losses_test)     
            #     visual_loss_train(e_losses) 
            time2 = time.time()
            print ("running time for this epoch:", time2 - time1)
            time01 = time.time()
            print ("---------------------------------------------------------------")
            print ("---------------------------------------------------------------")
        return e_losses, net

    X_day, Y_day = 21,21
    #hyperparameter for the learning
    DROPOUT, ALPHA = 0.50, 0.20
    NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE = 100, 8, 0.0001
    HIDDEN_DIM_1, OUT_DIM_1, HIDDEN_DIM_2 = 6,4,3
    infection_normalize_ratio = 100.0
    web_search_normalize_ratio = 100.0
    train_ratio = 0.7
    validate_ratio = 0.1

    #3.2.1 define the model
    input_dim_1, hidden_dim_1, out_dim_1, hidden_dim_2 = len(train_x_y[0][0][1][1]),    HIDDEN_DIM_1, OUT_DIM_1, HIDDEN_DIM_2 
    dropout_1, alpha_1, N = DROPOUT, ALPHA, len(train_x_y[0][0][1])
    G_L_Model = MULTIWAVE_SpecGCN_LSTM(X_day, Y_day, input_dim_1, hidden_dim_1, out_dim_1, hidden_dim_2, dropout_1,N, device)         ###Attention
    #3.2.2 train the model
    num_epochs, batch_size, learning_rate = NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE                                                 #model train
    criterion = nn.MSELoss() 
    e_losses, trained_model = train_process(train_x_y, learning_rate, num_epochs, G_L_Model, criterion, batch_size,                          vali_data, test_data)
    return e_losses, trained_model

def train():
    pass

if __name__ == "__main__":
    train()