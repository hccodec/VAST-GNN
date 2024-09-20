# -*- encoding: utf-8 -*-
"""
@File    :   train.py
@Time    :   2024/04/12 16:30:37
@Author  :   HCCODEC
"""

import torch, random
from torch import nn
import time, copy, numpy as np
from math import *
from model.net import *


def train_new():
    x_day, y_day = 21, 7
    dropout, alpha = 0.5, 0.2
    num_epochs, batch_size, learning_rate = 100, 8, 1e-3
    hidden_dim_1, hidden_dim_2, out_dim_1 = 6, 3, 4
    ratio = {
        'train': 0.7, 'val': 0.1,
        'inf_norm': 100., 'web_norm': 100.
        }
    


def generate_known_links(adjacency_matrix):
    num_nodes = len(adjacency_matrix)
    known_links = []

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adjacency_matrix[i][j]:
                known_links.append((i, j))

    return known_links


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    adj: torch.nn.Module,
    features: torch.nn.Module,
    y: torch.nn.Module,
    criterion=F.mse_loss,
):
    optimizer.zero_grad()
    output = model(adj, features)
    loss_train = criterion(output, y)
    loss_train.backward(retain_graph=True)
    optimizer.step()
    return output, loss_train


def test(
    model: torch.nn.Module,
    adj: torch.Tensor,
    features: torch.Tensor,
    y: torch.Tensor,
    num_nodes: int,
    criterion=F.mse_loss,
):
    """
    Test the model

    Parameters:
    model (torch.nn.Module): Model to test
    adj (torch.Tensor): Adjacency matrix
    features (torch.Tensor): Features matrix
    y (torch.Tensor): Labels matrix

    Returns:
    output (torch.Tensor): Output predictions of the model
    loss_test (torch.Tensor): Loss of the model
    """
    output = model(adj, features)
    loss_test = criterion(output, y)
    loss_test = float(loss_test.detach().cpu().numpy())

    o = output.cpu().detach().numpy()
    l = y.cpu().numpy()
    err = np.sum(abs(o - l)) / num_nodes

    return output, loss_test, err


def run_tgnn_model(
    graphs: list,
    features: list,
    y: list,
    idx_train: list,
    idx_val: list,
    graph_window: int,
    shift: int,
    batch_size: int,
    device: torch.device,
    test_sample: int,
    results_dir,
    country,
):

    early_stop = 100

    import os

    best_model_path = os.path.join(results_dir, f"model_best_{country}.pth.tar")
    results_csv_path = os.path.join(results_dir, "results_" + country + ".csv")

    from preprocess_tgnn import (
        generate_batches,
        analyze_generated_batches,
        judge_batches,
        AverageMeter,
    )

    adj_train, features_train, y_train = generate_batches(
        graphs,
        features,
        y,
        idx_train,
        graph_window,
        shift,
        batch_size,
        device,
        test_sample,
    )
    adj_val, features_val, y_val = generate_batches(
        graphs,
        features,
        y,
        idx_val,
        graph_window,
        shift,
        batch_size,
        device,
        test_sample,
    )
    adj_test, features_test, y_test = generate_batches(
        graphs, features, y, [test_sample], graph_window, shift, batch_size, device, -1
    )

    ############################################# 定义模型  START
    num_nodes, input_size = (
        features_train[0].shape[0] // (len(idx_train) * graph_window),
        features_train[0].shape[1],
    )
    hidden_size, num_layers, num_heads = 32, 2, 4
    model = MPNN_LSTM(
        nfeat=7, nhid=64, nout=1, n_nodes=num_nodes, window=graph_window, dropout=0.5
    ).cuda()
    # model = TemporalGraphLearner(
    #     num_nodes=num_nodes,
    #     in_channels=input_size,
    #     hidden_channels=hidden_size,
    #     out_channels=1,
    #     num_layers=num_layers,
    #     seq_len=graph_window,
    # )
    # model = GNNLSTM(input_size, hidden_size, num_layers, num_nodes, num_heads)
    criterion = F.mse_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
    ############################################# 定义模型  END

    n_train_batches = ceil(len(idx_train) / batch_size)

    best_val_acc = 1e8
    val_among_epochs, train_among_epochs, err_among_epochs = [], [], []

    model = model.cuda()

    with open(results_csv_path, "w") as fw:

        fw.write(f"Epoch,train_loss,val_loss,val_err,time\n")
        try:
            for epoch in range(10000):
                start = time.time()
                train_loss = AverageMeter()
                val_loss = AverageMeter()
                model.train()
                for batch in range(n_train_batches):
                    adj, features, y = (
                        adj_train[batch],
                        features_train[batch],
                        y_train[batch],
                    )
                    output, loss = train(model, optimizer, adj, features, y, criterion)
                    train_loss.update(loss.data.item(), output.size(0))

                model.eval()
                output, val_loss, val_err = test(
                    model, adj_val[0], features_val[0], y_val[0], num_nodes
                )

                msg = (
                    f"Epoch: {epoch + 1:3d}, train_loss = {train_loss.avg:.5f}, "
                    + f"val_loss = {val_loss:.5f}, val_err = {val_err:.5f}, time = {time.time() - start:.5f}"
                )
                fw.write(
                    f"{epoch + 1},{train_loss.avg:.5f},{val_loss:.5f},{val_err:.5f},{time.time() - start:.5f}\n"
                )
                if not epoch % 50:
                    print(msg)

                train_among_epochs.append(train_loss.avg)
                val_among_epochs.append(val_loss)
                err_among_epochs.append(val_err)

                # if(len(set([round(val_e) for val_e in val_among_epochs[-int(early_stop/2):]])) == 1):#
                lis = [
                    round(val_e, 1)
                    for val_e in val_among_epochs[-int(early_stop / 2) :]
                ]
                if len(lis) > 1 and len(set(lis)) == 1:  #
                    print("Break early stop at epoch", epoch)
                    # stop = True
                    break

                # if (
                #     len(train_among_epochs) > 1
                #     and abs(train_among_epochs[-1] - train_loss.avg) < 1e-3
                # ):
                #     print("Little improvement by further training, early stop.")
                #     break

                if val_loss < best_val_acc:
                    import os

                    best_val_acc = val_loss
                    torch.save(
                        {
                            "state_dict": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                        },
                        best_model_path,
                    )

                scheduler.step(val_loss)
        except KeyboardInterrupt:
            print("[KeyboardInterrupt, Stop this epoch loop.]")

        if os.path.exists(best_model_path):
            # Test
            checkpoint = torch.load(best_model_path)
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            model.eval()
            output, loss, err = test(
                model, adj_test[0], features_test[0], y_test[0], num_nodes
            )
            print(f"Testing result: err = {err} (with loss {loss:.5f})")
        else:
            print("No trained model. Skip testing.")
            err = 1e8

    return train_among_epochs, val_among_epochs, err_among_epochs, best_model_path, err


def run_tokyo_model(train_x_y, vali_data, test_data, device, results_dir, args):

    r = random.random

    import os

    best_model_path = os.path.join(results_dir, f"model_best_jp.pth.tar")
    results_csv_path = os.path.join(results_dir, "results_jp.csv")


    def normalize_mobility_column(x_batch):
        # x_batch：(n_batch, 0/1, 2*i+1)
        x_batch_new = copy.copy(x_batch)
        n_batch = len(x_batch_new)
        # days = round(len(x_batch[0][0]) / 2)
        days = len(x_batch_new[0][2])
        for i in range(n_batch):
            for j in range(days):

                x_batch_new[i][0][2 * j] /= np.sum(
                    x_batch_new[i][0][2 * j], axis=0
                )  # 每一列的和都为 1

        return x_batch_new

    def validate_test_process(trained_model, vali_test_data):
        trained_model.eval()
        criterion = nn.MSELoss()
        vali_test_y = [vali_test_data[i][1] for i in range(len(vali_test_data))]
        y_real = torch.tensor(np.array(vali_test_y)).to(device)
        vali_test_x = [vali_test_data[i] for i in range(len(vali_test_data))]
        vali_test_x = normalize_mobility_column(vali_test_x)
        _, y_hat = trained_model.run_specGCN_lstm(vali_test_x)  ###Attention
        loss = criterion(y_hat.float(), y_real.float())
        return loss, y_hat, y_real
        
    def adjust_lambda(args, epoch, consider_start_epoch=True):
        epoch_start = args.graph_learner_decay_start_epoch
        k = args.graph_learner_decay_k  # 衰减速率

        lambda_0 = args.graph_learner_decay_start_lambda
        y0 = lambda_0 #  f(epoch_start)

        if consider_start_epoch:
            if epoch <= epoch_start: return lambda_0
            else:
                y0 = adjust_lambda(args, epoch_start, False)

        if args.graph_learner_decay == 'linear':
            res = 1 + (k - 1) / args.epochs * epoch
        elif args.graph_learner_decay == 'step':
            res = 1 - k * 10 * (epoch // (args.epochs * k * 10))
        elif args.graph_learner_decay == 'square':
            res = 1 + (k - 1) * (epoch / args.epochs) ** 2
        elif args.graph_learner_decay == 'exp':
            res = 2 - np.exp(np.log(2 - k) * (epoch / args.epochs))
        return lambda_0 * res * (1 - k) / (y0 - k)

    def train_epoch_option(
        model: MULTIWAVE_SpecGCN_LSTM, opt, criterion, trainX_c, trainY_c, batch_size, epoch, reconstruct_graph=False
    ):
        model.train()
        losses = []
        batch_num = 0
        for beg_i in range(0, len(trainX_c), batch_size):
            batch_num += 1
            # if batch_num % 16 == 0:
            #     print(
            #         ", batch_num: {}/{}".format(
            #             batch_num, int(len(trainX_c) / batch_size)
            #         )
            #     )
                # print ("batch_num: ", batch_num, "total batch number: ", int(len(trainX_c)/batch_size))
            x_batch = trainX_c[beg_i : beg_i + batch_size]
            y_batch = trainY_c[beg_i : beg_i + batch_size]
            y_batch = torch.tensor(np.array(y_batch), device=device)
            opt.zero_grad()
            x_batch = normalize_mobility_column(
                x_batch
            )  # conduct the column normalization
            adj_hat, y_hat = model.run_specGCN_lstm(x_batch, reconstruct_graph)  ###Attention
            adj_batch = torch.tensor(np.array(
                    [[_t for i, _t in enumerate(t[0]) if i % 2 == 0] for t in x_batch]
                    )).to(device)
            
            if reconstruct_graph:
                lambda_t = adjust_lambda(args, epoch)
                loss = criterion(adj_hat.float(), adj_batch.float()) + \
                    lambda_t * criterion(y_hat.float(), y_batch.float())  # MSE loss
            else:
                loss = criterion(y_hat.float(), y_batch.float())  # MSE loss
            # opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.data.cpu().numpy())
        return sum(losses) / float(len(losses)), model

    def train_process(
        train_data, lr, num_epochs, net, criterion, bs, vali_data, test_data, reconstruct_graph
    ):
        opt = torch.optim.Adam(net.parameters(), lr, betas=(0.9, 0.999), weight_decay=0)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.95)

        train_y = [train_data[i][1] for i in range(len(train_data))]
        e_losses, e_losses_vali, e_losses_test = [], [], []
        time00 = time.time()

        with open(results_csv_path, "w") as fw: pass
        with open(results_csv_path, "a") as fw: fw.write(f"Epoch,train_loss,val_loss,rmse_validate,rmse_test,mae_validate,mae_test,test_loss,time\n")
        loss_best = 1e7
        for e in range(num_epochs):

            time1 = time.time()
            print("current epoch: {}/{}".format(e, num_epochs), end=", ")
            # print ("current epoch: ",e, "total epoch: ", num_epochs)
            random_index_train = list(range(len(train_data)))
            random.shuffle(random_index_train, random=r)
            trainX_sample = [
                train_data[random_index_train[j]]
                for j in range(len(random_index_train))
            ]
            trainY_sample = [
                train_y[random_index_train[j]] for j in range(len(random_index_train))
            ]
            loss, net = train_epoch_option(
                net, opt, criterion, trainX_sample, trainY_sample, bs, e, reconstruct_graph
            )
            
            # scheduler.step()

            loss *= infection_normalize_ratio * infection_normalize_ratio
            print("Loss (train/validate/test) : ", end="")
            print("{}".format(loss),end=", ",)
            e_losses.append(loss)

            loss_vali, y_hat_vali, y_real_vali = validate_test_process(net, vali_data)
            loss_test, y_hat_test, y_real_test = validate_test_process(net, test_data)

            loss_vali = float(loss_vali) * infection_normalize_ratio * infection_normalize_ratio
            loss_test = float(loss_test) * infection_normalize_ratio * infection_normalize_ratio

            e_losses_vali.append(loss_vali)
            e_losses_test.append(loss_test)

            print("{}".format(loss_vali),end=", ",)
            print("{}".format(loss_test))
            # if e>=2 and (e+1)%10 ==0:
            #     visual_loss(e_losses, e_losses_vali, e_losses_test)
            #     visual_loss_train(e_losses)
            time2 = time.time()

            print("time: {}s".format(time2 - time1), end='')
            
            if loss_test < loss_best:
                loss_best = loss_test
                print(' (Best model saved)', end='')
                torch.save(net, best_model_path)

            
            from vali_test import compute
            rmse_validate, rmse_test, mae_validate, mae_test = compute(
                vali_data, y_hat_vali, y_real_vali,
                test_data, y_hat_test, y_real_test,
                infection_normalize_ratio
                )

            # fw.write(f"Epoch,train_loss,val_loss,val_err,test_loss,time\n")
            with open(results_csv_path, "a") as fw:
                fw.write("{},{:.5f},{:.5f},{},{},{},{},{:.5f},{:.2f}s\n".format(
                    e, loss, loss_vali,
                    round(np.mean(rmse_validate), 3), round(np.mean(rmse_test), 3), round(np.mean(mae_validate), 3), round(np.mean(mae_test), 3),
                    loss_test, time2 - time1
                ))
            print("\n---------------------------------------------------------------")

        return e_losses, net

    X_day, Y_day = 21, 7
    # hyperparameter for the learning
    DROPOUT, ALPHA = 0.50, 0.20
    NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE = args.epochs, 8, 0.0001
    HIDDEN_DIM_1, OUT_DIM_1, HIDDEN_DIM_2 = 6, 4, 3
    infection_normalize_ratio = 100.0
    web_search_normalize_ratio = 100.0
    train_ratio = 0.7
    validate_ratio = 0.1

    # 3.2.1 define the model
    input_dim_1, hidden_dim_1, out_dim_1, hidden_dim_2 = (
        len(train_x_y[0][0][1][1]),
        HIDDEN_DIM_1,
        OUT_DIM_1,
        HIDDEN_DIM_2,
    )
    dropout_1, alpha_1, N = DROPOUT, ALPHA, len(train_x_y[0][0][1])
    # G_L_Model = MyModel(
    #     X_day,
    #     Y_day,
    #     input_dim_1,
    #     hidden_dim_1,
    #     out_dim_1,
    #     hidden_dim_2,
    #     dropout_1,
    #     N,
    #     device,
    # ).to(
    #     device
    # )  ###Attention
    G_L_Model = MULTIWAVE_SpecGCN_LSTM(X_day, Y_day, input_dim_1, hidden_dim_1, out_dim_1, hidden_dim_2, dropout_1, N, device).to(device)         ###Attention
    # 3.2.2 train the model
    num_epochs, batch_size, learning_rate = (
        NUM_EPOCHS,
        BATCH_SIZE,
        LEARNING_RATE,
    )  # model train
    criterion = nn.MSELoss()
    e_losses, trained_model = train_process(
        train_x_y,
        learning_rate,
        num_epochs,
        G_L_Model,
        criterion,
        batch_size,
        vali_data,
        test_data,
        args.use_graph_learner
    )

    validation_result, validate_hat, validate_real = validate_test_process(trained_model, vali_data)
    test_result, test_hat, test_real = validate_test_process(trained_model, test_data)

    from vali_test import compute
    compute(
        vali_data, validate_hat, validate_real,
        test_data, test_hat, test_real,
        infection_normalize_ratio
        )


    return e_losses, trained_model

if __name__ == "__main__":
    raise NotImplementedError("暂不支持直接运行该文件，可通过 main.py 间接执行")
    # train()
