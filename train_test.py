import torch, time, os
from torch import nn
import numpy as np
from eval import compute_err, compute_mae_rmse, metrics_labels, compute_correlation, compute_hits_at_k

from utils.logger import file_logger, logger

# from utils.tensorboard import writer
from utils.utils import adjust_lambda, font_underlined, catch, font_green, font_yellow, min_max_adj, random_mask, rm_self_loops
from models.dynst import dynst_extra_info


# logger = logger.getLogger()


@catch()
def train_process(
    model,
    criterion,
    epochs,
    lr,
    lr_min,
    lr_scheduler_stepsize,
    lr_scheduler_gamma,
    lr_weight_decay,
    train_loader,
    val_loader,
    test_loader,
    early_stop_patience,
    node_observed_ratio,
    case_normalize_ratio,
    graph_lambda_0,
    graph_lambda_n,
    graph_lambda_epoch_max,
    graph_lambda_method,
    device,
    writer,
    result_paths,
    comp_last
):

    opt = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr,
        betas=(0.9, 0.999),
        weight_decay=lr_weight_decay,
    )
    # opt = torch.optim.Adam(model.parameters(), lr, betas=(0.9, 0.999), weight_decay=lr_weight_decay)
    # scheduler = torch.optim.lr_scheduler.StepLR(
    #     opt, step_size=lr_scheduler_stepsize, gamma=lr_scheduler_gamma
    # )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt, mode='min', factor=lr_scheduler_gamma, patience=lr_scheduler_stepsize, min_lr=lr_min, threshold=1e-4)

    params_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"# {params_total} params")


    losses = {"train": [], "val": [], "test": []}

    time_start = time.time()

    # with open(result_paths["csv"], "w"):
    #     pass
    # with open(result_paths["csv"], "a") as fw:
    #     fw.write(
    #         f"Epoch,loss_train,loss_val,loss_test,mae_val,rmse_val,mae_test,rmse_test,err,lr,time\n"
    #     )

    loss_best, model_best, epoch_best = float("inf"), None, 0
    early_stop_wait = 0  # 控制 early stop 的行为。当其值达到阈值时停止训练
    early_stop = False

    try:
        for e in range(epochs):

            msg_file_logger = ""
            time1 = time.time()

            _lr = opt.param_groups[0]["lr"]


            adj_lambda = adjust_lambda(
                e, epochs, graph_lambda_0, graph_lambda_n, graph_lambda_epoch_max, graph_lambda_method
            )
            print(
                f"[Epoch] {font_underlined(font_yellow(e))}/{epochs}, [lr] {_lr}",
                end="",
            )
            msg_file_logger += (
                f"[Epoch] {e}/{epochs}, [lr] {_lr}, "
            )

            print(f", [adj_lambda] {adj_lambda:.5f}", end="")
            msg_file_logger += f"[adj_lambda] {adj_lambda}, "

            model.train()
            loss_res = []
            hits10_res = []

            for data in train_loader:
                opt.zero_grad()

                data = tuple(d.to(device) for d in data)
                y_hat, x_case, y_case, x_mob, y_mob, idx_dataset = train_model(data, adj_lambda, model, node_observed_ratio)

                if isinstance(y_hat, tuple):
                    y_hat, adj_hat = y_hat

                    with torch.no_grad():

                        adj_gt = torch.cat([x_mob, y_mob], dim=1)
                        adj_gt_no_diag = rm_self_loops(adj_gt)

                        mean_adj_hat = adj_hat.mean(axis=(-2, -1), keepdims=True)
                        std_adj_hat = adj_hat.std(axis=(-2, -1), keepdims=True)
                        mean_adj_gt_no_diag = adj_gt_no_diag.mean(axis=(-2, -1), keepdims=True)
                        std_adj_gt_no_diag = adj_gt_no_diag.std(axis=(-2, -1), keepdims=True)
                        
                        # adj_hat = (adj_hat - mean_adj_hat) / std_adj_hat * std_adj_gt_no_diag + mean_adj_gt_no_diag
                        # adj_hat = rm_self_loops(adj_hat)

                        adj_hat = adj_hat * mean_adj_gt_no_diag / mean_adj_hat

                    # loss = criterion(y_case.float(), y_hat.float()) + adj_lambda * criterion(adj_gt_no_diag.float(), adj_hat.float())
                    loss = criterion(y_case.float(), y_hat.float())

                    hits10 = compute_hits_at_k(adj_hat.float(), adj_gt_no_diag.float())
                    hits10_res.append(hits10)
                else:
                    loss = criterion(y_case.float(), y_hat.float())

                loss.backward(retain_graph=True)
                opt.step()

                loss_res.append(loss.data.cpu().item())

            loss = np.mean(loss_res)
            if len(hits10_res) > 0: hits10 = np.mean(hits10_res)

            # 记录 train loss
            msg_file_logger += f"[Loss(train/val/test)] {loss:.3f}/"
            print(f", [Loss(train/val/test)] {loss:.3f}/", end="")
            losses["train"].append(loss)

            loss_val, y_real_val, y_hat_val, adj_real_val, adj_hat_val = validate_test_process(model, criterion,
                                                                                               val_loader)
            loss_test, y_real_test, y_hat_test, adj_real_test, adj_hat_test = validate_test_process(model, criterion,
                                                                                                    test_loader)

            if len(hits10_res) > 0:
                # hits10_train = compute_hits_at_k(adj_hat_train, adj_real_train)
                hits10_val = compute_hits_at_k(adj_hat_val, adj_real_val)
                hits10_test = compute_hits_at_k(adj_hat_test, adj_real_test)

            # 拼接记录 val/test loss
            msg_file_logger += f"{loss_val:.3f}/{loss_test:.3f}"
            print(f"{font_yellow(loss_val)}/{loss_test:.3f}", end="")
            losses["val"].append(loss_val)
            losses["test"].append(loss_test)

            if len(hits10_res) > 0:
                # 记录 hits 10
                msg_file_logger += ", [HITS@10(train/val/test)] {:.5f}/{:.5f}/{:.5f}".format(
                    hits10, hits10_val,hits10_test
                )
                print(", [HITS@10(train/val/test)] {:.5f}/{:.5f}/{:.5f}".format(
                    hits10, hits10_val,hits10_test
                ), end="")

            # 记录 时间
            time2 = time.time()
            print(f" ({time2 - time1:.3f}s)", end="")
            msg_file_logger += f" ({time2 - time1:.3f}s)"

            if loss_val < loss_best:
                # 记录 保存最优模型
                msg_file_logger += " (Best model saved)"
                print(font_yellow(" (Best model saved)"), end="")
                loss_best, model_best, epoch_best = loss_val, model, e
                torch.save(model_best, result_paths["model"])
            # # early stop 规则：连续 args.early_stop_patience 个 epoch没有更优结果，停止训练
            #     early_stop_wait = 0
            # else:
            #     early_stop_wait += 1
            #     if early_stop_wait == early_stop_patience:
            #         early_stop = True

            # early stop 规则：loss val 连续 3 次差值小于 1e-2
            else:
                last = 3
                last_losses = losses['val'][-min(last, len(losses['val'])):]
                if len(losses['val']) > 2 and max(last_losses) - min(last_losses) < 1e-1:
                    msg_file_logger += " (Early stop)"
                    print(" (Early stop)")
                    early_stop = True

            print()
            file_logger.info(msg_file_logger)

            metrics = *compute_mae_rmse(y_hat_val, y_real_val), *compute_mae_rmse(y_hat_test, y_real_test)

            err_val, err_test = compute_err(y_hat_val, y_real_val, comp_last), compute_err(y_hat_test, y_real_test, comp_last)

            logger.info("[val(MAE/RMSE)] {:.3f}/{:.3f}, [test(MAE/RMSE)] {:.3f}/{:.3f}".format(*metrics))
            logger.info(f"[err_val] {err_val:.3f}, [err_test] {font_green(err_test)}")

            # # fw.write(f"Epoch,loss_train,loss_val,loss_test,mae_val,rmse_val,mae_test,rmse_test,time\n")
            # with open(result_paths["csv"], "a") as fw:
            #     fw.write(
            #         "{},{:.5f},{:.5f},{:.5f},{},{},{},{},{},{},{},{:2f}s\n".format(
            #             e, loss, loss_val, loss_test, *metrics, err_val, err_test, lr, time2 - time1
            #         )
            #     )

            # region tensorboard

            # Graph
            if e == 0:
                writer.add_graph(model, (x_case, y_case, x_mob, y_mob, idx_dataset))
            # Histogram
            for name, param in model.named_parameters():
                writer.add_histogram(name, param, e)
            # # PR Curve
            # if e % 5 == 0:
            #     writer.add_pr_curve_raw(
            #         'PR Curve',
            #         y_real_val.reshape(-1).tolist(),
            #         y_hat_val.reshape(-1).tolist(),
            #         global_step=e
            #     )
            # Text
            writer.add_text(
                "Text",
                f"Epoch: {e}, Loss: {loss:.5f}, Loss_val: {loss_val:.5f}, Loss_test: {loss_test:.5f}, "
                f"MAE_val: {metrics[0]:.5f}, RMSE_val: {metrics[1]:.5f}, MAE_test: {metrics[2]:.5f}, RMSE_test: {metrics[3]:.5f},"
                f"Err_val: {err_val:.5f}, Err_test: {err_test:.5f}, LR: {lr:.5f}, Time(s): {time2 - time1:.5f}",
                global_step=e,
            )
            # Scalar
            writer.add_scalar("Loss/train", loss, e)
            writer.add_scalar("Loss/validate", loss_val, e)
            writer.add_scalar("Loss/test", loss_test, e)
            for i, metric_label in enumerate(metrics_labels):
                writer.add_scalar(f"Metric/{metric_label}", metrics[i], e)
            writer.add_scalar("Metric/Err_val", err_val, e)
            writer.add_scalar("Metric/Err_test", err_test, e)
            if len(hits10_res) > 0:
                writer.add_scalar("Metric/HITS@10_train", hits10, e)
                writer.add_scalar("Metric/HITS@10_val", hits10_val, e)
                writer.add_scalar("Metric/HITS@10_test", hits10_test, e)
            writer.add_scalar("Others/Learning_Rate", lr, e)
            writer.add_scalar("Others/Time(s)", time2 - time1, e)
            
            # endregion

            if early_stop:
                break

            scheduler.step(loss_val)
            # scheduler.step()

            for group in opt.param_groups:
                if group["lr"] < lr_min:
                    group["lr"] = lr_min

            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # 梯度剪裁

    except KeyboardInterrupt:
        logger.info("已手动停止训练")

    return losses, model, epoch_best, loss_best


@torch.no_grad()
def validate_test_process(model: nn.Module, criterion, dataloader):
    device = next(model.parameters()).device
    model.eval()
    loss_res = []
    y_hat_res, adj_hat_res, adj_real_res = [], [], []

    for data in dataloader:

        data = tuple(d.to(device) for d in data)
        y_hat, x_case, y_case, x_mob, y_mob, idx_dataset = run_model(data, model)
        # adj_gt = torch.cat([x_mob, y_mob], dim = -3)
        adj_gt = min_max_adj(torch.cat([x_mob, y_mob], dim = 1))
        adj_real_res.append(adj_gt.float())

        # 读取结果计算 loss
        if isinstance(y_hat, tuple):
            y_hat, adj_hat = y_hat  # 适配启用图学习器的情况
            
            adj_hat = min_max_adj(adj_hat) # TODO: HITS10

            adj_hat_res.append(adj_hat.float())

        loss = criterion(y_case.float(), y_hat.float())
        loss_res.append(loss.item())
        y_hat_res.append(y_hat.float())

    loss = np.mean(loss_res)

    y_hat_res = torch.cat(y_hat_res) # 含 batch 维度：cat
    adj_hat_res = torch.cat(adj_hat_res) if len(adj_hat_res) > 0 else [] # 含 batch 维度：cat
    adj_real_res = torch.cat(adj_real_res) # 含 batch 维度：cat

    y_real_res = torch.stack([d[1] for d in dataloader.dataset]).to(y_hat_res.device) # 不含 batch 维度：stack
    return loss, y_real_res, y_hat_res, adj_real_res, adj_hat_res


def eval_process(model, criterion, train_loader, val_loader, test_loader, comp_last):
    # 读取模型
    trained_model = None
    if isinstance(model, str):
        assert os.path.exists(model)
        trained_model = torch.load(model)
    else:
        trained_model = model

    # 分别在 train/val/test 三个数据集上跑出结果

    loss_train, y_real_train, y_hat_train, adj_real_train, adj_hat_train = validate_test_process(trained_model, criterion, train_loader)
    loss_val, y_real_val, y_hat_val, adj_real_val, adj_hat_val = validate_test_process(trained_model, criterion, val_loader)
    loss_test, y_real_test, y_hat_test, adj_real_test, adj_hat_test = validate_test_process(trained_model, criterion, test_loader)

    metrics = *compute_mae_rmse(y_hat_val, y_real_val), *compute_mae_rmse(y_hat_test, y_real_test)
    err_val, err_test = compute_err(y_hat_val, y_real_val, comp_last), compute_err(y_hat_test, y_real_test, comp_last)

    corr_train = compute_correlation(y_hat_train, adj_real_train)[0]
    corr_val = compute_correlation(y_hat_val, adj_real_val)[0]
    corr_test = compute_correlation(y_hat_test, adj_real_test)[0]

    if not adj_hat_train == []: 
        hits10_train = compute_hits_at_k(adj_hat_train, adj_real_train)
        hits10_val = compute_hits_at_k(adj_hat_val, adj_real_val)
        hits10_test = compute_hits_at_k(adj_hat_test, adj_real_test)

    logger.info(
        "[val(MAE/RMSE)] {:.3f}/{:.3f}, [test(MAE/RMSE)] {:.3f}/{:.3f}".format(*metrics))
    
    if adj_hat_train == []: 
        logger.info(
            "[err(val/test)] {:.3f}/{}, [corr(train/val/test)] {:.3f}/{:.3f}/{:.3f}".format(
                err_val, font_green(err_test), corr_train, corr_val, corr_test))
    else:
        logger.info(
            "[err(val/test)] {:.3f}/{}, [corr(train/val/test)] {:.3f}/{:.3f}/{:.3f}, [hits10(train/val/test)] {:.5f}/{:.5f}/{:.5f}".format(
                err_val, font_green(err_test), corr_train, corr_val, corr_test, hits10_train, hits10_val, hits10_test))

    res = {
        "mae_val": metrics[0],
        "rmse_val": metrics[1],
        "mae_test": metrics[2],
        "rmse_test": metrics[3],
        "err_val": err_val,
        "err_test": err_test,
        "corr_train": corr_train,
        "corr_val": corr_val,
        "corr_test": corr_test
    }
    if not adj_hat_train == []:
        res = {**res, **{"hits10_train": hits10_train, "hits10_val": hits10_val, "hits10_test": hits10_test}}
    return res

# 训练测试模型的子过程

def train_model(data, adj_lambda, model, node_observed_ratio = 0.8):
    
    x_case, y_case, x_mob, y_mob, idx_dataset = data
    x_case, y_case, x_mob, y_mob, idx_dataset = random_mask((x_case, y_case, x_mob, y_mob, idx_dataset), node_observed_ratio)

    return run_model(data, model, adj_lambda)

def run_model(data, model, adj_lambda = None):

    x_case, y_case, x_mob, y_mob, idx_dataset = data
    y_hat = model(x_case.float(), y_case.float(), x_mob.float(), y_mob.float(), adj_lambda)

    # 适应模型同时输出图结构的情况
    y_hat_shape = y_hat[0].shape if isinstance(y_hat, tuple) else y_hat.shape
    if y_case.shape != y_hat_shape: y_case = y_case[:, -y_hat_shape[1]:]
    assert y_hat_shape == y_case.shape

    return y_hat, x_case, y_case, x_mob, y_mob, idx_dataset
