import torch, time
from torch import nn
from eval import compute_err, compute_metrics, compute_correlation, metrics_labels

from utils.logger import file_logger, logger

# from utils.tensorboard import writer
from utils.utils import adjust_lambda, font_underlined, catch, font_green, font_yellow, process_batch, hits_at_k
from models.dynst.model import dynst, dynst_extra_info


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
    validation_loader,
    test_loader,
    early_stop_patience,
    nodes_observed_ratio,
    case_normalize_ratio,
    graph_lambda_0,
    graph_lambda_k,
    graph_lambda_method,
    device,
    writer,
    result_paths,
    enable_graph_learner,
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
                e, graph_lambda_0, graph_lambda_k, graph_lambda_method
            )

            print(
                f"[Epoch] {font_underlined(font_yellow(e))}/{epochs}, [lr] {_lr}, ",
                end="",
            )
            msg_file_logger += (
                f"[Epoch] {e}/{epochs}, [lr] {_lr}, [adj_lambda] {adj_lambda}, "
            )

            if enable_graph_learner:
                print(f"[adj_lambda] {adj_lambda}, ", end="")
                msg_file_logger += f"[adj_lambda] {adj_lambda}, "

            model.train()
            loss_res = []

            for data in train_loader:
                opt.zero_grad()

                data = tuple(d.to(device) for d in data)
                y_hat, casex, casey, mobility, idx, extra_info = train_model(data, adj_lambda, model, nodes_observed_ratio)

                adj_y = extra_info.dataset_extra if isinstance(extra_info, dynst_extra_info) else extra_info
                assert isinstance(adj_y, torch.Tensor) and adj_y.shape == (*casey.shape[:-1], casey.shape[-2])
                adj_gt = torch.cat([mobility, adj_y], dim = 1)

                if enable_graph_learner:
                    y_hat, adj_hat = y_hat
                    loss = criterion(casey.float(), y_hat.float()) + adj_lambda * criterion(
                        adj_gt.float(), adj_hat.float()
                    )
                    hits10 = hits_at_k(adj_hat, adj_gt)
                else:
                    loss = criterion(casey.float(), y_hat.float())

                loss.backward(retain_graph=True)
                opt.step()

                loss_res.append(loss.data.cpu().numpy())

            loss = sum(loss_res) / len(loss_res)

            loss *= case_normalize_ratio**2

            # todo：loss 放缩  ratio ** 2
            msg_file_logger += f"[Loss(train/val/test)] {loss:.3f}/"
            print(f"[Loss(train/val/test)] {loss:.3f}/", end="")
            losses["train"].append(loss)

            loss_val, y_hat_val, y_real_val = validate_test_process(
                model, criterion, validation_loader, device
            )
            loss_test, y_hat_test, y_real_test = validate_test_process(
                model, criterion, test_loader, device
            )
            loss_val *= case_normalize_ratio**2
            loss_test *= case_normalize_ratio**2

            msg_file_logger += f"{loss_val:.3f}/{loss_test:.3f}"
            print(f"{font_yellow(loss_val)}/{loss_test:.3f}, ", end="")
            losses["val"].append(loss_val)
            losses["test"].append(loss_test)

            time2 = time.time()
            print(f"({time2 - time1:.3f}s)", end="")
            msg_file_logger += f"({time2 - time1:.3f}s)"

            if loss_val < loss_best:
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

            metrics = compute_metrics(
                y_hat_val, y_real_val, y_hat_test, y_real_test, case_normalize_ratio
            )
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
                writer.add_graph(model, (casex, casey, mobility, idx, extra_info) if isinstance(extra_info, torch.Tensor) else (casex, casey, mobility))
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
def validate_test_process(model: nn.Module, criterion, dataloader, device):
    model.eval()
    loss_res = []
    y_hat_res, y_real_res = None, None

    for data in dataloader:

        data = tuple(d.to(device) for d in data)
        y_hat, casex, casey, mobility, idx, extra_info = run_model(data, model)

        if isinstance(y_hat, tuple):
            y_hat, _ = y_hat  # 适配启用图学习器的情况
        loss = criterion(casey.float(), y_hat.float())

        loss_res.append(loss.item())
        if y_hat_res is None:
            y_hat_res = y_hat
        else:
            y_hat_res = torch.cat([y_hat_res, y_hat], 0)
        if y_real_res is None:
            y_real_res = casey
        else:
            y_real_res = torch.cat([y_real_res, casey], 0)

    return sum(loss_res) / len(loss_res), y_hat_res, y_real_res


def eval_process(
    model,
    criterion,
    train_loader,
    validation_loader,
    test_loader,
    y_days,
    case_normalize_ratio,
    device,
    comp_last
):

    trained_model = torch.load(model) if isinstance(model, str) else model

    train_result, train_hat, train_real = validate_test_process(
        trained_model, criterion, train_loader, device
    )
    validation_result, validation_hat, validation_real = validate_test_process(
        trained_model, criterion, validation_loader, device
    )
    test_result, test_hat, test_real = validate_test_process(
        trained_model, criterion, test_loader, device
    )

    metrics = compute_metrics(
        validation_hat, validation_real, test_hat, test_real, case_normalize_ratio
    )

    err_val = compute_err(validation_hat, validation_real, comp_last)
    err_test = compute_err(test_hat, test_real, comp_last)
    # correlation = compute_correlation(
    #     train_hat,
    #     train_real,
    #     validation_hat,
    #     validation_real,
    #     test_hat,
    #     test_real,
    #     y_days,
    # )

    return {
        "mae_val": metrics[0],
        "rmse_val": metrics[1],
        "mae_test": metrics[2],
        "rmse_test": metrics[3],
        "err_val": err_val,
        "err_test": err_test,
        # "correlation_train": correlation[0][0],
        # "correlation_val": correlation[1][0],
        # "correlation_test": correlation[2][0],
    }

# 训练测试模型的子过程

def train_model(data, adj_lambda, model, nodes_observed_ratio = 0.8):
    
    data = process_batch(data, adj_lambda, model, nodes_observed_ratio)

    return run_model(data, model)

def run_model(data, model):

    casex, casey, mobility, idx, extra_info = data
    y_hat = model(casex, casey, mobility, extra_info)

    # 适应模型同时输出图结构的情况
    y_hat_shape = y_hat[0].shape if isinstance(y_hat, tuple) else y_hat.shape
    if casey.shape != y_hat_shape: casey = casey[:, -y_hat_shape[1]:]
    assert y_hat_shape == casey.shape

    return y_hat, casex, casey, mobility, idx, extra_info
