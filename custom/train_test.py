import torch, time
from torch import nn
from eval import compute, get_correlation, metrices_labels

from utils.logger import file_logger, logger
# from utils.tensorboard import writer
from utils.utils import font_underlined, catch, font_green, font_yellow
# logger = logger.getLogger()

@catch()
def train_process(
    model, criterion, epochs, lr, lr_min, lr_scheduler_stepsize, lr_scheduler_gamma,
    train_loader, validation_loader, test_loader,
    early_stop_patience, case_normalize_ratio, device, writer, result_paths):
    
    opt = torch.optim.Adam(model.parameters(), lr, betas=(0.9, 0.999), weight_decay=0)

    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=lr_scheduler_stepsize, gamma=lr_scheduler_gamma)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        # opt, mode='min', factor=0.7, patience=5, min_lr=1e-5, threshold=1e-4)
    
    losses = {'train': [], 'val': [], 'test': []}

    time_start = time.time()

    with open(result_paths['csv'], 'w'): pass
    with open(result_paths['csv'], "a") as fw:
        fw.write(f"Epoch,loss_train,loss_val,loss_test,mae_val,rmse_val,mae_test,rmse_test,lr,time\n")
    

    loss_best, model_best, epoch_best = float("inf"), None, 0
    early_stop_wait = 0 # 控制 early stop 的行为。当其值达到阈值时停止训练
    early_stop = False

    try:
        for e in range(epochs):

            msg_file_logger = ""
            time1 = time.time()

            _lr = opt.param_groups[0]['lr']

            print(f"[Epoch] {font_underlined(font_yellow(e))}/{epochs}, [lr] {_lr:.7f}, ", end='')
            msg_file_logger += f"[Epoch] {e}/{epochs}, [lr] {_lr}, "
            
            
            model.train()
            loss_res = []

            for mobility, text, casex, casey, idx in train_loader:
                opt.zero_grad()

                mobility = mobility.to(device)
                text = text.to(device)
                casex = casex.to(device)
                casey = casey.to(device)
                idx = idx.to(device)

                y_hat = model(mobility, text, casex, idx)
                y = casey[:,:,:,0].to(device)

                loss = criterion(y.float(), y_hat.float())
                loss.backward()
                opt.step()

                loss_res.append(loss.data.cpu().numpy())

            loss = sum(loss_res) / len(loss_res)


            loss *= case_normalize_ratio ** 2

            # todo：loss 放缩  ratio ** 2
            msg_file_logger += f"[Loss(train/val/test)] {loss:.3f}/"
            print(f"[Loss(train/val/test)] {loss:.3f}/", end='')
            losses['train'].append(loss)

            loss_val, y_hat_val, y_real_val = validate_test_process(model, criterion, validation_loader, device)
            loss_test, y_hat_test, y_real_test = validate_test_process(model, criterion, test_loader, device)
            loss_val *= case_normalize_ratio ** 2
            loss_test *= case_normalize_ratio ** 2

            msg_file_logger += f"{loss_val:.3f}/{loss_test:.3f}"
            print(f"{font_yellow(loss_val)}/{loss_test:.3f}, ", end='')
            losses['val'].append(loss_val)
            losses['test'].append(loss_test)

            time2 = time.time()
            print(f'({time2 - time1:.3f}s)', end='')

            if loss_val < loss_best:
                msg_file_logger += " (Best model saved)"
                print(font_yellow(" (Best model saved)"), end='')
                loss_best , model_best, epoch_best = loss_val, model, e
                torch.save(model_best, result_paths['model'])
                early_stop_wait = 0
            # # early stop 规则：连续 args.early_stop_patience 个 epoch没有更优结果，停止训练
            else:
                early_stop_wait += 1
                if early_stop_wait == early_stop_patience:
                    early_stop = True

            # early stop 规则：loss val 连续 2 次差值小于 1e-2
            # last = 3
            # last_losses = losses['val'][-min(last, len(losses['val'])):]
            # if len(losses['val']) > 2 and max(last_losses) - min(last_losses) < 1e-2:
            #     msg_file_logger += " (Early stop)"
            #     print(" (Early stop)")
            #     early_stop = True

            print()
            file_logger.info(msg_file_logger)

            metrices = compute(y_hat_val, y_real_val, y_hat_test, y_real_test, case_normalize_ratio)

            # fw.write(f"Epoch,loss_train,loss_val,loss_test,mae_val,rmse_val,mae_test,rmse_test,time\n")
            with open(result_paths['csv'], 'a') as fw:
                fw.write('{},{:.5f},{:.5f},{:.5f},{},{},{},{},{},{:2f}s\n'.format(
                    e, loss, loss_val, loss_test, *metrices, lr, time2 - time1
                ))

            writer.add_scalar('Loss/train', loss, e)
            writer.add_scalar('Loss/validate', loss_val, e)
            writer.add_scalar('Loss/test', loss_test, e)
            for i, metric_label in enumerate(metrices_labels):
                writer.add_scalar(f"Metric/{metric_label}", metrices[i], e)
            writer.add_scalar('Others/Learning_Rate', lr, e)
            writer.add_scalar('Others/Time(s)', time2 - time1, e)
            
            if early_stop: break

            scheduler.step()

            for group in opt.param_groups:
                if group['lr'] < lr_min:
                    group['lr'] = lr_min

            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # 梯度剪裁

    except KeyboardInterrupt:
        logger.info("已手动停止训练")
        
    
    return losses, model, epoch_best


@torch.no_grad()
def validate_test_process(model: nn.Module, criterion, dataloader, device):
    model.eval()
    loss_res = []
    y_hat_res, y_real_res = None, None

    for mobility, text, casex, casey, idx in dataloader:
        mobility = mobility.to(device)
        text = text.to(device)
        casex = casex.to(device)
        casey = casey.to(device)
        idx = idx.to(device)

        y = casey[:,:,:,0]
        y_hat = model(mobility, text, casex, idx)
        y = y.to(y_hat.device)
        loss = criterion(y.float(), y_hat.float())

        loss_res.append(loss.item())
        if y_hat_res is None: y_hat_res = y_hat
        else: y_hat_res = torch.cat([y_hat_res, y_hat], 0)
        if y_real_res is None: y_real_res = y
        else: y_real_res = torch.cat([y_real_res, y], 0)

    return sum(loss_res) / len(loss_res), y_hat_res, y_real_res


def eval_process(model, criterion,
                 train_loader, validation_loader, test_loader,
                 y_days, case_normalize_ratio, device):
    
    trained_model = torch.load(model) if isinstance(model, str) else model

    validation_result, validation_hat, validation_real = validate_test_process(trained_model, criterion, validation_loader, device)
    test_result, test_hat, test_real = validate_test_process(trained_model, criterion, test_loader, device)

    metrices = compute(
        validation_hat, validation_real,
        test_hat, test_real, case_normalize_ratio
    )
    
    train_result, train_hat, train_real = validate_test_process(trained_model, criterion, train_loader, device)
    get_correlation(
        train_hat, train_real, validation_hat, validation_real, test_hat, test_real, y_days
        )