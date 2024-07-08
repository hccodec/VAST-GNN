import traceback
import torch, time, os
from torch import nn
from eval import compute, metrices_labels

from utils.logger import file_logger, logger
# from utils.tensorboard import writer
from utils.utils import catch, green, yellow
# logger = logger.getLogger()

def train_epoch(
        args, model, opt, criterion, train_loader, e
):
    model.train()
    loss_res = []

    for mobility, text, casex, casey, idx in train_loader:
        opt.zero_grad()

        mobility = mobility.to(args.device)
        text = text.to(args.device)
        casex = casex.to(args.device)
        casey = casey.to(args.device)
        idx = idx.to(args.device)

        y_hat = model(mobility, text, casex, idx)
        y = casey[:,:,:,0].to(y_hat.device)

        loss = criterion(y.float(), y_hat.float())
        loss.backward()
        opt.step()

        loss_res.append(loss.data.cpu().numpy())

    return sum(loss_res) / len(loss_res), model

@catch()
def train_process(
    args, model, criterion,
    train_loader, validation_loader, test_loader,
    writer):
    
    opt = torch.optim.Adam(model.parameters(), args.lr, betas=(0.9, 0.999), weight_decay=0)

    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.7)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        # opt, mode='min', factor=0.7, patience=5, min_lr=1e-5, threshold=1e-4)
    
    result_paths = {
        'model': os.path.join(args.result_dir, "best_model_jp.pth"),
        'csv': os.path.join(args.result_dir, 'results_jp.csv')
    }
    losses = {'train': [], 'val': [], 'test': []}

    time_start = time.time()

    with open(result_paths['csv'], 'w'): pass
    with open(result_paths['csv'], "a") as fw:
        fw.write(f"Epoch,loss_train,loss_val,loss_test,mae_val,rmse_val,mae_test,rmse_test,lr,time\n")
    

    loss_best = 1e7
    model_best = None
    early_stop = False

    try:
        for e in range(args.epochs):

            msg_file_logger = ""
            time1 = time.time()

            lr = opt.param_groups[0]['lr']

            print(f"[Epoch] {green(e)}/{args.epochs}, [lr] {lr:.7f}, ", end='')
            msg_file_logger += f"[Epoch] {e}/{args.epochs}, [lr] {lr}, "
            loss, model = train_epoch(
                args, model, opt, criterion, train_loader, e
            )
            loss *= args.case_normalize_ratio ** 2

            # todo：loss 放缩  ratio ** 2
            msg_file_logger += f"[Loss(train/val/test)] {loss:.3f}/"
            print(f"[Loss(train/val/test)] {loss:.3f}/", end='')
            losses['train'].append(loss)

            loss_val, y_hat_val, y_real_val = validate_test_process(model, criterion, validation_loader)
            loss_test, y_hat_test, y_real_test = validate_test_process(model, criterion, test_loader)
            loss_val *= args.case_normalize_ratio ** 2
            loss_test *= args.case_normalize_ratio ** 2

            msg_file_logger += f"{loss_val:.3f}/{loss_test:.3f}"
            print(f"{yellow(loss_val)}/{loss_test:.3f}")
            losses['val'].append(loss_val)
            losses['test'].append(loss_test)

            time2 = time.time()
            print(f'[time] {time2 - time1:.3f}s', end='')

            if loss_val < loss_best:
                msg_file_logger += " (Best model saved)"
                print(" (Best model saved)", end='')
                torch.save(model, result_paths['model'])
                loss_best = loss_val

            # early stop 规则：loss val 连续 2 次差值小于 1e-2
            last = 2
            last_losses = losses['val'][-min(last, len(losses['val'])):]
            if len(losses['val']) > 2 and max(last_losses) - min(last_losses) < 1e-2:
                msg_file_logger += " (Early stop)"
                print(" (Early stop)")
                early_stop = True

            print()
            file_logger.info(msg_file_logger)

            metrices = compute(y_hat_val, y_real_val, y_hat_test, y_real_test, args.case_normalize_ratio)

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

            for g in opt.param_groups:
                if g['lr'] < args.lr_min:
                    g['lr'] = args.lr_min
        print('-' * 20)


    except KeyboardInterrupt:
        file_logger.info("已手动停止训练")
    except Exception as e:
        file_logger.info("出现错误，中断训练")
        logger.info(str(e))
        for line in traceback.format_exc():
            logger.info(str(line))
    
    return losses, model



def validate_test_process(model: nn.Module, criterion, dataloader):
    model.eval()
    loss_res = []
    y_hat_res, y_real_res = None, None

    for mobility, text, casex, casey, idx in dataloader:
        mobility = mobility.to(model.device)
        text = text.to(model.device)
        casex = casex.to(model.device)
        casey = casey.to(model.device)
        idx = idx.to(model.device)

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

