import os, sys, torch
sys.path.append(os.getcwd())
from torch.utils.tensorboard import SummaryWriter

from utils.data_process import load_data, split_dataset 
from train_test import train_process, validate_test_process, eval_process
from eval import get_correlation, compute

from utils.logger import logger
from utils.utils import select_model, set_random_seed, parse_args

def main():
    args = parse_args()

    result_paths = {
        'model': os.path.join(args.result_dir, "model_jp_best.pth"),
        'model_latest': os.path.join(args.result_dir, "model_jp_latest.pth"),
        'csv': os.path.join(args.result_dir, 'results_jp.csv'),
        'log': os.path.join(args.result_dir, 'log.txt')
    }

    logger.info(f"运行结果将保存至 {args.result_dir}")

    with open(os.path.join(args.result_dir, 'help.txt'), 'w') as f:
        for k, v in args._get_kwargs():
            f.write("{}: {}\n".format(k, v))


    set_random_seed(args.seed)

    preprocessed_data_dir, databinfile = args.preprocessed_data_dir, args.databinfile
    
    start_date, end_date, x_days, y_days = args.startdate, args.enddate, args.xdays, args.ydays
    data_dir, case_normalize_ratio, text_normalize_ratio = args.data_dir, args.case_normalize_ratio, args.text_normalize_ratio

    data_origin, date_all = load_data(args)
    
    # 分割数据集
    train_loader, validation_loader, test_loader, train_origin, validation_origin, test_origin, train_indices, validation_indices, test_indices = split_dataset(args, data_origin, date_all)

    model = select_model(args, train_loader)
    criterion = torch.nn.MSELoss()

    logger.info("数据准备完成，开始训练")

    writer = SummaryWriter(args.result_dir)

    lr = args.lr
    lr_min = args.lr_min
    lr_scheduler_stepsize = args.lr_scheduler_stepsize
    lr_scheduler_gamma = args.lr_scheduler_gamma
    epochs = args.epochs
    device = args.device
    early_stop_patience = args.early_stop_patience
    case_normalize_ratio = args.case_normalize_ratio

    losses, trained_model, epoch_best = train_process(
        model, criterion, epochs, lr, lr_min, lr_scheduler_stepsize, lr_scheduler_gamma,
        train_loader, validation_loader, test_loader,
        early_stop_patience, case_normalize_ratio,
        device, writer, result_paths
    )
    writer.close()

    logger.info("训练完毕，开始评估: ")

    logger.info(f"最新")
    eval_process(trained_model, criterion,
                 train_loader, validation_loader, test_loader,
                 y_days, case_normalize_ratio, device)
    
    logger.info(f"Loss validate 最小 (epoch {epoch_best})")
    eval_process(result_paths['model'], criterion,
                 train_loader, validation_loader, test_loader,
                 y_days, case_normalize_ratio, device)

    logger.info(f"实验（波次 {args.wave}, 预测范围 {args.xdays}->{args.ydays}）结束")
    logger.info(f"实验结果已保存至 {args.result_dir}")

if __name__ == "__main__":
    main()