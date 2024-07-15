import os, sys, torch
sys.path.append(os.getcwd())
from torch.utils.tensorboard import SummaryWriter

from utils.data_process import load_data, split_dataset 
from train_test import train_process, validate_test_process, eval_process
from eval import get_correlation, compute

from utils.logger import logger
from utils.utils import font_yellow, select_model, set_random_seed, parse_args

def main():
    args = parse_args()
    try:
        
        preprocessed_data_dir, databinfile = args.preprocessed_data_dir, args.databinfile
        
        set_random_seed(args.seed)
        logger.info(f"运行结果将保存至 {args.result_dir}")

        # 准备数据集
        data_origin, date_all = load_data(args)
        train_loader, validation_loader, test_loader, train_origin, validation_origin, test_origin, train_indices, validation_indices, test_indices = split_dataset(args, data_origin, date_all)
        logger.info("数据准备完成，开始训练")

        # 记录实验参数
        result_paths = {
            'model': os.path.join(args.result_dir, "model_jp_best.pth"),
            'model_latest': os.path.join(args.result_dir, "model_jp_latest.pth"),
            'csv': os.path.join(args.result_dir, 'results_jp.csv'),
            'log': os.path.join(args.result_dir, 'log.txt'),
            'help': os.path.join(args.result_dir, 'help.txt')
        }

        with open(result_paths['help'], 'w') as f:
            f.write('[args]\n')
            for k, v in args._get_kwargs():
                f.write("{}: {}\n".format(k, v))

        # 选择模型
        model, model_args = select_model(args, train_loader)
        criterion = torch.nn.MSELoss()
        
        # 记录模型参数
        with open(result_paths['help'], 'a') as f:
            f.write('\n[model_args]\n')
            f.write(str(model_args) + '\n')

        # 初始化 tensorboard 记录器
        writer = SummaryWriter(args.result_dir)

        lr = args.lr
        lr_min = args.lr_min
        lr_scheduler_stepsize = args.lr_scheduler_stepsize
        lr_scheduler_gamma = args.lr_scheduler_gamma
        epochs = args.epochs
        device = args.device
        early_stop_patience = args.early_stop_patience
        case_normalize_ratio = args.case_normalize_ratio

        # start_date, end_date, x_days, y_days = args.startdate, args.enddate, args.xdays, args.ydays
        # data_dir, case_normalize_ratio, text_normalize_ratio = args.data_dir, args.case_normalize_ratio, args.text_normalize_ratio

        losses, trained_model, epoch_best = train_process(
            model, criterion, epochs, lr, lr_min, lr_scheduler_stepsize, lr_scheduler_gamma,
            train_loader, validation_loader, test_loader,
            early_stop_patience, case_normalize_ratio,
            device, writer, result_paths
        )
        writer.close()

        logger.info('')
        logger.info("训练完毕，开始评估: ")

        logger.info('-' * 20)
        logger.info(font_yellow("[最新]"))
        eval_process(trained_model, criterion,
                    train_loader, validation_loader, test_loader,
                    args.ydays, case_normalize_ratio, device)
        
        logger.info('-' * 20)
        logger.info(font_yellow(f"[最小 val loss (epoch {epoch_best})]"))
        eval_process(result_paths['model'], criterion,
                    train_loader, validation_loader, test_loader,
                    args.ydays, case_normalize_ratio, device)

        logger.info(f"实验（波次 {args.wave}, 预测范围 {args.xdays}->{args.ydays}）结束")
    finally:
        logger.info(f"实验结果已保存至 {args.result_dir}")

if __name__ == "__main__":
    main()