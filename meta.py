import os
from tensorboardX import SummaryWriter

import torch

from train_test import train_process
from train_test_meta import meta_test_process, meta_test_train_process, meta_train_process
from utils.logger import logger
from utils.model_selector import select_model
from utils.utils import font_yellow, get_country

def maml_train(args, result_paths, meta_data, i_country):

    country_name = meta_data["country_names"][i_country]
    country_code = meta_data["country_codes"][i_country]
    
    # 从 args 读取数据
    lr = args.lr
    lr_min = args.lr_min
    lr_scheduler_stepsize = args.lr_scheduler_stepsize
    lr_weight_decay = args.lr_weight_decay
    lr_scheduler_gamma = args.lr_scheduler_gamma
    lr_meta = args.lr_meta
    epochs = args.epochs
    device = args.device
    early_stop_patience = args.early_stop_patience
    node_observed_ratio = args.node_observed_ratio
    case_normalize_ratio = args.case_normalize_ratio

    comp_last = args.comp_last

    # graph_lambda = args.lambda_graph_loss[country_name][(1 + args.shift) if args.ydays == 1 else args.ydays] if country_name in args.lambda_graph_loss else 0
    graph_lambda = args.graph_lambda

    logger.info(f"开始针对国家 {country_name} 进行元训练...")

    # 记录实验参数
    result_paths.update(
        {
            "model": os.path.join(
                args.result_dir, f"model_{country_code}_best.pth"
            ),
            "model_latest": os.path.join(
                args.result_dir, f"model_{country_code}_latest.pth"
            ),
            "model_eta": os.path.join(
                args.result_dir, f"model_{country_code}_eta.pth"
            ),
            # "csv": os.path.join(args.result_dir, f"results_{country_code}.csv"),
            "tensorboard": os.path.join(
                args.result_dir, f"tensorboard_{country_code}"
            ),
        }
    )

    # 初始化 tensorboard 记录器
    with SummaryWriter(result_paths["tensorboard"]) as writer:

        criterion = torch.nn.functional.mse_loss

        train_loader, val_loader, test_loader = meta_data['data'][country_name][0]

        # 选择模型
        model, model_args = select_model(args, train_loader)
        opt = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr,
            betas=(0.9, 0.999),
            weight_decay=lr_weight_decay,
        )
        torch.save({'state_dict': model.state_dict(),'optimizer' : opt.state_dict(),}, result_paths["model_eta"])
        
        # criterion = torch.nn.MSELoss()

        # 记录模型参数
        with open(result_paths["args"], "a", encoding="utf-8") as f: f.write(f"\n[model_args:{country_name}]\n{str(model_args)}\n")

        meta_train_process(
            model, criterion, epochs, i_country, opt,
            lr, lr_min, lr_scheduler_stepsize, lr_scheduler_gamma, lr_weight_decay, lr_meta,
            meta_data, # train_loader, val_loader, test_loader,
            early_stop_patience, node_observed_ratio, case_normalize_ratio, graph_lambda,
            # graph_lambda_0, graph_lambda_n, graph_lambda_epoch_max, graph_lambda_method,
            device, writer, result_paths, comp_last
        )

        logger.info("")
        logger.info(f"元训练完毕，开始元测试: {country_name}")

        model, model_args = select_model(args, train_loader)
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

        checkpoint = torch.load(result_paths["model_eta"])
        model.load_state_dict(checkpoint['state_dict'])
        opt.load_state_dict(checkpoint['optimizer'])

        losses, trained_model, epoch_best, loss_best = meta_test_train_process(
            model, opt, scheduler, criterion, epochs,
            lr, lr_min, lr_scheduler_stepsize, lr_scheduler_gamma, lr_weight_decay,
            train_loader, val_loader, test_loader,
            early_stop_patience, node_observed_ratio, case_normalize_ratio,
            graph_lambda,
            # graph_lambda_0,
            # graph_lambda_n,
            # graph_lambda_epoch_max,
            # graph_lambda_method,
            device, writer, result_paths, comp_last)

        torch.save(trained_model.state_dict(), result_paths["model_latest"])
        
        logger.info("")
        logger.info(f"训练完毕，开始评估: {country_name}")
        
        logger.info("-" * 20)
        logger.info(font_yellow(f"[最新 (epoch {len(losses['train']) - 1})]"))
        metrics_latest = meta_test_process(args, result_paths["model_latest"], criterion, train_loader, val_loader, test_loader, comp_last)
        logger.info("-" * 20)
        logger.info(font_yellow(f"[最小 val loss (epoch {epoch_best})]"))
        metrics_minvalloss = meta_test_process(args, result_paths["model"], criterion, train_loader, val_loader, test_loader, comp_last)

        writer.add_hparams(
            {
                **{
                    k: (v if isinstance(v, (int, float, str, bool, torch.Tensor)) else str(v))
                    for k, v in vars(args).items() if k in ["xdays", "ydays", "window", "batch_size", "lr", "lr_min", "seed"]
                },
                **{"country_name": country_name, "country_code": country_code},
            },
            {
                **{f"{k}_minvalloss": float(v) for k, v in metrics_minvalloss.items() if not k == 'outputs'},
                **{f"{k}_latest": float(v) for k, v in metrics_latest.items() if not k == 'outputs'},
            },
        )


import torch, time, os, re
from torch import nn
import numpy as np
from eval import compute_err, compute_mae_rmse, metrics_labels, compute_correlation, compute_hits_at_k

from utils.logger import file_logger, logger

# from utils.tensorboard import writer
from utils.model_selector import select_model
from utils.utils import adjust_lambda, font_underlined, font_green, font_yellow, min_max_adj, rm_self_loops, getLaplaceMat, scale_adj
from models.VAST_GNN import vast_gnn_extra_info


# logger = logger.getLogger()
