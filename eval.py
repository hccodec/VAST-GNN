import torch
from scipy import stats
import numpy as np

from utils.logger import logger
from utils.utils import catch, font_green

metrics_labels = ["MAE_val", "RMSE_val", "MAE_test", "RMSE_test"]

@catch("出现错误，无法计算相关性")
def compute_correlation(
        train_hat, train_real, validate_hat, validate_real, test_hat, test_real, y_days
):
    try:
        validate_hat_sum = [float(torch.sum(validate_hat[i][y_days-1])) for i in range(len(validate_hat))]
        validate_real_sum = [float(torch.sum(validate_real[i][y_days-1])) for i in range(len(validate_real))]
        correlation_val = stats.pearsonr(validate_hat_sum, validate_real_sum)
        # print ("the correlation between validation: ", correlation_val[0])
    except Exception as e:
        correlation_val = [-1]
        logger.info("出现错误，无法计算 val 相关性: " + str(e))
    try:
        #test
        test_hat_sum = [float(torch.sum(test_hat[i][y_days-1])) for i in range(len(test_hat))]
        test_real_sum = [float(torch.sum(test_real[i][y_days-1])) for i in range(len(test_real))]
        correlation_test = stats.pearsonr(test_hat_sum, test_real_sum)
    except Exception as e:
        correlation_test = [-1]
        logger.info("出现错误，无法计算 test 相关性: " + str(e))
        # print ("the correlation between test: ", correlation_test[0])
    try:
        #train
        train_hat_sum = [float(torch.sum(train_hat[i][0])) for i in range(len(train_hat))]
        train_real_sum = [float(torch.sum(train_real[i][0])) for i in range(len(train_real))]
        correlation_train = stats.pearsonr(train_hat_sum, train_real_sum)
        # print ("the correlation between train: ", correlation_train[0])
    except Exception as e:
        correlation_test = [-1]
        logger.info("出现错误，无法计算 train 相关性: " + str(e))
    logger.info(f"Correlation(train/val/test): {correlation_train[0]}, {correlation_val[0]}, {correlation_test[0]}")

    return correlation_train, correlation_val, correlation_test

RMSELoss = lambda _y, y: float(torch.sqrt(torch.mean((_y - y) ** 2)))
MAELoss = lambda _y, y: float(torch.mean(torch.div(torch.abs(_y - y), 1)))

def compute_err(output, y_test, comp_last: bool):
    o = output.cpu().detach().numpy()
    l = y_test.cpu().numpy()

    if comp_last: o, l = o[:, -1], l[:, -1]
    #--------------- Average error per region
    # error = np.mean(np.sum(abs(o - l), -1) / output.size(-1))
    assert o.shape == l.shape

    error = np.average(abs(o - l))
    # logger.info(f"[err] {font_green(error)}")

    return error

def compute_metrics(
        validation_hat, validation_real,
        test_hat, test_real, case_normalize_ratio
):
    assert validation_hat.shape == validation_real.shape
    mae_val, mae_test, rmse_val, rmse_test = [], [], [], []
    for i in range(len(validation_hat)):
        mae_val.append(MAELoss(validation_hat[i],validation_real[i]))
        rmse_val.append(RMSELoss(validation_hat[i],validation_real[i]))
    for i in range(len(test_hat)):
        mae_test.append(MAELoss(test_hat[i],test_real[i]))
        rmse_test.append(RMSELoss(test_hat[i],test_real[i]))

    mae_val = round(np.mean(np.array(mae_val) * case_normalize_ratio), 3)
    mae_test = round(np.mean(np.array(mae_test) * case_normalize_ratio), 3)
    rmse_val = round(np.mean(np.array(rmse_val) * case_normalize_ratio), 3)
    rmse_test = round(np.mean(np.array(rmse_test) * case_normalize_ratio), 3)

    return mae_val, rmse_val, mae_test, rmse_test
