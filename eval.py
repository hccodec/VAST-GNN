import torch
from scipy import stats
import numpy as np
from tensorboard.plugins.debugger_v2.debug_data_provider import ALERTS_BLOB_TAG_PREFIX

from utils.logger import logger
from utils.utils import catch, font_green, min_max_adj

metrics_labels = ["MAE_val", "RMSE_val", "MAE_test", "RMSE_test"]

@catch("出现错误，无法计算相关性")
def compute_correlation(hat_data, real_data):
    try:
        hat_sum = [float(torch.sum(hat_data[i][-1])) for i in range(len(hat_data))]
        real_sum = [float(torch.sum(real_data[i][-1])) for i in range(len(real_data))]
        correlation = stats.pearsonr(hat_sum, real_sum)
        return correlation
    except Exception as e:
        print(f"出现错误，无法计算相关性: {e}")
        return [-1]

def compute_err(output, y_test, comp_last: bool):
    o = output.cpu().detach().numpy()
    l = y_test.cpu().numpy()

    if comp_last: o, l = o[:, -1], l[:, -1]
    #--------------- Average error per region
    # error = np.mean(np.sum(abs(o - l), -1) / output.size(-1))
    assert o.shape == l.shape

    error = np.average(abs(o - l))
    return error

RMSELoss = lambda _y, y: float(torch.sqrt(torch.mean((_y - y) ** 2)))
MAELoss = lambda _y, y: float(torch.mean(torch.div(torch.abs(_y - y), 1)))

def compute_mae_rmse(y_hat, y):
    assert  y_hat.shape == y.shape
    mae = round(np.mean([MAELoss(y_hat[i], y[i]) for i in range(len(y))]), 3)
    rmse = round(np.mean([RMSELoss(y_hat[i], y[i]) for i in range(len(y))]), 3)
    return mae, rmse


def hits_at_k(A, A_hat, k, threshold_ratio):
    """
    计算 Hits@k 指标，用于图链接预测的评估（适用于有向图）。

    参数:
    - A: torch.tensor, 真实邻接矩阵 (N x N)，0 或 1 表示是否有边
    - A_hat: torch.tensor, 预测邻接矩阵 (N x N)，小数表示边存在的概率
    - k: int, 取前 k 个预测边进行 Hits@k 评估
    - threshold_ratio: float, 用于判断 A_hat 中的概率是否足够高以认为存在边

    返回:
    - hits_k: float, Hits@k 值
    """
    # 确保 A 和 A_hat 是方阵
    assert A.shape == A_hat.shape, "A and A_hat must有 the same shape"

    A = torch.where(A > threshold_ratio, 1, 0)
    A_hat = torch.where(A_hat > threshold_ratio, 1, 0)

    N = A.shape[0]

    # 将 A_hat 拉直并根据得分排序，忽略对角线
    A_hat_flat = A_hat.view(-1)
    sorted_indices = torch.argsort(A_hat_flat, descending=True)

    # 构建一个忽略对角线的 mask
    mask = torch.ones_like(A_hat)
    mask.fill_diagonal_(0)  # 忽略对角线
    mask_flat = mask.view(-1)  # 将 mask 拉平

    # 只保留非对角线部分的排序索引
    non_diag_indices = sorted_indices[mask_flat[sorted_indices] == 1]
    # 取前 k 个预测的边索引
    top_k_indices = non_diag_indices[:k]

    # 将 top_k_indices 转换为矩阵中的 (i, j) 对应的行列索引
    i_indices = top_k_indices // N
    j_indices = top_k_indices % N

    # 统计 Hits@k 中的命中次数
    hits = 0
    for i, j in zip(i_indices, j_indices):
        if A[i, j] == 1:  # 如果在真实邻接矩阵 A 中存在这条边
            hits += 1

    # 计算 Hits@k (命中次数 / k)
    hits_k = hits / k
    return hits_k

@torch.no_grad()
def compute_hits_at_k(A_hat_batch, A_batch, k=10, threshold_ratio=0.5):
    assert A_hat_batch.shape == A_batch.shape
    A_hat_batch, A_batch = A_hat_batch.clone().cpu(), A_batch.clone().cpu()

    A_hat_batch, A_batch = min_max_adj(A_hat_batch), min_max_adj(A_batch)

    total_hits = 0
    batch_size, num_days, n, _ = A_hat_batch.shape
    hits_tensor = torch.zeros(batch_size, num_days, n, dtype=torch.float32)
    for b in range(batch_size):
        for d in range(num_days):
            A, A_hat = A_batch[b, d], A_hat_batch[b, d]
            hits_per_user = hits_at_k(A, A_hat, k, threshold_ratio)

            # 累加到总 hits@k 计数器
            total_hits += hits_per_user

            # 记录每个用户是否达到阈值
            hits_tensor[b, d] = hits_per_user

        # 计算平均 hits@k
        average_hits_at_k = total_hits / (batch_size * num_days * k)
        # average_hits_at_k = average_hits_at_k.item()
        return average_hits_at_k
