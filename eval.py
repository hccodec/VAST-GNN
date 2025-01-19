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
    o = output.cpu().detach().numpy() if hasattr(output, 'cpu') else output
    l = y_test.cpu().numpy() if hasattr(y_test, 'cpu') else y_test

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


def hits_at_k(A, A_hat, k=10, threshold_ratio=0.5):
    assert A.shape == A_hat.shape, "A and A_hat must have the same shape"

    A, A_hat = A.clone(), A_hat.clone()

    # A = torch.where(A > threshold_ratio, 1, 0)
    # A_hat = torch.where(A_hat > threshold_ratio, 1, 0)

    N = A.shape[0]

    # 将 A_hat 拉直并根据得分排序，忽略对角线
    sorted_indices = torch.argsort(A_hat.view(-1), descending=True)
    mask = (1 - torch.eye(N)).view(-1)
    # 只保留非对角线部分的排序索引
    non_diag_indices = sorted_indices[mask[sorted_indices] == 1]
    # 取前 k% 个预测的边索引
    top_k_indices = non_diag_indices[: int(k / 100 * len(non_diag_indices))]

    indices_A_hat_top_k = torch.tensor([(i.item(), j.item()) for i, j in zip(top_k_indices // N, top_k_indices % N)])

    A_non_diag = A * (1 - torch.eye(A.shape[0]))
    indices_A = A_non_diag.nonzero()
    # 统计 Hits@k 中的命中次数
    # hits = 0
    # for i, j in zip(i_indices, j_indices):
    #     if A[i, j] > 0:  # 如果在真实邻接矩阵 A 中存在这条边
    #         hits += 1
    # # 计算 Hits@k (命中次数 / k)
    # hits_k = hits / A.count_zero().item()

    # A 中所有边，是否在 A_hat 的前 10%
    hits = sum([pos_edge in indices_A_hat_top_k for pos_edge in indices_A])
    hits_k = hits / len(indices_A)

    return hits_k

@torch.no_grad()
def compute_hits_at_k(A_hat_batch, A_batch, k=10, threshold_ratio=0.5):
    assert A_hat_batch.shape == A_batch.shape
    A_hat_batch, A_batch = A_hat_batch.clone().cpu(), A_batch.clone().cpu()

    # A_hat_batch, A_batch = min_max_adj(A_hat_batch), min_max_adj(A_batch)

    total_hits = []
    batch_size, num_days, n, _ = A_hat_batch.shape
    for b in range(batch_size):
        for d in range(num_days):
            A, A_hat = A_batch[b, d], A_hat_batch[b, d]
            hits_per_user = hits_at_k(A, A_hat, k, threshold_ratio)

            # 累加到总 hits@k 计数器
            total_hits.append(hits_per_user)

        # 计算平均 hits@k
        average_hits_at_k = np.mean(total_hits)
        # average_hits_at_k = average_hits_at_k.item()
        return average_hits_at_k
