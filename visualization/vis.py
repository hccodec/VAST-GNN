import os, re
import pandas as pd
import matplotlib.pyplot as plt

from utils.logger import logger
from utils.utils import font_yellow, get_country

import torch

from eval import compute_err, hits_at_k
from utils.utils import rm_self_loops

def generate_case_relations(regions, x_case, y_case, y_hat, dataset = "test"):
    '''
    将病例数据按天整理成 csv。每一列代表一天
    '''
    assert dataset in ["train", "val", "test"]

    # 准备病例数数据
    cases = torch.cat((x_case[:,:, :, [-1]], y_case), 1)

    # ** 由于数据是新增病例。所以要累加 **
    cases = torch.cumsum(cases, dim=1)
    y_hat = (y_hat + cases[:, [-1]])

    # 统计指标，并取得效果最好的 batch
    errs = [compute_err(y_case[i], y_hat[i], False) for i in range(y_case.shape[0])]
    batch_err = errs.index(min(errs))

    cases = cases.cpu().numpy()
    y_hat = y_hat.cpu().numpy()

    cases_df = pd.DataFrame({
        "label": regions,
        **{f"cases_real_{i}": l for i, l in enumerate([list(cases[batch_err, d, :, -1]) for d in range(cases.shape[1])])},
        **{f"cases_hat": list(y_hat[batch_err, 0, :, 0])}
    })
    # cases_df = pd.merge(relations, cases_df, on='label', how='left')[["FID", *cases_df.columns]]
    return f"cases_{dataset}_{batch_err}.csv", cases_df

def generate_mob_relations(regions, x_mob, y_mob, hats, dataset = "test"):
    '''
    将流量数据按天整理成 csv。每个文件代表一天，每一天内是三元组（from,to,value），用于 ArcGIS 绘制
    
    方案：传入 locations 数据 (包含字段：FID, label, lat, lon) -> 使用 regions 筛选 label 并对筛选结果进行经纬度统计，使用字典处理同名
    '''
    assert dataset in ["train", "val", "test"]

    # locations = locations.T # 方便将地名作为索引
    
    # 准备OD数数据 (只取 测试集 效果最好的 batch)
    mobs = torch.cat((x_mob, y_mob), 1)
    mobs = rm_self_loops(mobs)

    # 统计指标
    hits = [[hits_at_k(mobs[i, j], hats[i, j]) for j in range(mobs.shape[1])] for i in range(mobs.shape[0])]
    hits = [sum(h) / len(h) for h in hits]
    batch_hits = hits.index(max(hits))

    mobs = mobs[batch_hits].cpu().numpy()
    hats = hats[batch_hits].cpu().numpy()

    OD_data = []
    for day in range(mobs.shape[0]):
        mob = mobs[day]
        hat = hats[day]

        OD_data_day = []
        for i_O in range(mob.shape[0]):
            for i_D in range(mob.shape[1]):
                if mob[i_O, i_D] != 0 or hat[i_O, i_D] != 0:
                    OD_data_day.append(dict(zip(
                        ("O", "D", "value_real", "value_hat"),
                        (regions[i_O], regions[i_D], mob[i_O, i_D], hat[i_O, i_D]))))
        df_real = pd.DataFrame.from_records(OD_data_day)
        OD_data.append((f"adj_{dataset}_{batch_hits}_{day}.csv", df_real))

    return OD_data

def vis(test_dir = "1022", exp = "7", country = "EN", save_dir = ''):
    '''
    可视化模型的预测结果
    :param test_dir: 测试目录
    :param exp: 实验编号
    :param country: 国家代码
    :param save_dir: 保存目录
    :return:
    '''
    # 读取配置文件
    logger.info(font_yellow(f"开始可视化 {country}"))

    # 提取对应模型
    # models = [os.path.join(dirpath, filename) for dirpath, dirnames, filenames in os.walk('.') if '' in dirpath and f"tests_{test_dir}" in dirpath
    #  for filename in filenames if filename.endswith("best.pth") and country in filename and f"exp_{exp}" in dirpath]
    save_dir = f"{test_dir}_{exp}" if save_dir == '' else save_dir
    models = []
    for dirpath, _, filenames in os.walk('.'):
        # 检查目录路径条件
        if (f"tests_{test_dir}" in dirpath or f"test_{test_dir}" in dirpath) and f"exp_{exp}" in dirpath and f"s2" in dirpath:
            for filename in filenames:
                # 检查文件名条件
                if filename.endswith("best.pth") and country in filename:
                    full_path = os.path.join(dirpath, filename)
                    models.append(full_path)
    return vis1(models, country, save_dir)

def vis1(models, country = "EN", save_dir = ''):
    '''
    可视化模型的预测结果
    :param models: 模型路径
    :param country: 国家代码
    :param save_dir: 保存目录
    :return:
    '''
    for model in models:
        logger.info(f"开始加载模型 {model}")
        model_str, xdays, ydays, window, shift = re.search('([^/]+)_(\d+)_(\d+)_w(\d+)_s(\d+)', model).groups()

        # 准备模型推理结果
        from utils.test import test
        res, meta_data, args = test(model_path=model, device='cpu')
        
        country_code, country_name = get_country(country, meta_data)

        regions = meta_data["regions"][country_name]

        if country_code == 'EN':
        # 单对英国，将其行政区域代码转为地名
            region_dict = pd.read_csv("visualization/region_names_EN.csv").set_index("code").to_dict()["region"]
            assert str(sorted(regions)) == str(sorted(region_dict.keys()))
            regions = [region_dict[r] for r in regions]

        # 获取测试结果
        ((loss_train, y_real_train, y_hat_train, adj_real_train, adj_hat_train), \
            (loss_val, y_real_val, y_hat_val, adj_real_val, adj_hat_val), \
                (loss_test, y_real_test, y_hat_test, adj_real_test, adj_hat_test))  = res["outputs"]
        
        y_real_train, y_hat_train, adj_real_train, adj_hat_train = map(
            lambda x: x.cpu() if 'cpu' in dir(x) else x, (y_real_train, y_hat_train, adj_real_train, adj_hat_train))
        y_real_val, y_hat_val, adj_real_val, adj_hat_val = map(
            lambda x: x.cpu() if 'cpu' in dir(x) else x, (y_real_val, y_hat_val, adj_real_val, adj_hat_val))
        y_real_test, y_hat_test, adj_real_test, adj_hat_test = map(
            lambda x: x.cpu() if 'cpu' in dir(x) else x, (y_real_test, y_hat_test, adj_real_test, adj_hat_test))
        
        # 获取预处理后的数据集
        x_case_train, y_case_train, x_mob_train, y_mob_train, indices_train = meta_data["data"][country_name][0][0].dataset.tensors
        x_case_val,   y_case_val,   x_mob_val,   y_mob_val,   indices_val   = meta_data["data"][country_name][0][1].dataset.tensors
        x_case_test,  y_case_test,  x_mob_test,  y_mob_test,  indices_test  = meta_data["data"][country_name][0][2].dataset.tensors

        # 准备病例数数据 (取 测试集 效果最好的 batch)
        regions_observed = [regions[i] for i in meta_data["selected_indices"][country_name]]
        fn_case, case_relation = generate_case_relations(regions_observed, x_case_test, y_case_test, y_hat_test, "test")
        if not adj_hat_test == []: mob_relation = generate_mob_relations(regions_observed, x_mob_test, y_mob_test, adj_hat_test, "test")

        vis_resdir = os.path.join(f"visualization/results_visualization", save_dir,
                                  "x{}_y{}_w{}_s{}_{}".format(xdays, ydays, window, shift, model_str), country)
        logger.info(f"正在存储数据到目录 [{vis_resdir}] 中")
        os.makedirs(vis_resdir, exist_ok=True)
        # case
        case_relation.to_csv(os.path.join(vis_resdir, fn_case), index=False)
        # # relations 包括所有的、unobserved、observed的
        # relations.to_csv(os.path.join(vis_resdir, f"{country_name}_relations.csv"), index=False)
        # mob
        if not adj_hat_test == []:
            for fn, mob_data in mob_relation:
                mob_data.to_csv(os.path.join(vis_resdir, fn), index=False)

        logger.info(f"数据已存储")
    logger.info(font_yellow(f"结束可视化 {country}"))

if __name__ == '__main__':
    test_dir = "0209_seed7"
    shifts = '2'

    vis(test_dir, '1_baselines_lstm_50/', "h1n1", "0209_seed7_lstm")
    vis(test_dir, '1_baselines_lstm_50/', "h3n2", "0209_seed7_lstm")
    vis(test_dir, '1_baselines_lstm_50/', "BV", "0209_seed7_lstm")
    vis(test_dir, '1_baselines_lstm_50/', "BY", "0209_seed7_lstm")

    vis(test_dir, '1_baselines_mpnn_lstm_50/', "h1n1", "0209_seed7_mpnn_lstm")
    vis(test_dir, '1_baselines_mpnn_lstm_50/', "h3n2", "0209_seed7_mpnn_lstm")
    vis(test_dir, '1_baselines_mpnn_lstm_50/', "BV", "0209_seed7_mpnn_lstm")
    vis(test_dir, '1_baselines_mpnn_lstm_50/', "BY", "0209_seed7_mpnn_lstm")

    vis(test_dir, '1_baselines_mpnn_tl_50/', "h1n1", "0209_seed7_mpnn_tl")
    vis(test_dir, '1_baselines_mpnn_tl_50/', "h3n2", "0209_seed7_mpnn_tl")
    vis(test_dir, '1_baselines_mpnn_tl_50/', "BV", "0209_seed7_mpnn_tl")
    vis(test_dir, '1_baselines_mpnn_tl_50/', "BY", "0209_seed7_mpnn_tl")


    # vis(test_dir, '2_50_graph_lambda_7/', "h1n1", "0209_seed7_dynst")
    # vis(test_dir, '2_50_graph_lambda_6/', "h3n2", "0209_seed7_dynst")
    # vis(test_dir, '2_50_graph_lambda_0/', "BV",   "0209_seed7_dynst")
    # vis(test_dir, '2_50_graph_lambda_2/', "BY",   "0209_seed7_dynst")
    # from best_results import paths
    
    # paths[f'o{node_observation_ratio}'].sort_index().loc[(ydays, country_code)].iterrows()

# if __name__ == '__main__':
#     test_dir = "0106_all"
#     exp = '2'
#     vis(test_dir, exp, "EN")
#     vis(test_dir, exp, "FR")
#     vis(test_dir, exp, "IT")
#     vis(test_dir, exp, "ES")

    # vis(test_dir, exp, "NZ")

