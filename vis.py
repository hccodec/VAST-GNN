import zipfile, os, torch
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from eval import MAELoss, RMSELoss
from eval import compute_err, hits_at_k

from utils.logger import logger
from utils.utils import progress_indicator, rm_self_loops

def read_shape(country_code):
    # 读取地理数据
    shapes_zip_path = "data/mapfiles/gadm41_GBR_shp.zip"

    # with zipfile.ZipFile(shapes_zip_path) as z:
    #     names = z.namelist()
    # shps = [n for n in names if n.endswith("shp")]
    shp = "gadm41_GBR_3.shp"

    shape = gpd.read_file(f"zip://{shapes_zip_path}!/{shp}")

    # shapes[i].columns
    # [['GID_0', 'COUNTRY', 'geometry'],
    # ['GID_1', 'GID_0', 'COUNTRY', 'NAME_1', 'VARNAME_1', 'NL_NAME_1', 'TYPE_1', 'ENGTYPE_1', 'CC_1', 'HASC_1', 'ISO_1', 'geometry'],
    # ['GID_2', 'GID_0', 'COUNTRY', 'GID_1', 'NAME_1', 'NL_NAME_1', 'NAME_2', 'VARNAME_2', 'NL_NAME_2', 'TYPE_2', 'ENGTYPE_2', 'CC_2', 'HASC_2', 'geometry'],
    # ['GID_3', 'GID_0', 'COUNTRY', 'GID_1', 'NAME_1', 'NL_NAME_1', 'GID_2', 'NAME_2', 'NL_NAME_2', 'NAME_3', 'VARNAME_3', 'NL_NAME_3', 'TYPE_3', 'ENGTYPE_3', 'CC_3', 'HASC_3', 'geometry'],
    # ['GID_4', 'GID_0', 'COUNTRY', 'GID_1', 'NAME_1', 'GID_2', 'NAME_2', 'GID_3', 'NAME_3', 'NAME_4', 'VARNAME_4', 'TYPE_4', 'ENGTYPE_4', 'CC_4', 'geometry']]
    # shapes[1:]["NAME_1"]
    return shape

def generate_case_csv(regions, meta_data, country_name, x_case, y_case, y_hat, output_dir, dataset = "test"):
    '''
    将病例数据按天整理成 csv。每一列代表一天
    '''
    assert dataset in ["train", "val", "test"]

    # 准备病例数数据 (只取 测试集 效果最好的 batch)
    cases = torch.cat((x_case[:,:, :, [-1]], y_case), 1)

    # ** 由于数据是新增病例。所以要累加 **
    cases = torch.cumsum(cases, dim=1)
    y_hat = (y_hat + cases[:, [-1]])

    # 统计指标
    errs = [compute_err(y_case[i], y_hat[i], False) for i in range(10)]
    batch_err = errs.index(min(errs))

    cases = cases.cpu().numpy()
    y_hat = y_hat.cpu().numpy()

    data_cases = {
        "regions": [regions[i] for i in meta_data["selected_indices"][country_name]],
        **{f"cases_real_{i}": l for i, l in enumerate([list(cases[batch_err, d, :, -1]) for d in range(cases.shape[1])])},
        **{f"cases_hat": list(y_hat[batch_err, 0, :, 0])}
    }
    cases_df = pd.DataFrame(data_cases)

    path = os.path.join(output_dir, f"cases_{dataset}_{batch_err}.csv")
    cases_df.to_csv(path)
    return path

def generate_graphs_csv(regions_total, meta_data, country_name, x_mob, y_mob, adj_hat, output_dir, dataset = "test"):
    '''
    将流量数据按天整理成 csv。每个文件代表一天，每一天内是三元组（from,to,value），用于 ArcGIS 绘制
    '''
    assert dataset in ["train", "val", "test"]
    
    # 准备OD数数据 (只取 测试集 效果最好的 batch)
    mobs = torch.cat((x_mob, y_mob), 1)

    mobs = rm_self_loops(mobs)
    # 统计指标
    hits = [[hits_at_k(mobs[i, j], adj_hat[i, j]) for j in range(mobs.shape[1])] for i in range(mobs.shape[0])]
    hits = [sum(h) / len(h) for h in hits]
    batch_hits = hits.index(max(hits))

    mobs = mobs.cpu().numpy()
    adj_hat = adj_hat.cpu().numpy()
    region_names = [regions_total[i] for i in meta_data["selected_indices"][country_name]]

    graph_dirname = f"mobs_{dataset}_{batch_hits}"
    os.makedirs(os.path.join(output_dir, graph_dirname), exist_ok=True)
    
    # 读取 CSV 文件
    df = {
        2: pd.read_csv("data/mapfiles/coords_output/Export_Output_2.txt")[["NAME_2", "lat_2", "lon_2"]],
        3: pd.read_csv("data/mapfiles/coords_output/Export_Output_3.txt")[["NAME_2", "NAME_3", "lat_3", "lon_3"]],
        4: pd.read_csv("data/mapfiles/coords_output/Export_Output_4.txt")[["NAME_2", "NAME_3", "NAME_4", "lat_4", "lon_4"]]
    }
    region_coords = {
        22: df[2]["NAME_2"].to_list(),
        32: df[3]["NAME_2"].to_list(),
        33: df[3]["NAME_3"].to_list(),
        42: df[4]["NAME_2"].to_list(),
        43: df[4]["NAME_3"].to_list(),
        44: df[4]["NAME_4"].to_list(),
    }
    # 元素分别从三个级别的文件统计出现次数，以字符串的形式统计. 如 (0, 0, 0, 2, 0, 1) 表示 lv4-2 出现 2 次同时 lv4-4 出现 1 次
    region_level_counts = [tuple(region_coords[k].count(name) for k in region_coords) for name in region_names]
    # 统计区域的最高级别及其出现次数，及其索引
    same_elemenet_indices = lambda e, l: [_i for _i, _e in enumerate(l) if e == _e]
    levels = []  # 初始化一个空列表，用于存储结果

    # 遍历每个 region_level_counts 的索引和元组
    for i_region, tup in enumerate(region_level_counts):
        # 初始化标记和默认值
        found = False
        level_info = (None, None, None)

        # 遍历当前元组中的索引和对应的值
        for i, value in enumerate(tup):
            if value != 0:  # 检查值是否非零
                # 获取索引和相应的元素
                e, l = region_names[i_region], list(region_coords.values())[i]
                level_info = (i, value, [_i for _i, _e in enumerate(l) if e == _e])
                found = True
                break  # 找到后退出循环

        # 添加结果到 levels 列表
        levels.append(level_info)

    # 根据 levels 统计每个地区的经纬度
    locations = {}  # 初始化一个空列表，用于存储结果
    # 遍历每个 region_names 的索引和名称
    for i, region_name in enumerate(region_names):
        # 获取 levels 中的相关信息
        level = levels[i][0]
        count = levels[i][1]
        indices = levels[i][2]

        # 获取对应的坐标
        key_df, key_NAME = str(list(region_coords.keys())[level])
        csv_content = df[int(key_df)].iloc[indices]
        if csv_content.value.shape >= (2,):
            pass
        # 将区域名称和坐标添加到 locations 列表
        locations[region_name] = list(csv_content.values[0, -2:])

    paths = []
    for day in range(mobs.shape[1]):
        mob = mobs[batch_hits][day]
        data_mobs = {"from": [], "to": [], "value": []}
        for i in range(mob.shape[0]):
            for j in range(mob.shape[1]):
                if mob[i, j] == 0: continue
                # data_mobs["from"].append(region_names[i])
                # data_mobs["to"].append(region_names[j])
                data_mobs["O_x"], data_mobs["O_y"] = coords[region_names[i]]
                data_mobs["D_x"], data_mobs["D_y"] = coords[region_names[j]]
                data_mobs["value"].append(mob[i, j])
        mobs_df = pd.DataFrame(data_mobs)
        path = os.path.join(output_dir, graph_dirname, f"mobs_{dataset}_{batch_hits}_{day}.csv")
        mobs_df.to_csv(path)
        paths.append(path)
    return paths

def vis():
    # 准备模型推理结果
    country_code = "EN"

    from test import test
    res, meta_data, args = test(model_dir="results/tests_1022/exp_7_EN_ES_lambda_best/dataforgood/dynst_7_1_w7_s2_20241022204031",
            country_code="EN")
    
    country_name = meta_data["country_names"][meta_data["country_codes"].index(country_code)]

    regions = [v for k, v in meta_data["regions"].items()]

    # 单对英国，将其行政区域代码转为地名
    region_dict = pd.read_csv(
        os.path.join(args.data_dir, args.dataset, country_name, "region_names.csv")).set_index("code").to_dict()["region"]
    assert str(sorted(regions[0])) == str(sorted(region_dict.keys()))
    regions = [region_dict[r] for r in regions[0]]

    # 获取测试结果
    ((loss_train, y_real_train, y_hat_train, adj_real_train, adj_hat_train), \
        (loss_val, y_real_val, y_hat_val, adj_real_val, adj_hat_val), \
            (loss_test, y_real_test, y_hat_test, adj_real_test, adj_hat_test))  = res["outputs"]
    
    y_real_train, y_hat_train, adj_real_train, adj_hat_train = map(
        lambda x: x.cpu(), (y_real_train, y_hat_train, adj_real_train, adj_hat_train))
    y_real_val, y_hat_val, adj_real_val, adj_hat_val = map(
        lambda x: x.cpu(), (y_real_val, y_hat_val, adj_real_val, adj_hat_val))
    y_real_test, y_hat_test, adj_real_test, adj_hat_test = map(
        lambda x: x.cpu(), (y_real_test, y_hat_test, adj_real_test, adj_hat_test))
    
    # 获取预处理后的数据集
    x_case_train,   y_case_train,   x_mob_train,    y_mob_train,    indices_train   = meta_data["data"]["England"][0][0].dataset.tensors
    x_case_val,     y_case_val,     x_mob_val,      y_mob_val,      indices_val     = meta_data["data"]["England"][0][1].dataset.tensors
    x_case_test,    y_case_test,    x_mob_test,     y_mob_test,     indices_test    = meta_data["data"]["England"][0][2].dataset.tensors

    # 准备病例数数据 (只取 测试集 效果最好的 batch)
    vis_resdir = "results_visualization"
    logger.info(f"正在存储数据到目录 [{vis_resdir}] 中")
    os.makedirs(vis_resdir, exist_ok=True)
    case_path = generate_case_csv(regions, meta_data, country_name, x_case_test, y_case_test, y_hat_test, output_dir = vis_resdir)
    graph_paths = generate_graphs_csv(regions, meta_data, country_name, x_mob_test, y_mob_test, adj_hat_test, output_dir = vis_resdir)
    logger.info(f"数据已存储")


                                        
    # # 保存为GeoJSON文件
    # csv_file = "cases_data.csv"
    # with open(csv_file, 'w') as f:
    #     f.write(cases_df)


    # shape = read_shape("EN")
    # # 合并数据
    # # 假设你的地理数据中有一个名为 'name' 的列对应地区名称
    # merged = shape.merge(cases1_df, left_on='NAME_3', right_on='region', how='left')
    # # 将合并后的数据转换为GeoJSON格式
    # geojson_data = merged.to_xls()

    # # 绘制地图
    # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    # merged.plot(column='cases', ax=ax, legend=True,
    #             legend_kwds={'label': "Number of Cases"},
    #             cmap='OrRd', edgecolor='black')
    # plt.axis('off')
    # plt.title('COVID-19 Cases by Region', fontsize=20)
    # plt.show()

if __name__ == '__main__':
    vis()

