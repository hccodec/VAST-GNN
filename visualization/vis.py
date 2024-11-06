import zipfile, os, torch, re
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

from eval import MAELoss, RMSELoss
from eval import compute_err, hits_at_k

from utils.logger import logger
from utils.utils import get_country, progress_indicator, rm_self_loops
from visualization.utils.utils import get_region_relation, fetch_from_exported_txt


df_map_filenames = {
    "FR": "visualization/mapfiles/coords_output/gadm40/gadm40_FRA_3.csv",
    "ES": "visualization/mapfiles/coords_output/gadm40/gadm40_ESP_3.csv",
    "IT": "visualization/mapfiles/coords_output/gadm40/gadm40_ITA_2.csv",
    "EN": "visualization/mapfiles/coords_output/gadm40/gadm40_GBR_3_Clip.csv"
}

# def read_shape(country_code):
#     # 读取地理数据
#     shapes_zip_path = "../data/mapfiles/gadm41_GBR_shp.zip"

#     # with zipfile.ZipFile(shapes_zip_path) as z:
#     #     names = z.namelist()
#     # shps = [n for n in names if n.endswith("shp")]
#     shp = "gadm41_GBR_3.shp"

#     shape = gpd.read_file(f"zip://{shapes_zip_path}!/{shp}")

#     # shapes[i].columns
#     # [['GID_0', 'COUNTRY', 'geometry'],
#     # ['GID_1', 'GID_0', 'COUNTRY', 'NAME_1', 'VARNAME_1', 'NL_NAME_1', 'TYPE_1', 'ENGTYPE_1', 'CC_1', 'HASC_1', 'ISO_1', 'geometry'],
#     # ['GID_2', 'GID_0', 'COUNTRY', 'GID_1', 'NAME_1', 'NL_NAME_1', 'NAME_2', 'VARNAME_2', 'NL_NAME_2', 'TYPE_2', 'ENGTYPE_2', 'CC_2', 'HASC_2', 'geometry'],
#     # ['GID_3', 'GID_0', 'COUNTRY', 'GID_1', 'NAME_1', 'NL_NAME_1', 'GID_2', 'NAME_2', 'NL_NAME_2', 'NAME_3', 'VARNAME_3', 'NL_NAME_3', 'TYPE_3', 'ENGTYPE_3', 'CC_3', 'HASC_3', 'geometry'],
#     # ['GID_4', 'GID_0', 'COUNTRY', 'GID_1', 'NAME_1', 'GID_2', 'NAME_2', 'GID_3', 'NAME_3', 'NAME_4', 'VARNAME_4', 'TYPE_4', 'ENGTYPE_4', 'CC_4', 'geometry']]
#     # shapes[1:]["NAME_1"]
#     return shape

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
    errs = [compute_err(y_case[i], y_hat[i], False) for i in range(10)]
    batch_err = errs.index(min(errs))

    cases = cases.cpu().numpy()
    y_hat = y_hat.cpu().numpy()

    cases_df = pd.DataFrame({
        "label": regions,
        **{f"cases_real_{i}": l for i, l in enumerate([list(cases[batch_err, d, :, -1]) for d in range(cases.shape[1])])},
        **{f"cases_hat": list(y_hat[batch_err, 0, :, 0])}
    })

    return f"cases_{dataset}_{batch_err}.csv", cases_df

def generate_mob_relations(locations, regions, x_mob, y_mob, adj_hat, dataset = "test"):
    '''
    将流量数据按天整理成 csv。每个文件代表一天，每一天内是三元组（from,to,value），用于 ArcGIS 绘制
    
    方案：传入 locations 数据 (包含字段：FID, label, lat, lon) -> 使用 regions 筛选 label 并对筛选结果进行经纬度统计，使用字典处理同名
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

    region_names = regions

    OD_data = []
    for day in range(mobs.shape[1]):
        mob = mobs[batch_hits][day]
        data_mobs = []
        for i_O in range(mob.shape[0]):
            for i_D in range(mob.shape[1]):
                if mob[i_O, i_D] == 0: continue
                # data_mobs["from"].append(region_names[i])
                # data_mobs["to"].append(region_names[j])
                dic = dict(zip(
                    ("O_x", "O_y", "D_x", "D_y", "v"),
                    (*locations[region_names[i_O]], *locations[region_names[i_D]], mob[i_O, i_D])))
                data_mobs.append(dic)
        mobs_df = pd.DataFrame(data_mobs)
        # path = os.path.join(output_dir, graph_dirname, f"mobs_{dataset}_{batch_hits}_{day}.csv")
        # mobs_df.to_csv(path)
        OD_data.append((f"mobs_{dataset}_{batch_hits}_{day}.csv", mobs_df))

    return f"mobs_{dataset}_{batch_hits}", OD_data

def vis(test_dir = "1022", exp = "7", country = "EN"):
    
    logger.info("=" * 50)
    logger.info(" " * ((50 - len(f"开始可视化 {country}")) // 2) + f"开始可视化 {country}")
    logger.info("=" * 50)

    # 提取对应模型
    models = [os.path.join(dirpath, filename) for dirpath, dirnames, filenames in os.walk(f"results/tests_{test_dir}")
     for filename in filenames if filename.endswith("best.pth") and country in filename and f"exp_{exp}" in dirpath]
    
    for model in models:
        logger.info(f"开始加载模型 {model}")
        xdays, ydays, window, shift = re.search('dynst_(\d+)_(\d+)_w(\d+)_s(\d+)', model).groups()

        # 准备模型推理结果
        from test import test
        res, meta_data, args = test(fn_model=model, country=country)
        
        country_code, country_name = get_country(country, meta_data)

        regions = meta_data["regions"][country_name]

        if country_code == 'EN':
        # 单对英国，将其行政区域代码转为地名
            region_dict = pd.read_csv(
                os.path.join(args.data_dir, args.dataset, country_name, "region_names.csv")).set_index("code").to_dict()["region"]
            assert str(sorted(regions)) == str(sorted(region_dict.keys()))
            regions = [region_dict[r] for r in regions]

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
        x_case_train,   y_case_train,   x_mob_train,    y_mob_train,    indices_train   = meta_data["data"][country_name][0][0].dataset.tensors
        x_case_val,     y_case_val,     x_mob_val,      y_mob_val,      indices_val     = meta_data["data"][country_name][0][1].dataset.tensors
        x_case_test,    y_case_test,    x_mob_test,     y_mob_test,     indices_test    = meta_data["data"][country_name][0][2].dataset.tensors


        # 准备病例数数据 (只取 测试集 效果最好的 batch)
        df_map = pd.read_csv(df_map_filenames[country_code])
        relations, locations, unmatched_regions = get_region_relation(df_map, regions)
        
        regions_observed = [regions[i] for i in meta_data["selected_indices"][country_name]]
        fn_case, case_relation = generate_case_relations(regions_observed, x_case_test, y_case_test, y_hat_test)
        dir_mob, mob_relation = generate_mob_relations(locations, regions_observed, x_mob_test, y_mob_test, adj_hat_test)

        vis_resdir = os.path.join(f"visualization/results_visualization", "x{}_y{}_w{}_s{}".format(xdays, ydays, window, shift), country)
        logger.info(f"正在存储数据到目录 [{vis_resdir}] 中")
        os.makedirs(vis_resdir, exist_ok=True)
        # case
        case_relation.to_csv(os.path.join(vis_resdir, fn_case))
        # location
        locations.T.to_csv(os.path.join(vis_resdir, f"{country_name}_locations.csv"), index=False)
        # mob
        os.makedirs(os.path.join(vis_resdir, dir_mob), exist_ok=True)
        for fn, mob_data in mob_relation:
            mob_data.to_csv(os.path.join(vis_resdir, dir_mob, fn))

        logger.info(f"数据已存储")

    logger.info("=" * 50)
    logger.info(" " * ((50 - len(f"结束可视化 {country}")) // 2) +
                f"结束可视化 {country}")
    logger.info("=" * 50)


                                        
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
    test_dir = "1022"
    exp = "7"
    vis(test_dir, exp, "EN")
    vis(test_dir, exp, "IT")
    vis(test_dir, exp, "ES")
    vis(test_dir, exp, "FR")
    vis(test_dir, exp, "NZ")

