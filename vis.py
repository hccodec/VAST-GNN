import zipfile, os, torch
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

from utils.utils import progress_indicator

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
    
    ((loss_train, y_real_train, y_hat_train, adj_real_train, adj_hat_train), \
        (loss_val, y_real_val, y_hat_val, adj_real_val, adj_hat_val), \
            (loss_test, y_real_test, y_hat_test, adj_real_test, adj_hat_test))  = res["outputs"]
    
    x_case_train,   y_case_train,   x_mob_train,    y_mob_train,    indices_train   = meta_data["data"]["England"][0][0].dataset.tensors
    x_case_val,     y_case_val,     x_mob_val,      y_mob_val,      indices_val     = meta_data["data"]["England"][0][1].dataset.tensors
    x_case_test,    y_case_test,    x_mob_test,     y_mob_test,     indices_test    = meta_data["data"]["England"][0][2].dataset.tensors

    cases = torch.cat((x_case_test[:,:, :, [-1]], y_case_test), 1)

    # ** 由于数据是新增病例。所以要累加 **
    cases = torch.cumsum(cases, dim=1).cpu().numpy()
    y_hat_test = y_hat_test.cpu().numpy() + cases[:, [-1]]

    # 准备病例数数据 (只取 测试集 效果最好的 batch)
    data = {
        "regions": [regions[i] for i in meta_data["selected_indices"][country_name]],
        **{f"cases_real_{i}": l for i, l in enumerate([list(cases[0, :, :, -1][i]) for i in range(cases.shape[1])])},
        **{f"cases_hat": list(y_hat_test.cpu().numpy()[0, 0, :, 0])}
    }

    cases_df = pd.DataFrame(data)
    cases_df.to_csv("cases_data.csv")
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

