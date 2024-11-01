import zipfile, os
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

from utils.utils import progress_indicator

def read_shapes():
    # 读取地理数据
    shapes_zip_path = "data/mapfiles/gadm41_GBR_shp.zip"

    with zipfile.ZipFile(shapes_zip_path) as z:
        names = z.namelist()
    shps = [n for n in names if n.endswith("shp")]

    qbar = progress_indicator(shps, leave=False)
    shapes = []

    for i in range(len(shps)):
        qbar.set_description(f"正在从压缩文件 {os.path.basename(shapes_zip_path)} 读取地理文件 {shps[i]}")
        shapes.append(gpd.read_file(f"zip://{shapes_zip_path}!/{shps[i]}"))
        qbar.update()

    # shapes[i].columns
    # [['GID_0', 'COUNTRY', 'geometry'],
    # ['GID_1', 'GID_0', 'COUNTRY', 'NAME_1', 'VARNAME_1', 'NL_NAME_1', 'TYPE_1', 'ENGTYPE_1', 'CC_1', 'HASC_1', 'ISO_1', 'geometry'],
    # ['GID_2', 'GID_0', 'COUNTRY', 'GID_1', 'NAME_1', 'NL_NAME_1', 'NAME_2', 'VARNAME_2', 'NL_NAME_2', 'TYPE_2', 'ENGTYPE_2', 'CC_2', 'HASC_2', 'geometry'],
    # ['GID_3', 'GID_0', 'COUNTRY', 'GID_1', 'NAME_1', 'NL_NAME_1', 'GID_2', 'NAME_2', 'NL_NAME_2', 'NAME_3', 'VARNAME_3', 'NL_NAME_3', 'TYPE_3', 'ENGTYPE_3', 'CC_3', 'HASC_3', 'geometry'],
    # ['GID_4', 'GID_0', 'COUNTRY', 'GID_1', 'NAME_1', 'GID_2', 'NAME_2', 'GID_3', 'NAME_3', 'NAME_4', 'VARNAME_4', 'TYPE_4', 'ENGTYPE_4', 'CC_4', 'geometry']]
    # shapes[1:]["NAME_1"]
    return shapes

def test_model():
    # 准备模型推理结果
    from test import test
    res, meta_data = test(model_dir="results/tests_1022/exp_7_EN_ES_lambda_best/dataforgood/dynst_7_1_w7_s2_20241022204031",
            country_code="EN")
    
    ((loss_train, y_real_train, y_hat_train, adj_real_train, adj_hat_train), \
        (loss_val, y_real_val, y_hat_val, adj_real_val, adj_hat_val), \
            (loss_test, y_real_test, y_hat_test, adj_real_test, adj_hat_test))  = res["outputs"]

# 准备病例数数据（示例）
data = {
    # 'region': ['Region_A', 'Region_B', 'Region_C'],  # 替换为实际地区名称
    # 'cases': [100, 200, 300]  # 替换为实际病例数
}
cases_df = pd.DataFrame(data)

# 合并数据
# 假设你的地理数据中有一个名为 'name' 的列对应地区名称
merged = shapes.merge(cases_df, left_on='NAME_1', right_on='region', how='left')

# 绘制地图
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
merged.plot(column='cases', ax=ax, legend=True,
            legend_kwds={'label': "Number of Cases"},
            cmap='OrRd', edgecolor='black')
plt.axis('off')
plt.title('COVID-19 Cases by Region', fontsize=20)
plt.show()
