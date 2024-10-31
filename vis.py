import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

# 读取地理数据
shapes_zip_path = "data/mapfiles/gadm41_GBR_shp.zip"
shapefile_name = "gadm41_GBR_1.shp"
shapes = gpd.read_file(f"zip://{shapes_zip_path}!/{shapefile_name}")

# 准备病例数数据（示例）
data = {
    'region': ['Region_A', 'Region_B', 'Region_C'],  # 替换为实际地区名称
    'cases': [100, 200, 300]  # 替换为实际病例数
}
cases_df = pd.DataFrame(data)

# 合并数据
# 假设你的地理数据中有一个名为 'name' 的列对应地区名称
merged = shapes.merge(cases_df, left_on='name', right_on='region', how='left')

# 绘制地图
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
merged.plot(column='cases', ax=ax, legend=True,
            legend_kwds={'label': "Number of Cases"},
            cmap='OrRd', edgecolor='black')
plt.axis('off')
plt.title('COVID-19 Cases by Region', fontsize=20)
plt.show()
