import pandas as pd
import numpy as np
import os

from utils.logger import logger
from utils.utils import font_green
import re

char_map = {
    "é": "e", "è": "e", "ê": "e", "ë": "e",
    "á": "a", "à": "a", "â": "a", "ä": "a", "ã": "a", "å": "a",
    "î": "i", "ï": "i", "í": "i", "ì": "i",
    "ô": "o", "ö": "o", "ó": "o", "ò": "o", "õ": "o",
    "û": "u", "ü": "u", "ù": "u", "ú": "u", "ů": "u",
    "ç": "c", "œ": "oe", "æ": "ae", "ñ": "n",
    "Á": "A", "É": "E", "Í": "I", "Ó": "O", "Ú": "U", "Ç": "C", "Ñ": "N",
    "-": "_", "–": "_", "—": "_", " ": "_", "_": "_", "'": "_",
    ".": "", ",": "", ";": "", ":": "", "(": "_", ")": "_",
    "[": "_", "]": "_", "{": "_", "}": "_",
    "&": "_and_", "/": "_", "\\": "_", "%": "_percent_"
}

char_trans = lambda x: str(x).strip().lower().translate(str.maketrans(char_map))

region_map = {
    "bizkaia": "vizcaya",
    "gipuzkoa": "guipuzcoa",
    "araba/alava": "alava",
    "bolzano_bozen": "bolzano",
    "firenze": "florence",
    'la_aquila': "l_aquila",
    'mantova': "mantua",
    'padova': "padua",
    'reggio_calabria': "reggio_di_calabria",
    'reggio_emilia': "reggio_nell_emilia",
    'siracusa': "syracuse"
}

def f_trans(x):
    # 首先处理字符替换
    x = char_trans(x)
    
    # 然后使用正则表达式替换地名
    for k, v in region_map.items():
        x = re.sub(re.escape(char_trans(k)), char_trans(v), x)
        x = re.sub(r'_+', '_', x)
    return x

def convert_region_lst(s):
    '''
    此函数处理地名。首先转换成小写。再将法语字母转化成英语字母
    '''
    return s if isinstance(s, str) else [f_trans(_s) for _s in s] if isinstance(s, list) else s

def get_region_relation(df_map, regions):
    '''
    获取 地图上与数据集上对应关系，并返回一个 DataFrame
    @param df_map: 地图数据
    @param regions: 数据集上的地区名称
    @return: DataFrame, regions_set, labels_set
    '''

    if isinstance(df_map, str) and os.path.exists(df_map) and os.path.isfile(df_map):
        df_map = pd.read_csv(df_map)

    name2 = [str(n) for n in df_map["NAME_2"]]
    name3 = [str(n) for n in df_map["NAME_3"]] if "NAME_3" in df_map.columns else []

    # 将 name2 中 nan 的值替换为 name3 对应值
    name2 = [(name3[i] if name2[i] == 'nan' else name2[i]) for i in range(len(name2))]

    converted_name2 = convert_region_lst(name2)
    converted_name3 = convert_region_lst(name3)
    converted_regions = convert_region_lst(regions)


    # 将 name2 中不属于数据集的元素赋为空字符串，以此作为 label。
    # 空字符串表示 name2 未匹配到。label 表示地图上的区域对应于数据集上的区域名称
    # label 存储的是 df_map 中的地理名称，即 NAME_2 或 NAME_3
    label = [(regions[converted_regions.index(converted_name2[i])]
              if converted_name2[i] in converted_regions else np.nan) for i in range(len(name2))]

    # 使用 name3 进一步匹配 regions
    unmatch_regions = []
    for i in range(len(regions)):
        if converted_regions[i] in converted_name2: continue
        # 若 name3 里有，则补充至 label
        if converted_regions[i] in converted_name3:
            index = converted_name3.index(converted_regions[i])
            label[index] = regions[i]
        # 否则统计到 unmatch_region，表示未在地图上找到对应地区
        if converted_regions[i] not in converted_name3:
            unmatch_regions.append(regions[i])

    logger.info(f"地图上多的区域有 {label.count('')}/{len(label)} 个")
    logger.info(f"数据集多的区域有 {len(unmatch_regions)}/{len(regions)} 个"
                if len(unmatch_regions) else font_green("数据集所有区域均已匹配"))

    relation = pd.DataFrame({
        "FID": df_map["FID"],
        "label": label,
        "lat": df_map["lat"],
        "lon": df_map["lon"]
    })

    locations = relation.groupby("label").agg({'lat': "mean", "lon": "mean"})

    return relation, locations, unmatch_regions

def fetch_from_exported_txt(region_names):#

    # 读取 CSV 文件
    df = {
        # 2: pd.read_csv("data/mapfiles/coords_output/Export_Output_2.txt")[["NAME_2", "lat_2", "lon_2"]],
        3: pd.read_csv("../data/mapfiles/coords_output/Export_Output_3.txt")[["NAME_2", "NAME_3", "NAME", "lat", "lon"]],
        # 4: pd.read_csv("data/mapfiles/coords_output/Export_Output_4.txt")[["NAME_2", "NAME_3", "NAME_4", "lat_4", "lon_4"]]
    }
    region_coords = {
        # 22: df[2]["NAME_2"].to_list(),
        32: df[3]["NAME_2"].to_list(),
        33: df[3]["NAME_3"].to_list(),
        # 42: df[4]["NAME_2"].to_list(),
        # 43: df[4]["NAME_3"].to_list(),
        # 44: df[4]["NAME_4"].to_list(),
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

        # 找到 tup 中的首个非零元素，则匹配到指定行
        for i, value in enumerate(tup):
            if value != 0:  # 检查值是否非零
                # 获取索引和相应的元素
                e, l = region_names[i_region], list(region_coords.values())[i]
                level_info = (i, value, [_i for _i, _e in enumerate(l) if e == _e])
                found = True
                break  # 找到后退出循环

        # 添加结果到 levels 列表
        levels.append(level_info)

    # 检查是否全匹配成功
    assert all([sum(l[:-1]) > 0 for l in levels])

    # 根据 levels 生成 region_names 和 NAME 字段的对应关系: {NAME: region_name}
    shp_regions = {}

    # 根据 levels 统计每个地区的经纬度
    locations = {}  # 初始化一个空列表，用于存储结果
    # 遍历每个 region_names 的索引和名称
    for i, region_name in enumerate(region_names):
        # 获取 levels 中的相关信息
        level = levels[i][0]
        count = levels[i][1]
        indices = levels[i][2]

        [shp_regions.update({df[3]["NAME"][_i]: region_name}) for _i in indices]

        # 获取对应的坐标
        key_df, key_NAME = str(list(region_coords.keys())[level])
        csv_content = df[int(key_df)].iloc[indices]
        # 求平均
        locations[region_name] = list(csv_content.values[:, -2:].mean(0))

    return shp_regions, locations

