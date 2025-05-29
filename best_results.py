# -*- coding: utf-8 -*-
import easydict
import pandas as pd
import numpy as np
import os

# 使用 key0, key1, key2 为 data 转变成 pandas DataFrame 使其能同时通过索引和 key 访问
convert_mae_table = lambda s: np.array(
    [
        [_i.strip() for _i in i.split("\t") if _i != ""]
        for i in s.replace(" ", "\t").split("\n")
        if i != ""
    ]
).T

pack_data = lambda maes, keys: pd.DataFrame(
    {"mae": maes.reshape(-1)},
    index=pd.MultiIndex.from_product(
        keys,
        names=["y", "country", "model"],
    ),
)

def get_expexted_maes():

    keys = [
        [3, 7, 14],
        ["EN", "FR", "IT", "ES", "NZ", "JP"],
        ["lstm", "mpnn_lstm", "mpnn_tl", "dynst"],
    ]

    maes_80 = """
5.355	4.534	9.679	23.581	48.999  360.948	4.968	2.393	16.408	28.633	51.909  359.656	5.192	2.472	16.828	40.516	95.56   282.966
4.408	1.549	8.764	24.589	42.111  236.96	4.78	1.631	9.615	29.46	77.672  306.906	5.756	2.038	14.631	40.156	82.498  377.468
4.795	1.485	8.604	24.016	26.954  280.568	4.813	1.357	10.837	27.907	87.864  319.918	5.463	1.854	18.087	36.725	86.638  406.357
3.919	1.355	7.903	23.058	24.607  218.474	4.259	1.299	8.552	26.37	24.47   229.056	4.683	1.724	12.276	29.203	38.223  250.657"""
    maes_50 = """
4.669	2.877	12.877	22.654	58.469	406.134 5.283	1.870	17.395	37.802	65.017	318.185 5.215	2.601	24.819	53.229	86.963  258.199
5.077	1.817	11.639	22.475	28.901	291.568 5.355	2.047	11.063	34.563	42.777	348.285 5.134	2.047	17.904	51.458	110.469 326.615
5.032	1.754	10.388	19.279	34.938	225.71  5.308	1.778	12.328	33.722	58.682	331.014 6.073	2.213	21.929	45.341	183.378 347.541
4.523	1.545	8.919	18.194	26.006	223.516 4.577	1.578	9.304	29.875	33.628	315.112 5.125	1.899	12.682	33.028	46.371  257.598"""

    data_50 = pack_data(convert_mae_table(maes_50), keys)
    data_80 = pack_data(convert_mae_table(maes_80), keys)

    return easydict.EasyDict({"o50": data_50, "o80": data_80})

def get_expexted_maes_flunet():

    keys = [
        [3],
        ["h1n1", "h3n2", "BV", "BY"],
        ["lstm", "mpnn_lstm", "mpnn_tl", "dynst"],
    ]

    maes_50 = """
729.042 340.026  105.985    43.628
640.373 342.864 75.818  27.742
714.620 357.428 85.832  25.51
519.879 262.144 70.672  16.589"""

    data_50 = pack_data(convert_mae_table(maes_50), keys)

    return easydict.EasyDict({"o50": data_50})


expexted_maes = get_expexted_maes()
expexted_maes_flunet = get_expexted_maes_flunet()
# paths
