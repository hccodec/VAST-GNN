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

pack_data = lambda maes, data, keys: pd.DataFrame(
    {"mae": maes.reshape(-1), "path": data.reshape(-1)},
    index=pd.MultiIndex.from_product(
        keys,
        names=["y", "country", "model"],
    ),
)

def get_paths():

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
    data_80 = np.array(
        [
            [
                [
                    "results_2024/tests_1128/exp_1_lstm/dataforgood/lstm_7_1_w7_s2_20241128114839/model_EN_best.pth",
                    "results_2024/tests_1128/exp_1_mpnn_lstm/dataforgood/mpnn_lstm_7_1_w7_s2_20241128102848/model_EN_best.pth",
                    "results_2024/tests_1201/exp_2_mpnn_lstm_maml_80/dataforgood/mpnn_lstm_7_1_w7_s2_20241201130312/model_EN_best.pth",
                    "results_2025_1/test_0106_all/exp_2_80_all/dataforgood/dynst_7_1_w7_s2_20250106171649/model_EN_best.pth",
                ],
                [
                    "results_2024/tests_1128/exp_1_lstm/dataforgood/lstm_7_1_w7_s2_20241128103124/model_FR_best.pth",
                    "results_2024/tests_1128/exp_1_mpnn_lstm/dataforgood/mpnn_lstm_7_1_w7_s2_20241128102848/model_FR_best.pth",
                    "results_2024/tests_1201/exp_2_mpnn_lstm_maml_80/dataforgood/mpnn_lstm_7_1_w7_s2_20241201130312/model_FR_best.pth",
                    "results_2025_1/test_0106_all/exp_2_80_all/dataforgood/dynst_7_1_w7_s2_20250106163802/model_FR_best.pth",
                ],
                [
                    "results_2024/tests_1128/exp_1_lstm/dataforgood/lstm_7_1_w7_s2_20241128121218/model_IT_best.pth",
                    "results_2024/tests_1128/exp_1_mpnn_lstm/dataforgood/mpnn_lstm_7_1_w7_s2_20241128102848/model_IT_best.pth",
                    "results_2024/tests_1201/exp_2_mpnn_lstm_maml_80/dataforgood/mpnn_lstm_7_1_w7_s2_20241201130312/model_IT_best.pth",
                    "results_2025_1/test_0106_all/exp_2_80_all/dataforgood/dynst_7_1_w7_s2_20250106163802/model_IT_best.pth",
                ],
                [
                    "results_2024/tests_1128/exp_1_lstm/dataforgood/lstm_7_1_w7_s2_20241128103152/model_ES_best.pth",
                    "results_2024/tests_1128/exp_1_mpnn_lstm/dataforgood/mpnn_lstm_7_1_w7_s2_20241128102848/model_ES_best.pth",
                    "results_2024/tests_1201/exp_2_mpnn_lstm_maml_80/dataforgood/mpnn_lstm_7_1_w7_s2_20241201131439/model_ES_best.pth",
                    "results_2025_1/test_0106_all/exp_2_80_all/dataforgood/dynst_7_1_w7_s2_20250106163802/model_ES_best.pth",
                ],
                [
                    "results_2024/tests_1210/exp_1_baselines_lstm_80/dataforgood/lstm_7_1_w7_s2_20241210110540/model_NZ_best.pth",
                    "results_2024/tests_1210/exp_1_baselines_mpnn_lstm_80/dataforgood/mpnn_lstm_7_1_w7_s2_20241210110540/model_NZ_best.pth",
                    "results_2024/tests_1210/exp_1_baselines_mpnn_tl_80/dataforgood/mpnn_lstm_7_1_w7_s2_20241210110540/model_NZ_best.pth",
                    "results_2024/tests_1210/exp_2_80_graph_lambda_5/dataforgood/dynst_7_1_w7_s2_20241210140100/model_NZ_best.pth",
                ],
                [
                    "results_2025_1/tests_0104_100days_seed2/exp_1_baselines_lstm_80/japan/lstm_7_1_w7_s2_20250105003930/model_JP_best.pth",
                    "results_2025_1/tests_0104_100days_seed2/exp_1_baselines_mpnn_lstm_80/japan/mpnn_lstm_7_1_w7_s2_20250105003609/model_JP_best.pth",
                    "results_2025_1/tests_0104_100days_seed2/exp_1_baselines_mpnn_tl_80/japan/mpnn_lstm_7_1_w7_s2_20250105002825/model_JP_best.pth",
                    "results_2025_1/test_0106_all/exp_2_80_all/japan/dynst_7_1_w7_s2_20250106163803/model_JP_best.pth",
                ],
            ],
            [
                [
                    "results_2024/tests_1128/exp_1_lstm/dataforgood/lstm_7_1_w7_s6_20241129055225/model_EN_best.pth",
                    "results_2024/tests_1128/exp_1_mpnn_lstm/dataforgood/mpnn_lstm_7_1_w7_s6_20241129052121/model_EN_best.pth",
                    "results_2024/tests_1201/exp_2_mpnn_lstm_maml_80/dataforgood/mpnn_lstm_7_1_w7_s6_20241201130502/model_EN_best.pth",
                    "results_2025_1/test_0106_all/exp_2_80_all/dataforgood/dynst_7_1_w7_s6_20250106205938/model_EN_best.pth",
                ],
                [
                    "results_2024/tests_1128/exp_1_lstm/dataforgood/lstm_7_1_w7_s6_20241129012657/model_FR_best.pth",
                    "results_2024/tests_1128/exp_1_mpnn_lstm/dataforgood/mpnn_lstm_7_1_w7_s6_20241129012506/model_FR_best.pth",
                    "results_2024/tests_1201/exp_2_mpnn_lstm_maml_80/dataforgood/mpnn_lstm_7_1_w7_s6_20241201131736/model_FR_best.pth",
                    "results_2025_1/test_0106_all/exp_2_80_all/dataforgood/dynst_7_1_w7_s6_20250106222555/model_FR_best.pth",
                ],
                [
                    "results_2024/tests_1128/exp_1_lstm/dataforgood/lstm_7_1_w7_s6_20241129103630/model_IT_best.pth",
                    "results_2024/tests_1128/exp_1_mpnn_lstm/dataforgood/mpnn_lstm_7_1_w7_s6_20241129093459/model_IT_best.pth",
                    "results_2024/tests_1201/exp_2_mpnn_lstm_maml_80/dataforgood/mpnn_lstm_7_1_w7_s6_20241201130312/model_IT_best.pth",
                    "results_2025_1/test_0106_all/exp_2_80_all/dataforgood/dynst_7_1_w7_s6_20250106171400/model_IT_best.pth",
                ],
                [
                    "results_2024/tests_1128/exp_1_lstm/dataforgood/lstm_7_1_w7_s6_20241128130639/model_ES_best.pth",
                    "results_2024/tests_1128/exp_1_mpnn_lstm/dataforgood/mpnn_lstm_7_1_w7_s6_20241128130446/model_ES_best.pth",
                    "results_2024/tests_1201/exp_2_mpnn_lstm_maml_80/dataforgood/mpnn_lstm_7_1_w7_s6_20241201130312/model_ES_best.pth",
                    "results_2025_1/test_0106_all/exp_2_80_all/dataforgood/dynst_7_1_w7_s6_20250106175411/model_ES_best.pth",
                ],
                [
                    "results_2024/tests_1210/exp_1_baselines_lstm_80/dataforgood/lstm_7_1_w7_s6_20241210110540/model_NZ_best.pth",
                    "results_2024/tests_1210/exp_1_baselines_mpnn_lstm_80/dataforgood/mpnn_lstm_7_1_w7_s6_20241210110540/model_NZ_best.pth",
                    "results_2024/tests_1210/exp_1_baselines_mpnn_tl_80/dataforgood/mpnn_lstm_7_1_w7_s6_20241210110540/model_NZ_best.pth",
                    "results_2024/tests_1210/exp_2_80_graph_lambda_5/dataforgood/dynst_7_1_w7_s6_20241210120750/model_NZ_best.pth",
                ],
                [
                    "results_2025_1/tests_0104_100days_seed2/exp_1_baselines_lstm_80/japan/lstm_7_1_w7_s6_20250104230922/model_JP_best.pth",
                    "results_2025_1/tests_0104_100days_seed2/exp_1_baselines_mpnn_lstm_80/japan/mpnn_lstm_7_1_w7_s6_20250104231153/model_JP_best.pth",
                    "results_2025_1/tests_0104_100days_seed2/exp_1_baselines_mpnn_tl_80/japan/mpnn_lstm_7_1_w7_s6_20250104230300/model_JP_best.pth",
                    "results_2025_1/test_0106_all/exp_2_80_all/japan/dynst_7_1_w7_s6_20250106163802/model_JP_best.pth",
                ],
            ],
            [
                [
                    "results_2024/tests_1128/exp_1_lstm/dataforgood/lstm_7_1_w7_s13_20241130044729/model_EN_best.pth",
                    "results_2024/tests_1128/exp_1_mpnn_lstm/dataforgood/mpnn_lstm_7_1_w7_s13_20241130040357/model_EN_best.pth",
                    "results_2024/tests_1201/exp_2_mpnn_lstm_maml_80/dataforgood/mpnn_lstm_7_1_w7_s13_20241201130312/model_EN_best.pth",
                    "results_2025_1/test_0106_all/exp_2_80_all/dataforgood/dynst_7_1_w7_s13_20250106163802/model_EN_best.pth",
                ],
                [
                    "results_2024/tests_1128/exp_1_lstm/dataforgood/lstm_7_1_w7_s13_20241129131830/model_FR_best.pth",
                    "results_2024/tests_1128/exp_1_mpnn_lstm/dataforgood/mpnn_lstm_7_1_w7_s13_20241129131644/model_FR_best.pth",
                    "results_2024/tests_1201/exp_2_mpnn_lstm_maml_80/dataforgood/mpnn_lstm_7_1_w7_s13_20241201131733/model_FR_best.pth",
                    "results_2025_1/test_0106_all/exp_2_80_all/dataforgood/dynst_7_1_w7_s13_20250106163803/model_FR_best.pth",
                ],
                [
                    "results_2024/tests_1128/exp_1_lstm/dataforgood/lstm_7_1_w7_s13_20241130130859/model_IT_best.pth",
                    "results_2024/tests_1128/exp_1_mpnn_lstm/dataforgood/mpnn_lstm_7_1_w7_s13_20241130123031/model_IT_best.pth",
                    "results_2024/tests_1201/exp_2_mpnn_lstm_maml_80/dataforgood/mpnn_lstm_7_1_w7_s13_20241201130453/model_IT_best.pth",
                    "results_2025_1/test_0106_all/exp_2_80_all/dataforgood/dynst_7_1_w7_s13_20250106163803/model_IT_best.pth",
                ],
                [
                    "results_2024/tests_1128/exp_1_lstm/dataforgood/lstm_7_1_w7_s13_20241128161014/model_ES_best.pth",
                    "results_2024/tests_1128/exp_1_mpnn_lstm/dataforgood/mpnn_lstm_7_1_w7_s13_20241128160830/model_ES_best.pth",
                    "results_2024/tests_1201/exp_2_mpnn_lstm_maml_80/dataforgood/mpnn_lstm_7_1_w7_s13_20241201131421/model_ES_best.pth",
                    "results_2025_1/test_0106_all/exp_2_80_all/dataforgood/dynst_7_1_w7_s13_20250106173159/model_ES_best.pth",
                ],
                [
                    "results_2024/tests_1210/exp_1_baselines_lstm_80/dataforgood/lstm_7_1_w7_s13_20241210110540/model_NZ_best.pth",
                    "results_2024/tests_1210/exp_1_baselines_mpnn_lstm_80/dataforgood/mpnn_lstm_7_1_w7_s13_20241210110540/model_NZ_best.pth",
                    "results_2024/tests_1210/exp_1_baselines_mpnn_tl_80/dataforgood/mpnn_lstm_7_1_w7_s13_20241210110540/model_NZ_best.pth",
                    "results_2024/tests_1210/exp_2_80_graph_lambda_2/dataforgood/dynst_7_1_w7_s13_20241210153022/model_NZ_best.pth",
                ],
                [
                    "results_2025_1/tests_0104_100days_seed2/exp_1_baselines_lstm_80/japan/lstm_7_1_w7_s13_20250104213850/model_JP_best.pth",
                    "results_2025_1/tests_0104_100days_seed2/exp_1_baselines_mpnn_lstm_80/japan/mpnn_lstm_7_1_w7_s13_20250104211554/model_JP_best.pth",
                    "results_2025_1/tests_0104_100days_seed2/exp_1_baselines_mpnn_tl_80/japan/mpnn_lstm_7_1_w7_s13_20250104213437/model_JP_best.pth",
                    "results_2025_1/test_0106_all/exp_2_80_all/japan/dynst_7_1_w7_s13_20250106212033/model_JP_best.pth",
                ],
            ],
        ]
    )
    data_50 = np.array(
        [
            [
                [
                    "results_2024/tests_1119/exp_7/dataforgood/lstm_7_1_w7_s2_20241119165850/model_EN_best.pth",
                    "results_2024/tests_1120/exp_1/dataforgood/mpnn_lstm_7_1_w7_s2_20241120193339/model_EN_best.pth",
                    "results_2024/tests_1201/exp_1_mpnn_lstm_maml/dataforgood/mpnn_lstm_7_1_w7_s2_20241201125738/model_EN_best.pth",
                    "results_2025_1/test_0106_all/exp_2_50_all/dataforgood/dynst_7_1_w7_s2_20250106163802/model_EN_best.pth",
                ],
                [
                    "results_2024/tests_1120/exp_1/dataforgood/lstm_7_1_w7_s2_20241120194051/model_FR_best.pth",
                    "results_2024/tests_1120/exp_1/dataforgood/mpnn_lstm_7_1_w7_s2_20241120193932/model_FR_best.pth",
                    "results_2024/tests_1128_maml/exp_1_mpnn_lstm_maml/dataforgood/mpnn_lstm_7_1_w7_s2_20241128214840/model_FR_best.pth",
                    "results_2025_1/test_0106_all/exp_2_50_all/dataforgood/dynst_7_1_w7_s2_20250106163802/model_FR_best.pth",
                ],
                [
                    "results_2024/tests_1120/exp_1/dataforgood/lstm_7_1_w7_s2_20241120193339/model_IT_best.pth",
                    "results_2024/tests_1120/exp_1/dataforgood/mpnn_lstm_7_1_w7_s2_20241120193339/model_IT_best.pth",
                    "results_2024/tests_1201/exp_1_mpnn_lstm_maml/dataforgood/mpnn_lstm_7_1_w7_s2_20241201125738/model_IT_best.pth",
                    "results_2024/tests_1129/exp_3_dynst_lambdas/dataforgood/dynst_7_1_w7_s2_20241130154732/model_IT_best.pth",
                ],
                [
                    "results_2024/tests_1120/exp_1/dataforgood/lstm_7_1_w7_s2_20241120193932/model_ES_best.pth",
                    "results_2024/tests_1120/exp_1/dataforgood/mpnn_lstm_7_1_w7_s2_20241120194200/model_ES_best.pth",
                    "results_2024/tests_1201/exp_1_mpnn_lstm_maml/dataforgood/mpnn_lstm_7_1_w7_s2_20241201125929/model_ES_best.pth",
                    "results_2025_1/test_0106_all/exp_2_50_all/dataforgood/dynst_7_1_w7_s2_20250106163802/model_ES_best.pth",
                ],
                [
                    "results_2024/tests_1210/exp_1_baselines_lstm_50/dataforgood/lstm_7_1_w7_s2_20241210110539/model_NZ_best.pth",
                    "results_2024/tests_1210/exp_1_baselines_mpnn_lstm_50/dataforgood/mpnn_lstm_7_1_w7_s2_20241210110540/model_NZ_best.pth",
                    "results_2024/tests_1210/exp_1_baselines_mpnn_tl_50/dataforgood/mpnn_lstm_7_1_w7_s2_20241210110540/model_NZ_best.pth",
                    "results_2024/tests_1210/exp_2_50_graph_lambda_8/dataforgood/dynst_7_1_w7_s2_20241210124708/model_NZ_best.pth",
                ],
                [
                    "results_2025_1/tests_0104_100days_seed2/exp_1_baselines_lstm_50/japan/lstm_7_1_w7_s2_20250104205705/model_JP_best.pth",
                    "results_2025_1/tests_0104_100days_seed2/exp_1_baselines_mpnn_lstm_50/japan/mpnn_lstm_7_1_w7_s2_20250104205704/model_JP_best.pth",
                    "results_2025_1/tests_0104_100days_seed2/exp_1_baselines_mpnn_tl_50/japan/mpnn_lstm_7_1_w7_s2_20250104205704/model_JP_best.pth",
                    "results_2025_1/test_0106_all/exp_2_50_all/japan/dynst_7_1_w7_s2_20250106202930/model_JP_best.pth",
                ],
            ],
            [
                [
                    "results_2024/tests_1120/exp_1/dataforgood/lstm_7_1_w7_s6_20241120193339/model_EN_best.pth",
                    "results_2024/tests_1120/exp_1/dataforgood/mpnn_lstm_7_1_w7_s6_20241120193339/model_EN_best.pth",
                    "results_2024/tests_1201/exp_1_mpnn_lstm_maml/dataforgood/mpnn_lstm_7_1_w7_s6_20241201125933/model_EN_best.pth",
                    "results_2024/tests_1129/exp_3_dynst_lambdas/dataforgood/dynst_7_1_w7_s6_20241130213416/model_EN_best.pth",
                ],
                [
                    "results_2024/tests_1120/exp_1/dataforgood/lstm_7_1_w7_s6_20241120194051/model_FR_best.pth",
                    "results_2024/tests_1120/exp_1/dataforgood/mpnn_lstm_7_1_w7_s6_20241120193932/model_FR_best.pth",
                    "results_2024/tests_1201/exp_1_mpnn_lstm_maml/dataforgood/mpnn_lstm_7_1_w7_s6_20241201125955/model_FR_best.pth",
                    "results_2024/tests_1129/exp_3_dynst_lambdas/dataforgood/dynst_7_1_w7_s6_20241130222712/model_FR_best.pth",
                ],
                [
                    "results_2024/tests_1120/exp_1/dataforgood/lstm_7_1_w7_s6_20241120193339/model_IT_best.pth",
                    "results_2024/tests_1120/exp_1/dataforgood/mpnn_lstm_7_1_w7_s6_20241120193339/model_IT_best.pth",
                    "results_2024/tests_1201/exp_1_mpnn_lstm_maml/dataforgood/mpnn_lstm_7_1_w7_s6_20241201125738/model_IT_best.pth",
                    "results_2025_1/test_0106_all/exp_2_50_all/dataforgood/dynst_7_1_w7_s6_20250106164541/model_IT_best.pth",
                ],
                [
                    "results_2024/tests_1120/exp_1/dataforgood/lstm_7_1_w7_s6_20241120193932/model_ES_best.pth",
                    "results_2024/tests_1120/exp_1/dataforgood/mpnn_lstm_7_1_w7_s6_20241120193944/model_ES_best.pth",
                    "results_2024/tests_1201/exp_1_mpnn_lstm_maml/dataforgood/mpnn_lstm_7_1_w7_s6_20241201125738/model_ES_best.pth",
                    "results_2025_1/test_0106_all/exp_2_50_all/dataforgood/dynst_7_1_w7_s6_20250106163802/model_ES_best.pth",
                ],
                [
                    "results_2024/tests_1210/exp_1_baselines_lstm_50/dataforgood/lstm_7_1_w7_s6_20241210110540/model_NZ_best.pth",
                    "results_2024/tests_1210/exp_1_baselines_mpnn_lstm_50/dataforgood/mpnn_lstm_7_1_w7_s6_20241210110539/model_NZ_best.pth",
                    "results_2024/tests_1210/exp_1_baselines_mpnn_tl_50/dataforgood/mpnn_lstm_7_1_w7_s6_20241210110540/model_NZ_best.pth",
                    "results_2024/tests_1210/exp_2_50_graph_lambda_3/dataforgood/dynst_7_1_w7_s6_20241210143733/model_NZ_best.pth",
                ],
                [
                    "results_2025_1/tests_0104_100days_seed2/exp_1_baselines_lstm_50/japan/lstm_7_1_w7_s6_20250105003114/model_JP_best.pth",
                    "results_2025_1/tests_0104_100days_seed2/exp_1_baselines_mpnn_lstm_50/japan/mpnn_lstm_7_1_w7_s6_20250105002320/model_JP_best.pth",
                    "results_2025_1/tests_0104_100days_seed2/exp_1_baselines_mpnn_tl_50/japan/mpnn_lstm_7_1_w7_s6_20250104235718/model_JP_best.pth",
                    "results_2025_1/test_0106_all/exp_2_50_all/japan/dynst_7_1_w7_s6_20250106163802/model_JP_best.pth",
                ],
            ],
            [
                [
                    "results_2024/tests_1120/exp_1/dataforgood/lstm_7_1_w7_s13_20241120193339/model_EN_best.pth",
                    "results_2024/tests_1120/exp_1/dataforgood/mpnn_lstm_7_1_w7_s13_20241120193339/model_EN_best.pth",
                    "results_2024/tests_1201/exp_1_mpnn_lstm_maml/dataforgood/mpnn_lstm_7_1_w7_s13_20241201125738/model_EN_best.pth",
                    "results_2025_1/test_0106_all/exp_2_50_all/dataforgood/dynst_7_1_w7_s13_20250106163802/model_EN_best.pth",
                ],
                [
                    "results_2024/tests_1120/exp_1/dataforgood/lstm_7_1_w7_s13_20241120193932/model_FR_best.pth",
                    "results_2024/tests_1120/exp_1/dataforgood/mpnn_lstm_7_1_w7_s13_20241120193932/model_FR_best.pth",
                    "results_2024/tests_1201/exp_1_mpnn_lstm_maml/dataforgood/mpnn_lstm_7_1_w7_s13_20241201125957/model_FR_best.pth",
                    "results_2025_1/test_0106_all/exp_2_50_all/dataforgood/dynst_7_1_w7_s13_20250106163803/model_FR_best.pth",
                ],
                [
                    "results_2024/tests_1120/exp_1/dataforgood/lstm_7_1_w7_s13_20241120193339/model_IT_best.pth",
                    "results_2024/tests_1120/exp_1/dataforgood/mpnn_lstm_7_1_w7_s13_20241120193340/model_IT_best.pth",
                    "results_2024/tests_1201/exp_1_mpnn_lstm_maml/dataforgood/mpnn_lstm_7_1_w7_s13_20241201125920/model_IT_best.pth",
                    "results_2025_1/test_0106_all/exp_2_50_all/dataforgood/dynst_7_1_w7_s13_20250106165047/model_IT_best.pth",
                ],
                [
                    "results_2024/tests_1120/exp_1/dataforgood/lstm_7_1_w7_s13_20241120193932/model_ES_best.pth",
                    "results_2024/tests_1120/exp_1/dataforgood/mpnn_lstm_7_1_w7_s13_20241120193932/model_ES_best.pth",
                    "results_2024/tests_1201/exp_1_mpnn_lstm_maml/dataforgood/mpnn_lstm_7_1_w7_s13_20241201125907/model_ES_best.pth",
                    "results_2025_1/test_0106_all/exp_2_50_all/dataforgood/dynst_7_1_w7_s13_20250106174123/model_ES_best.pth",
                ],
                [
                    "results_2024/tests_1210/exp_1_baselines_lstm_50/dataforgood/lstm_7_1_w7_s13_20241210110540/model_NZ_best.pth",
                    "results_2024/tests_1210/exp_1_baselines_mpnn_lstm_50/dataforgood/mpnn_lstm_7_1_w7_s13_20241210110539/model_NZ_best.pth",
                    "results_2024/tests_1210/exp_1_baselines_mpnn_tl_50/dataforgood/mpnn_lstm_7_1_w7_s13_20241210110539/model_NZ_best.pth",
                    "results_2024/tests_1210/exp_2_50_graph_lambda_5/dataforgood/dynst_7_1_w7_s13_20241210140155/model_NZ_best.pth",
                ],
                [
                    "results_2025_1/tests_0104_100days_seed2/exp_1_baselines_lstm_50/japan/lstm_7_1_w7_s13_20250104223140/model_JP_best.pth",
                    "results_2025_1/tests_0104_100days_seed2/exp_1_baselines_mpnn_lstm_50/japan/mpnn_lstm_7_1_w7_s13_20250104223419/model_JP_best.pth",
                    "results_2025_1/tests_0104_100days_seed2/exp_1_baselines_mpnn_tl_50/japan/mpnn_lstm_7_1_w7_s13_20250104223109/model_JP_best.pth",
                    "results_2025_1/test_0106_all/exp_2_50_all/japan/dynst_7_1_w7_s13_20250106194822/model_JP_best.pth",
                ],
            ],
        ]
    )


    data_50 = pack_data(convert_mae_table(maes_50), data_50, keys)
    data_80 = pack_data(convert_mae_table(maes_80), data_80, keys)

    return easydict.EasyDict({"o50": data_50, "o80": data_80})

def get_paths_flunet():

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
    data_50 = np.array(
        [
            [
                [
                    "results_2025_1/tests_0209_seed7/exp_1_baselines_lstm_50/flunet/lstm_7_1_w7_s2_20250212023714/model_h1n1_best.pth",
                    "results_2025_1/tests_0209_seed7/exp_1_baselines_mpnn_lstm_50/flunet/mpnn_lstm_7_1_w7_s2_20250212031259/model_h1n1_best.pth",
                    "results_2025_1/tests_0209_seed7/exp_1_baselines_mpnn_tl_50/flunet/mpnn_lstm_7_1_w7_s2_20250212024635/model_h1n1_best.pth",
                    "results_2025_1/tests_0209_seed7/exp_2_50_graph_lambda_7/flunet/dynst_7_1_w7_s2_20250212041727/model_h1n1_best.pth"
                ], [
                    "results_2025_1/tests_0209_seed7/exp_1_baselines_lstm_50/flunet/lstm_7_1_w7_s2_20250212174618/model_h3n2_best.pth",
                    "results_2025_1/tests_0209_seed7/exp_1_baselines_mpnn_lstm_50/flunet/mpnn_lstm_7_1_w7_s2_20250212172732/model_h3n2_best.pth",
                    "results_2025_1/tests_0209_seed7/exp_1_baselines_mpnn_tl_50/flunet/mpnn_lstm_7_1_w7_s2_20250212180613/model_h3n2_best.pth",
                    "results_2025_1/tests_0209_seed7/exp_2_50_graph_lambda_6/flunet/dynst_7_1_w7_s2_20250212185733/model_h3n2_best.pth"
                ], [
                    "results_2025_1/tests_0209_seed7/exp_1_baselines_lstm_50/flunet/lstm_7_1_w7_s2_20250213091421/model_BV_best.pth",
                    "results_2025_1/tests_0209_seed7/exp_1_baselines_mpnn_lstm_50/flunet/mpnn_lstm_7_1_w7_s2_20250213081547/model_BV_best.pth",
                    "results_2025_1/tests_0209_seed7/exp_1_baselines_mpnn_tl_50/flunet/mpnn_lstm_7_1_w7_s2_20250213082714/model_BV_best.pth",
                    "results_2025_1/tests_0209_seed7/exp_2_50_graph_lambda_0/flunet/dynst_7_1_w7_s2_20250213082524/model_BV_best.pth"
                ], [
                    "results_2025_1/tests_0209_seed7/exp_1_baselines_lstm_50/flunet/lstm_7_1_w7_s2_20250210010249/model_BY_best.pth",
                    "results_2025_1/tests_0209_seed7/exp_1_baselines_mpnn_lstm_50/flunet/mpnn_lstm_7_1_w7_s2_20250210003643/model_BY_best.pth",
                    "results_2025_1/tests_0209_seed7/exp_1_baselines_mpnn_tl_50/flunet/mpnn_lstm_7_1_w7_s2_20250210004508/model_BY_best.pth",
                    "results_2025_1/tests_0209_seed7/exp_2_50_graph_lambda_2/flunet/dynst_7_1_w7_s2_20250210011646/model_BY_best.pth",
                ],
            ],
        ]
    )


    data_50 = pack_data(convert_mae_table(maes_50), data_50, keys)

    return easydict.EasyDict({"o50": data_50})


paths = get_paths()
paths_flunet = get_paths_flunet()
# paths
