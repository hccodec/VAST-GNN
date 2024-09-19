"""
dataforgood jieguo 
"""

from utils.custom_datetime import DateRange, date2str, str2date
import geopandas as gpd, os, json, math, copy
import numpy as np
import torch, random
from torch.utils.data import DataLoader, TensorDataset
import pickle
from tqdm.auto import tqdm
from utils.logger import logger
from utils.utils import progress_indicator
import networkx as nx
import pandas as pd

data_dir = "data/dataforgood"
country_codes = ["IT", "ES", "EN", "FR"]
country_names = ["Italy", "Spain", "England", "France"]
country_start_dates = ["0224", "0312", "0313", "0310"]
country_end_date = "0512"
# window_size = 21

meta_info = {"name": country_names, "code": country_codes}

def load_data(args):
    databinfile = os.path.join(args.preprocessed_data_dir, args.databinfile)
    # if os.path.exists(databinfile):
    #     with open(databinfile, 'rb') as f:
    #         meta_data = pickle.load(f)
    #     logger.info('已从数据文件读取 dataforgood 数据集')
    #     return meta_data

    meta_data = {}
    for i in range(len(country_names)):
        data = _load_data(i, args.window)
        dataloaders, data_indices = split_dataset(args, *data)
        meta_data[country_names[i]] = (dataloaders, data, data_indices)

    if not os.path.exists(databinfile):
        os.makedirs(args.preprocessed_data_dir, exist_ok=True)
        with open(databinfile, 'wb') as f:
            pickle.dump(meta_data, f)

    return meta_data


def _load_data(i_country: int, window_size: int):
    """
    按国家从文件加载数据集
    """

    logger.info(f"正在读取国家 {country_names[i_country]} 的数据...")

    dates = [
        date2str(date, "%Y-%m-%d")
        for date in DateRange(
            f"2020{country_start_dates[i_country]}",
            str2date(f"2020{country_end_date}") + 1,
        )
    ]

    labels = pd.read_csv(
        os.path.join(
            data_dir,
            country_names[i_country],
            f"{country_names[i_country].lower()}_labels.csv",
        )
    ).set_index("name")

    Gs = []
    nodes_pre = []

    for i_dates in range(len(dates)):
        graph = pd.read_csv(
            os.path.join(
                data_dir,
                country_names[i_country],
                "graphs",
                "{}_{}.csv".format(country_codes[i_country], dates[i_dates]),
            ),
            header=None,
        )
        G = nx.DiGraph()
        nodes = sorted(set(list(graph[0]) + list(graph[1])))
        G.add_nodes_from(nodes)
        for row in graph.iterrows():
            G.add_edge(row[1][0], row[1][1], weight=row[1][2])
        Gs.append(G)
        assert not len(nodes_pre) or nodes_pre == nodes  # 所有图的结点都相同
        nodes_pre = nodes

    labels = labels.loc[nodes, dates]  # not needed

    adjs = np.stack([nx.adjacency_matrix(G).toarray() for G in Gs])  # A

    H = np.zeros((len(dates), len(nodes), window_size))  # X
    y = np.zeros((len(dates), len(nodes), 1))  # y

    for i_date in range(len(dates)):
        for i_node in range(len(nodes)):
            H[i_date, i_node, max(0, window_size - i_date) :] = labels.iloc[
                i_node, max(0, i_date - window_size) : i_date
            ]
            y[i_date, i_node] = labels.iloc[i_node, i_date]

    # train_loader, validation_loader, test_loader, train_origin, validation_origin, test_origin, train_indices, validation_indices, test_indices, data_origin, date_all
    H, y, adjs = torch.tensor(H), torch.tensor(y), torch.tensor(adjs)

    return H, y, adjs


# 分割数据集
def split_dataset(args, 
    H, y, adjs, batchsize=8, test_sample=15, shift=1, sep=10
):
    train_indices, val_indices, test_indices = [], [], []
    for i in range(min(args.window - 1, test_sample - sep), test_sample):
        if i < test_sample - sep or (i - test_sample + sep) & 1:
            train_indices.append(i)
        else:
            val_indices.append(i)
    test_indices.append(test_sample)

    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    test_indices = np.array(test_indices)

    x_days, y_days = args.xdays, args.ydays
    X = torch.zeros_like(H).unsqueeze(1).repeat(1, x_days, 1, 1)
    A = torch.zeros_like(adjs).unsqueeze(1).repeat(1, x_days, 1, 1)
    Y = torch.zeros_like(y).unsqueeze(1).repeat(1, y_days, 1, 1)

    for i_date in range(H.shape[0]):
        for i_xdays in range(x_days):
            _i_date = i_date + i_xdays - x_days + 1
            if _i_date < 0: _i_date = 0
            X[i_date, i_xdays] = H[_i_date]
            A[i_date, i_xdays] = adjs[_i_date]
        for i_ydays in range(y_days):
            _i_date = i_date + 1 + i_ydays
            if _i_date >= Y.size(0): _i_date = Y.size(0) - 1
            Y[i_date, i_ydays] = y[_i_date]

    y_indices = lambda indices: np.where(
        (test_sample > 0) & (indices + shift >= test_sample), indices, indices + shift
    )
    # (
    #     (indices + shift)
    #     if (test_sample > 0 and indices + shift >= test_sample)
    #     else indices
    # )

    train_data = (X[train_indices], Y[y_indices(train_indices)], A[train_indices])
    val_data = (X[val_indices], Y[y_indices(val_indices)], A[val_indices])
    test_data = (X[test_indices], Y[y_indices(test_indices)], A[test_indices])

    train_loader = DataLoader(
        TensorDataset(*train_data), batch_size=batchsize, shuffle=True
    )
    val_loader = DataLoader(TensorDataset(*val_data), batch_size=batchsize)
    test_loader = DataLoader(TensorDataset(*test_data), batch_size=batchsize)

    return (train_loader, val_loader, test_loader), (train_indices, val_indices, test_indices)

# def avg(data, *keys):
#     def recursive_avg(dicts):
#         res = dict()
#         keys = sorted(set.intersection(*(set(d.keys()) for d in dicts)))
#         for key in keys:
#             sub_dicts = [d[key] for d in dicts if key in d]
#             if all(isinstance(sub_dict, dict) for sub_dict in sub_dicts):
#                 res[key] = recursive_avg(sub_dicts)
#             else:
#                 res[key] = sum(sub_dicts) / len(sub_dicts)
#         return res

#     dicts = [data[k] for k in keys]
#     return recursive_avg(dicts)

# # 沿着天数进行滑动平均
# def smooth(data: dict, window_size = 7):
#     window_start = math.floor(window_size / 2)
#     window_end = math.ceil(window_size / 2)

#     data_copy = copy.copy(data) # 在原数据上移动窗口求平均

#     keys = sorted(data.keys())
#     for day in keys:
#         date_start = str2date(day) - window_start
#         date_end = str2date(day) + window_end

#         if date_start < str2date(keys[0]): date_start = str2date(keys[0])
#         if date_end > str2date(keys[-1]) + 1: date_end = str2date(keys[-1]) + 1


#         key_neighbors = DateRange(date2str(date_start), date2str(date_end))
#         key_neighbors = [date2str(k) for k in key_neighbors]
#         data[day] = avg(data_copy, *key_neighbors)
#     return data


# def interpolate(data, hint=''):
#     res = continuous(sorted(data.keys()))
#     r = []
#     for start, end in res:
#         v = avg(data, start, end)
#         start, end = str2date(start), str2date(end)
#         for date in DateRange(start + 1, end):
#             date = date2str(date)
#             data[date] = v
#             r.append(date)
#     logger.info(f'为 {hint} 插补 {len(r)} 条缺失数据 ({",".join(r)})')

# def read_json_dir(data_dir, data_type):
#     assert data_type in ["mobility", "text"]
#     data = dict()
#     for file in os.listdir(os.path.join(data_dir, data_type)):
#         if not file.endswith(".json"):
#             continue
#         with open(os.path.join(data_dir, data_type, file)) as f:
#             data[file.split("_")[-1].split(".")[0]] = json.load(f)

#     sub_keys = [tuple(data[k].keys()) for k in data]
#     assert len(set(sub_keys)) == 1

#     keys = sorted(data.keys())
#     sub_keys = sorted(list(sub_keys)[0])

#     logger.info(
#         "已读取 {} 天的 {} 数据 {}-{}，缺失 {} 日".format(
#             len(keys),
#             data_type,
#             keys[0],
#             keys[-1],
#             (str2date(keys[-1]) - str2date(keys[0])).days + 1 - len(keys),
#         )
#     )

#     interpolate(data, data_type)

#     return data

# def split_dataset(args, data_origin, date_all, mode=2):

#     # 先生成对应索引，再生成
#     n = len(date_all) - args.xdays - args.ydays + 1
#     train_ratio, validation_ratio = args.trainratio, args.validateratio

#     if hasattr(n, '__len__'): n = len(n)
#     assert train_ratio + validation_ratio < 1 and isinstance(n, int)

#     n_train, n_validation = int(n * train_ratio), int(n * validation_ratio)
#     n_test = n - n_train - n_validation

#     train_indices, validation_indices, test_indices = [], [], []
#     if mode == 0:
#         # sequential
#         train_indices = range(n_train)
#         validation_indices = np.arange(n_validation) + train_indices[-1] + 1
#     elif mode == 1:
#         # modulo 9
#         train_validate = list(range(n_train + n_validation))
#         for i in train_validate:
#             if i % 9: train_indices.append(i)
#             else: validation_indices.append(i)
#     elif mode == 2:
#         # backwards even index
#         train_validate = list(range(n_train + n_validation))
#         for i in train_validate:
#             # 从后往前，将索引为偶数的当验证集
#             if i < n_train - n_validation or i % 2: train_indices.append(i)
#             else: validation_indices.append(i)
#     test_indices = n - 1 - np.arange(n_test)
#     train_indices, validation_indices, test_indices = sorted(train_indices), sorted(validation_indices), sorted(test_indices)

#     # 打乱训练集顺序
#     random.shuffle(train_indices, random=random.random)

#     train_origin = [_data[train_indices] for _data in data_origin]
#     validation_origin = [_data[validation_indices] for _data in data_origin]
#     test_origin = [_data[test_indices] for _data in data_origin]

#     train_data, validation_data, test_data = TensorDataset(*train_origin), TensorDataset(*validation_origin), TensorDataset(*test_origin)
#     train_loader, validation_loader, test_loader = DataLoader(train_data, args.batchsize), DataLoader(validation_data, args.batchsize), DataLoader(test_data, args.batchsize)

#     return (
#         train_loader, validation_loader, test_loader,
#         train_origin, validation_origin, test_origin,
#         train_indices, validation_indices, test_indices
#     )

# def regulate_web_search_frequency(text, text_normalize_ratio):
#     '''
#     对齐 web 搜索关键词，并乘以 text_normalize_ratio
#     '''
#     # 提取数据中的 web 搜索词
#     sym_names_data = sorted(set(iii for i in [[sorted(text[k][_k].keys()) for _k in text[k]] for k in text] for ii in i for iii in ii))

#     #
#     sym_names = ['Pain', 'Headache', 'Cough', 'Diarrhea', 'Stress', 'Anxiety', 'Abdominal pain', 'Dizziness']
#     assert all([i in sym_names_data for i in sym_names])

#     n = len(sym_names)

#     # 按照所提取的 web 搜索词处理数据
#     for day in text:
#         for zone in text[day]:
#             res = {}
#             for sym in sym_names:
#                 res[sym] = (text[day][zone][sym] if sym in text[day][zone] else 0.) * text_normalize_ratio
#             text[day][zone] = res

#     zones = sorted(text[sorted(text)[0]])
#     assert all([sorted(text[k]) == zones for k in text])

#     return text, sym_names

# def min_max_normalize_web_search_frequency(text):
#     days = sorted(text.keys())
#     areas = sorted(text[days[0]].keys())
#     syms = sorted(text[days[0]][areas[0]])

#     for area in areas:
#         for sym in syms:
#             _lis = [text[day][area][sym] for day in days]
#             _lis_without_0 = [e for e in _lis if e != 0]

#             if len(_lis_without_0): _min, _max = min(_lis_without_0), max(_lis_without_0)
#             else: continue

#             if _min - _max: _res = [((e - _min) / (_max - _min)) if e else e for e in _lis]
#             else: _res = [1. for e in _lis]

#             for i, day in enumerate(days):
#                 text[day][area][sym] = _res[i]

#     return text

# def read_cases_data(data_dir, zones, case_normalize_ratio):

#     with open(os.path.join(data_dir, "covid_case/patient.json")) as f:
#         cases = json.load(f)

#     # 处理 cases 数据的 key 顺序：cases[zone][day] → cases[day][zone]
#     zones_in_cases = cases.keys()
#     days = set([tuple(cases[k].keys()) for k in cases.keys()])
#     assert len(days) == 1  # 确定 sub_keys 一致
#     zones, days = sorted(zones), list(days)[0]
#     cases_new = {}
#     for day in days:
#         for zone_in_cases in zones_in_cases:
#             day_new = date2str(str2date(day, "%Y/%m/%d"))
#             zone_new = zone_in_cases[:-1]
#             _v = float(cases[zone_in_cases][day]) * 1. / case_normalize_ratio
#             if zone_new not in zones: continue
#             if day_new in cases_new:
#                 cases_new[day_new][zone_new] = _v
#             else:
#                 cases_new[day_new] = {zone_new: _v}
#     days_new = sorted(cases_new.keys())

#     logger.info(
#         "已读取 {} 天的 {} 数据 {}-{}，缺失 {} 日".format(
#             len(days_new), 'cases', days_new[0], days_new[-1],
#             (str2date(days_new[-1]) - str2date(days_new[0])).days + 1 - len(days_new),
#         )
#     )

#     for day in DateRange('20200303', '20200331'):
#         cases_new[date2str(day)] = cases_new['20200401']
#     interpolate(cases_new, hint="cases")
#     # outlier
#     cases_new['20200331'] = avg(cases_new, '20200401', '20200401')
#     cases_new['20200910'] = avg(cases_new, '20200909', '20200912')
#     cases_new['20200911'] = avg(cases_new, '20200909', '20200912')
#     cases_new['20200511'] = avg(cases_new, '20200510', '20200512')
#     cases_new['20201208'] = avg(cases_new, '20201207', '20201209')
#     cases_new['20210208'] = avg(cases_new, '20210207', '20210209')
#     cases_new['20210214'] = avg(cases_new, '20210213', '20210215')

#     cases_subtracted = {}
#     days_new = sorted(cases_new.keys())
#     areas = list(cases_new[days_new[0]].keys())
#     for i in range(len(days_new) - 1):
#         for area in areas:
#             yesterday, today = days_new[i], days_new[i + 1]
#             _v = cases_new[today][area] - cases_new[yesterday][area]
#             if today in cases_subtracted:
#                 cases_subtracted[today][area] = _v
#             else:
#                 cases_subtracted[today] = {area: _v}

#     return cases_subtracted, cases_new

# def read_data(data_dir, start_date, end_date, x_days, y_days, case_normalize_ratio, text_normalize_ratio):

#     # 因 mobility 和 text 数据的限制，故裁剪数据，只取 13101-13123 这 23 个地区
#     zones = list(
#         gpd.read_file(os.path.join(data_dir, "tokyo_shapefile/tokyo.shp"))["JCODE"]
#     )[:23]

#     # 读取数据集中的数据
#     cases, cases_origin = read_cases_data(data_dir, zones, case_normalize_ratio)
#     mobility = read_json_dir(data_dir, "mobility")
#     text = read_json_dir(data_dir, "text")
#     # 将网络搜索数据整理成数组
#     text, sym_names = regulate_web_search_frequency(text, text_normalize_ratio)

#     qbar = progress_indicator(total=100, desc='Smoothing data', show_total=False)
#     cases = smooth(cases)
#     qbar.update(30)
#     mobility = smooth(mobility)
#     qbar.update(30)
#     text = smooth(text)
#     qbar.update(40)
#     qbar.close()

#     text = min_max_normalize_web_search_frequency(text)

#     # 原始数据读取完毕，开始数据预处理
#     date_all = DateRange(
#         str2date(start_date) - x_days,
#         str2date(end_date) + y_days)

#     date_all = [date2str(e) for e in date_all]
#     data_len = len(date_all) - x_days - y_days + 1

#     data_origin = [
#         np.empty((data_len, x_days, len(zones), x_days)),            # casex
#         np.empty((data_len, y_days, len(zones), y_days)),            # casey
#         np.empty((data_len, x_days, len(zones), len(zones))),        # adj: mobility
#         np.empty((data_len, x_days, len(zones), len(sym_names))),    # extra: text
#         np.empty((data_len))                                         # day_order
#     ]

#     qbar1 = progress_indicator(range(data_len), desc='Converting data')

#     # 将数据整理成 feature, label
#     for i in qbar1:
#         # 以 i 偏移, [0-20] 作 数据 20 的 feature, [21-27] 作数据 20 要预测的数据
#         features_days = date_all[i: i + x_days]
#         labels_days = date_all[i + x_days: i + x_days + y_days]
#         for i_day, day in enumerate(features_days):
#             # case x
#             _dates = [date2str(i) for i in DateRange(
#                 str2date(day) - 20, str2date(day) + 1
#                 )]
#             for zone in cases[day]:
#                 data_origin[0][i, i_day, zones.index(zone)] = \
#                     np.array([cases[d][zone] for d in _dates])
#                 # data_origin[2][i, i_day, zones.index(zone)] = cases[day][zone]
#             # mobility
#             for zone in mobility[day]:
#                 zones_from, zones_to = zone.split('_')
#                 data_origin[2][i, i_day, zones.index(zones_from), zones.index(zones_to)] = mobility[day][zone]
#             # text
#             for zone in text[day]:
#                 for i_sym, sym in enumerate(sym_names):
#                     data_origin[3][i, i_day, zones.index(zone), i_sym] = np.array([text[day][zone][sym]])
#         for i_day, day in enumerate(labels_days):
#             _dates = [date2str(i) for i in DateRange(
#                 str2date(day), str2date(day) + y_days
#                 )]
#             for zone in cases[day]:
#                 data_origin[1][i, i_day, zones.index(zone)] = \
#                     np.array([cases[d][zone] for d in _dates])
#         data_origin[4][i] = i

#     # 归一化 mobility，一个地区出发的所有值之和为 1
#     sum_over_zone_from = np.sum(data_origin[2], axis=-2)
#     sum_over_zone_from_expand = np.expand_dims(sum_over_zone_from, axis=-2)
#     data_origin[2] = data_origin[2] / sum_over_zone_from_expand

#     data_origin = [torch.tensor(d) for d in data_origin]
#     # data_origin[0] = data_origin[0] / data_origin[0].sum(-2).unsqueeze(-2) # 归一化 mobility, 一个地区出发的所有值之和为 1
#     # 归一化 casex

#     return data_origin, date_all

# def load_data(args):

#     preprocessed_data_dir, databinfile = args.preprocessed_data_dir, args.databinfile

#     start_date, end_date, x_days, y_days = args.startdate, args.enddate, args.xdays, args.ydays
#     data_dir, case_normalize_ratio, text_normalize_ratio = args.data_dir, args.case_normalize_ratio, args.text_normalize_ratio

#     res = None
#     path = os.path.join(preprocessed_data_dir, databinfile)
#     if os.path.exists(path):
#         try:
#             logger.info(f'尝试从 {path} 中读取数据文件')
#             with open(path, 'rb') as f:
#                 res = pickle.load(f)
#                 logger.info(f'已从 {path} 中读取数据文件')
#         except Exception as e:
#             logger.info('数据文件异常，重新读取')
#             res = None
#     else:
#         logger.info("数据文件不存在，重新读取")
#         if not os.path.exists(preprocessed_data_dir):
#             os.makedirs(preprocessed_data_dir, exist_ok=True)
#     if res is None:
#         res = read_data(data_dir, start_date, end_date, x_days, y_days, case_normalize_ratio, text_normalize_ratio)
#         with open(path, 'wb') as f:
#             pickle.dump(res, f)
#             logger.info(f'将数据文件保存至 {path}')
#     return res
