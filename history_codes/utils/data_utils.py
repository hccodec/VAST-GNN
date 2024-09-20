from utils.datetime import *
from tqdm.auto import tqdm

# l_bar='{desc}...({n_fmt}/{total_fmt} {percentage:3.2f}%)'
# r_bar= '{n_fmt}/{total_fmt}'
# r_bar= '{n_fmt}/{total_fmt} [{rate_fmt}{postfix}]'
# bar_format = f'{l_bar}|{{bar}}|{r_bar}{"{postfix}"} '
def bar_format(show_total):
    if show_total: return '{desc}...|{bar}|({n_fmt}/{total_fmt} {percentage:3.0f}%){postfix}'
    else: return '{desc}...{percentage:3.2f}% {postfix}'

def progress_indicator(*args, show_total=True, **kwargs):
    return tqdm(*args, **kwargs, bar_format=bar_format(show_total))

def generateDates(start, end):
    '''
    根据形如 20220103 的字符串 生成 起止时间中间的所有日期的字符串
    (包含 start 和 end)
    '''
    start, end = str2date(start), str2date(end) + timedelta(1)
    if not (end - start).days > 1: return None

    res = []
    while (end - start).days:
        res.append(date2str(start))
        start += timedelta(1)
    return res

def checkContinuous(lis: list, _type=None, _print=False):
    '''
    计算列表中连续的值的范围
    '''

    if _type == "date": convert = [str2date, date2str]
    else: convert = [lambda f: int(f)] * 2
    lis = [convert[0](i) for i in lis]
    msg = ('TOTAL', 'RANGE', 'LAST')

    lis.sort()
    # res 是元素为 tuple 的列表，以 (start, end) 的形式储存连续范围的起止
    res = []

    if _print: print(f"{msg[0]:6s} {convert[1](lis[0])} - {convert[1](lis[-1])}\n")
    
    start = lis[0]
    for i, l in enumerate(lis[1:]):
        _delta = l - lis[i]
        if isinstance(_delta, timedelta): _delta = _delta.days
        if _delta == 1: continue
        else:
            _res = (convert[1](start), convert[1](lis[i]))
            res.append(_res)
            start = l

            if _print: print(f"{msg[1]:6s} {_res}")

    if start == lis[0] and _print: print('Continuous'); return res

    _res = (convert[1](start), convert[1](l))
    res.append(_res)
    start = l

    if _print: print(f"{msg[2]:6s} {_res}")

    return res

def interpolate(dic: dict, avg = None, _type: str='date', _print=False):
    '''
    插补数据。数据是字典格式，其中key是日期，value是一个 {jCode: value} 格式的字典
    '''
    assert avg is not None

    # 获得 keys 的连续性信息
    res_continuous = checkContinuous(dic.keys(), "date")

    # 根据连续性信息插补字典对象
    res = []
    count = 0
    for i, (start, end) in enumerate(res_continuous[1:]):
        end_last = res_continuous[i][1]
        
        if _type == 'date':
            value = avg(dic, end_last, start)
            for d in generateDates(end_last, start)[1:-1]:
                if _print: print(f"插补 {d} 为 {end_last}-{start} 均值")
                dic[d] = value
                count += 1
        else:
            _res = list(range(int(end_last) + 1, int(start)))
            _res = [str(i) for i in _res]
            res += _res
    print(f'{count} new data interpreted')
    # no need to return because dict object is changable and the codes above operate directly on the dict object.
    # return dic
    