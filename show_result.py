import argparse, os, re

from utils.datetime import str2date

from utils.logger import logger

pattern_subdir = re.compile(r"^(.*)_(\d+)_(\d+)_w(\d+)_s(\d+)_(\d+)$")
countries = {"dataforgood": ["England", "France", "Italy", "Spain", "NewZealand"], "japan": ["Japan"]}

pattern_sort_key = re.compile(r"^(\d+)->(\d+) \(w(\d+)s(\d+)\)$")
def sort_key(item):
    # 提取数字并返回一个元组用于排序
    # 排序列表类似于 ['7->1 (w7s6)', '7->1 (w7s13)', '7->1 (w7s2)']
    match = pattern_sort_key.search(item)
    if match: return list(map(int, pattern_sort_key.search(item).groups()))
    else: return None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--subdir", default="", help="指定数据集下层的某目录")
    parser.add_argument("-d", "--dir", type=str, default="", help="数据集上层的目录")
    args =  parser.parse_args()
    if args.subdir != "":
        args.dir = os.path.dirname(os.path.dirname(args.subdir))
    return args

def extract_results(dir):
    results = {}

    datasets = os.listdir(dir)
    for dataset in datasets:
        exps = os.listdir(os.path.join(dir, dataset))
        exp_result = []
        for exp in exps:
            assert pattern_subdir.search(exp), exp
            model, xdays, ydays, window, shift, timestr = pattern_subdir.search(exp).groups()

            log_path = os.path.join(dir, dataset, exp, "log.txt")
            if os.path.exists(log_path): res = extract_from_logfile(log_path)
            else:
                log_country_paths = [os.path.join(dir, dataset, exp, country_path) for country_path in os.listdir(os.path.join(dir, dataset, exp)) if not country_path.startswith("tensorboard") and os.path.isdir(os.path.join(dir, dataset, exp, country_path))]
                if all([os.path.exists(p) for p in log_country_paths]):
                    res = {}
                    for country_path in log_country_paths:
                        country = country_path.split("/")[-1]
                        log_path = os.path.join(country_path, "log.txt")
                        res.update(extract_from_logfile(log_path))
            exp_result.append(dict(
                model=model, xdays=xdays, ydays=ydays, window=window, shift=shift, timestr=timestr, res=res
            ))
        results[dataset] = exp_result

    # if len(results) == 1: results = list(results.values())[0]
    return results

def extract_from_logfile(logpath):

    res = {}
    lines_to_keep = []
    capture = -1

    with open(logpath, encoding="utf-8") as f:
        for line in f:
            if "训练完毕，开始评估" in line:
                capture = 8
                lines_to_keep.append(line)
            elif capture > 0:
                capture -= 1
                lines_to_keep.append(line)
            elif capture == 0:
                res.update(process_log_segment(lines_to_keep))
                lines_to_keep = []
                capture = -1
    return res

def process_log_segment(lines):

    # 定义正则表达式来匹配日志中的信息
    pattern_country = re.compile(r"训练完毕，开始评估: (\w+)")
    pattern_loss = re.compile(r"\[val\(MAE/RMSE\)\] (\d+\.\d+)/(\d+\.\d+), \[test\(MAE/RMSE\)\] (\d+\.\d+)/(\d+\.\d+)")

    pattern_err = re.compile(r"\[err\(val/test\)\] (\d+\.\d+)/(\d+\.\d+)")
    pattern_corr = re.compile(r"\[corr\(train/val/test\)\] (.*)/(.*)/(.*)")
    pattern_hits10 = re.compile(r"\[hits10\(train/val/test\)\] (\d+\.\d+)/(\d+\.\d+)/(\d+\.\d+)")

    # pattern_err = re.compile(r"\[err_val\] (\d+\.\d+), \[err_test\] (\d+\.\d+)")
    pattern_latest_epoch = re.compile(r"\[最新 \(epoch (\d+)\)\]")
    pattern_min_val_epoch = re.compile(r"\[最小 val loss \(epoch (\d+)\)\]")

    match_patterns = [
        pattern_country, None, pattern_latest_epoch, pattern_loss, pattern_err,
        None, pattern_min_val_epoch, pattern_loss, pattern_err
    ]

    res = {}
    country = ""
    for i, (line, pattern) in enumerate(zip(lines, match_patterns)):
        match = pattern.search(line) if pattern else None
        if i == 0:
            country = match.groups(1)[0] if match else ""
        elif i == 2 or i == 6:
            epoch= list(map(float, match.groups(1)))[0] if match else '-'
            res['latest' if i == 2 else "minvalloss"] = dict(epoch=epoch)
        elif i == 3 or i == 7:
            losses = list(map(float, match.groups())) if match else -1
            res["latest" if i == 3 else "minvalloss"].update(dict(losses=losses))
        elif i == 4 or i == 8:
            # 提取 error
            err_val, err_test = list(map(float, match.groups()))
            res["latest" if i == 4 else "minvalloss"].update(dict(err_val=err_val, err_test=err_test))
            # 提取可能存在的 correlation
            if pattern_corr.search(line):
                corr_train, corr_val, corr_test = list(map(str, pattern_corr.search(line).groups()))
                res["latest" if i == 4 else "minvalloss"].update(dict(corr_train=corr_train, corr_val=corr_val, corr_test=corr_test))
            # 提取可能存在的 HITS@10
            if pattern_hits10.search(line):
                hits10_train, hits10_val, hits10_test = list(map(float, pattern_hits10.search(line).groups()))
                res["latest" if i == 4 else "minvalloss"].update(dict(hits10_train=hits10_train, hits10_val=hits10_val, hits10_test=hits10_test))
    # logger.info(lines)
    return {country: res}

def print_err(results, dataset, _models, i, subdir = None, mode = 0):
    s = {'minvalloss': {}, 'latest': {}}

    for result in results[dataset]:
        x, y, window, model, shift, timestr = result['xdays'], result['ydays'], result['window'], result['model'], result['shift'], result['timestr'] if 'timestr' in result else ''
        if model != _models[i]: continue
        r = result['res']

        if subdir:
            if not subdir.split("/")[-1] == f"{model}_{x}_{y}_w{window}_s{shift}_{timestr}": continue

        key = f"{x}->{y} (w{window}s{shift})"
        # key = f"{x}->{y} (w{window}s{shift}) {str(str2date(timestr, '%Y%m%d%H%M%S'))}"
        s['minvalloss'][key] = {}
        s['latest'][key] = {}
        for country in countries[dataset]:
            if not country in r:
                s['minvalloss'][key][country] = dict(
                    err_val="-",
                    err_test="-",
                    epoch="-"
                )
                s['latest'][key][country] = dict(
                    err_val="-",
                    err_test="-",
                    epoch="-"
                )
                continue

            epoch_minvalloss = int(r[country]['minvalloss']['epoch'])
            err_val_minvalloss = r[country]['minvalloss']['err_val']
            err_test_minvalloss = r[country]['minvalloss']['err_test']
            hits10_test_minvalloss = r[country]['minvalloss']['hits10_test']

            epoch_latest = int(r[country]['latest']['epoch'])
            err_val_latest = r[country]['latest']['err_val']
            err_test_latest = r[country]['latest']['err_test']
            hits10_test_latest = r[country]['minvalloss']['hits10_test']

            s['minvalloss'][key][country] = dict(
                err_val=f"{err_val_minvalloss}",
                err_test=f"{err_test_minvalloss}",
                hits10_test = f"{hits10_test_minvalloss}",
                epoch=f"{epoch_minvalloss}"
            )
            s['latest'][key][country] = dict(
                err_val=f"{err_val_latest}",
                err_test=f"{err_test_latest}",
                hits10_test = f"{hits10_test_latest}",
                epoch=f"{epoch_latest}"
            )

    msg = ""
    
    if subdir: msg += subdir + '\n'
    for k in s:
        if k != 'minvalloss': continue
        keys = sorted(s[k].keys(), key=sort_key)
        msg += ' | '.join(keys) + '\n'
        if mode == 0:
            msg += f'[{_models[i]:>{9}s}]\t' + '\t'.join(['\t'.join(map(lambda c: c['err_test'], v.values())) for v in [s[k][_k] for _k in keys]]) + '\n'
            msg += f'[{"epoch":>{9}s}]\t' + '\t'.join(['\t'.join(map(lambda c: c['epoch'], v.values())) for v in [s[k][_k] for _k in keys]]) + '\n'
        elif mode == 1:
            msg += '\t'.join(['\t'.join(map(lambda c: '\t'.join((c['err_test'], c['epoch'])), v.values())) for v in [s[k][_k] for _k in keys]]) + '\n'
        elif mode == 2:
            msg += f'[{_models[i]:>{9}s}]\t' + '\t'.join(['\t'.join(map(lambda c: c['hits10_test'], v.values())) for v in [s[k][_k] for _k in keys]]) + '\n'
            msg += f'[{_models[i]:>{9}s}]\t' + '\t'.join(['\t'.join(map(lambda c: c['err_test'], v.values())) for v in [s[k][_k] for _k in keys]]) + '\n'
            msg += f'[{"epoch":>{9}s}]\t' + '\t'.join(['\t'.join(map(lambda c: c['epoch'], v.values())) for v in [s[k][_k] for _k in keys]]) + '\n'
        elif mode == 3:
            msg += '\t'.join(['\t'.join(map(lambda c: '\t'.join((c['hits10_test'], c['err_test'], c['epoch'])), v.values())) for v in [s[k][_k] for _k in keys]]) + '\n'
        else: raise NotImplementedError
        # logger.info('[err_val ]', ' | '.join([' '.join(map(lambda c: c['err_val'], v.values())) for v in s[k].values()]))
        # logger.info(' | '.join([' '.join(map(lambda c: c['epoch'], v.values())) for v in s[k].values()]))
    
    return msg

def merge_results(results):
    '''
    合并 results 中的结果，返回一个列表，每个元素是一个字典，包含 model, xdays, ydays, window, shift, res
    合并依据是 model, xdays, ydays, window, shift, 并且假定当前述 key 对应 value 都相同的情形下， res 的 key 都不相同。然后把 res 合并
    所以合并后 model, xdays, ydays, window, shift 仅一套，配以合并后的 res
    思路：
    先定义一个 merged_results 列表存放最终结果
    若 merged_results 中没有该 model, xdays, ydays, window, shift 组合的 res，则将整个 result 添加到 merged_results 中
    若有该组合则更新 res，把新 res 字典与原来 res 合并（assert 新旧 res 键不重复）
    '''
    merged_results_dic = {}
    
    for dataset in results:
        merged_results_dic.update({dataset: {}})
        for result in results[dataset]:
            model, xdays, ydays, window, shift, res = result['model'], result['xdays'], result['ydays'], result['window'], result['shift'], result['res']
            k = (model, xdays, ydays, window, shift)
            if k not in merged_results_dic: merged_results_dic[dataset].update({k: res})
            else: merged_results_dic[dataset][k].update(res)

    merged_results = merged_results_dic.copy()
    for dataset in merged_results:
        for k, v in merged_results_dic[dataset].items():
            merged_results[dataset] = [dict(zip(['model', 'xdays', 'ydays', 'window', 'shift', 'res'], k + (v,))) for k, v in merged_results_dic[dataset].items()]
    
    return merged_results


def show_result(dir, subdir = "", mode = 0):

    results = extract_results(dir)
    results = merge_results(results)

    msg = "\n"
    for dataset in results:
        msg += dataset + '\n'
        # get models
        models = []
        for result in results[dataset]: models.append(result['model'])
            
        models = sorted(set(models), key=lambda x: {"mpnn_lstm": 0, "lstm": 1, "dynst": 2}.get(x, float('inf')))

        # assert set(models) == {'dynst', 'mpnn_lstm'}, models
        # models = ['mpnn_lstm', 'dynst']
        msg += "[err_test] {}\n".format(','.join(countries[dataset])) + '\n'
        for i in range(len(models)):
            msg += print_err(results, dataset, models, i, subdir, mode) + '\n'
        if mode == 0: logger.info(msg)
        msg += '\n'
    return msg

if __name__ == "__main__":
    args = parse_args()
    subdir, dir = args.subdir, args.dir
    show_result(dir, subdir)
    logger.info("")