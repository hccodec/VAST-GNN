import argparse, os, re

from utils.custom_datetime import str2date

pattern_subdir = re.compile(r"^(.*)_(\d+)_(\d+)_w(\d+)_s(\d+)_(\d+)$")
countries = ["England", "France", "Italy", "Spain"]

pattern_sort_key = re.compile(r"^(\d+)->(\d+) \(w(\d+)s(\d+)\)$")
def sort_key(item):
    # 提取数字并返回一个元组用于排序
    # 排序列表类似于 ['7->1 (w7s6)', '7->1 (w7s13)', '7->1 (w7s2)']
    match = pattern_sort_key.search(item)
    if match: return list(map(int, pattern_sort_key.search(item).groups()))
    else: return None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=str, default="test")
    parser.add_argument("--exp", type=str, default="test1")
    parser.add_argument("-d", "--dir", type=str, default="", help="数据集的上一层文件夹")
    parser.add_argument("-s", "--subdir", default="", help="指定某一子目录")
    args =  parser.parse_args()
    if args.subdir != "":
        args.dir = os.path.dirname(os.path.dirname(args.subdir))
    return args

def extract_results(args):
    result_dir, exp_name, dir = args.result_dir, args.exp, args.dir
    if dir == "": dir = f"results/{result_dir}/exp_{exp_name}"
    print(f"从 {dir} 中提取结果")
    results = {}

    datasets = os.listdir(dir)
    for dataset in datasets:
        exps = os.listdir(os.path.join(dir, dataset))
        exp_result = []
        for exp in exps:
            assert pattern_subdir.search(exp), exp
            model, xdays, ydays, window, shift, timestr = pattern_subdir.search(exp).groups()

            log_path = os.path.join(dir, dataset, exp,  "log.txt")
            if not os.path.exists(log_path): continue
            res = extract_from_logfile(log_path)

            exp_result.append(dict(
                model=model, xdays=xdays, ydays=ydays, window=window, shift=shift, timestr=timestr, res=res
            ))
        results[dataset] = exp_result

    if len(results) == 1: results = list(results.values())[0]
    return results

def extract_from_logfile(logpath):

    res = {}
    lines_to_keep = []
    capture = -1

    with open(logpath) as f:
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
    # print(lines)
    return {country: res}

def print_err(args, results, _models, i):
    s = {'minvalloss': {}, 'latest': {}}

    for result in results:
        x, y, window, model, shift, timestr = result['xdays'], result['ydays'], result['window'], result['model'], result['shift'], result['timestr']
        if model != _models[i]: continue
        r = result['res']

        if args.subdir:
            if not args.subdir.split("/")[-1] == f"{model}_{x}_{y}_w{window}_s{shift}_{timestr}": continue

        key = f"{x}->{y} (w{window}s{shift})"
        # key = f"{x}->{y} (w{window}s{shift}) {str(str2date(timestr, '%Y%m%d%H%M%S'))}"
        s['minvalloss'][key] = {}
        s['latest'][key] = {}
        for country in countries:
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

            epoch_minvalloss = r[country]['minvalloss']['epoch']
            err_val_minvalloss = r[country]['minvalloss']['err_val']
            err_test_minvalloss = r[country]['minvalloss']['err_test']

            epoch_latest = r[country]['latest']['epoch']
            err_val_latest = r[country]['latest']['err_val']
            err_test_latest = r[country]['latest']['err_test']

            s['minvalloss'][key][country] = dict(
                err_val=f"{err_val_minvalloss}",
                err_test=f"{err_test_minvalloss}",
                epoch=f"{epoch_minvalloss}"
            )
            s['latest'][key][country] = dict(
                err_val=f"{err_val_latest}",
                err_test=f"{err_test_latest}",
                epoch=f"{epoch_latest}"
            )
    _ = 1
    if args.subdir: print(args.subdir)
    for k in s:
        if k != 'minvalloss': continue
        keys = sorted(s[k].keys(), key=sort_key)
        print(' | '.join(keys))
        print(f'[{_models[i]:>{9}s}]', '\t'.join(['\t'.join(map(lambda c: c['err_test'], v.values())) for v in [s[k][_k] for _k in keys]]))
        # print('[err_val ]', ' | '.join([' '.join(map(lambda c: c['err_val'], v.values())) for v in s[k].values()]))
        # print(' | '.join([' '.join(map(lambda c: c['epoch'], v.values())) for v in s[k].values()]))


if __name__ == "__main__":
    args = parse_args()
    results = extract_results(args)
    
    # get models
    models = []
    for result in results: models.append(result['model'])
    models = list(set(models))
    # assert set(models) == {'dynst', 'mpnn_lstm'}, models
    # models = ['mpnn_lstm', 'dynst']
    print()
    print("[err_test] {}\n".format(','.join(countries)))
    for i in range(len(models)):
        print_err(args, results, models, i)
    print()