import argparse, os, re

from custom_datetime import str2date


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=str, default="test")
    parser.add_argument("--exp", type=str, default="test1")
    parser.add_argument("-d", "--dir", type=str, default="")
    return parser.parse_args()

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
            xdays, ydays, window, model, timestr = re.match(r"(\d+)_(\d+)_w(\d)_(.*)_(\d+)", exp).groups()

            log_path = os.path.join(dir, dataset, exp,  "log.txt")
            res = extract_from_logfile(log_path)

            exp_result.append(dict(
                xdays=xdays, ydays=ydays, window=window, model=model, timestr=timestr, res=res
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
    pattern_err = re.compile(r"\[err\] (\d+\.\d+)")
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
            err = list(map(float, match.groups(1)))[0]
            res["latest" if i == 4 else "minvalloss"].update(dict(err=err))
    # print(lines)
    return {country: res}

def print_err(results):
    countries = ["England", "France", "Italy", "Spain"]
    s = {'minvalloss': {}, 'latest': {}}

    for result in results:
        x, y, window = result['xdays'], result['ydays'], result['window']
        model, timestr, r = result['model'], result['timestr'], result['res']
        key = f"{x}->{y} ({window}) {str(str2date(timestr, '%Y%m%d%H%M%S'))}"
        s['minvalloss'][key] = {}
        s['latest'][key] = {}
        for country in countries:
            epoch_minvalloss = r[country]['minvalloss']['epoch']
            err_minvalloss = r[country]['minvalloss']['err']
            epoch_latest = r[country]['latest']['epoch']
            err_latest = r[country]['latest']['err']

            s['minvalloss'][key][country] = {'err': f"{err_minvalloss}"}
            s['latest'][key][country] = {'err': f"{err_latest}"}

            s['minvalloss'][key][country]['epoch'] = f"{epoch_minvalloss}"
            s['latest'][key][country]['epoch'] = f"{epoch_latest}"

    _ = 1
    print(countries)
    for k in s:
        print(k)
        print(' | '.join([' '.join(map(lambda c: c['err'], v.values())) for v in s[k].values()]))
        # print(' | '.join([' '.join(map(lambda c: c['epoch'], v.values())) for v in s[k].values()]))


if __name__ == "__main__":
    args = parse_args()
    results = extract_results(args)
    print_err(results)