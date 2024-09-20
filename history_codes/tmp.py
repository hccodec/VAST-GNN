# vis_results.py
from argparse import ArgumentParser
import pickle, os

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-e", "--exp", choices=['eu', 'jp'], default="eu", help="choose the experiment")
    parser.add_argument("-t", "--timestr", help="choose the specific time in , default for the latest one")
    args = parser.parse_args()
    return args


def make_config_filepath(args):
    timestr, exp = args.timestr, args.exp
    if timestr is None: timestr = sorted(os.listdir('results'))[-1]
    return os.path.join('results', timestr, exp)


if __name__ == "__main__":
    args = parse_args()
    fn = make_config_filepath(args)
    data_res = []
    for bin_file in os.listdir(fn):
        if not bin_file.endswith('bin'): continue
        with open(os.path.join(fn, bin_file), 'rb') as f:
            _v = pickle.load(f)
            data_res.append(_v)
    pass
