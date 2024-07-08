import argparse, json, os
from train import *
import torch, pickle
from utils.debugtools import tmpload, tmpsave

import random, numpy as np, time

# 设置随机种子
seed = 5
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)#让显卡产生的随机数一致
torch.cuda.manual_seed_all(seed)#多卡模式下，让所有显卡生成的随机数一致？这个待验证
np.random.seed(seed)#numpy产生的随机数一致
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from utils.datetime import datetime
now = datetime.now()

dataset_lst = ["eu", "jp"]

def make_results_dir(dataset: str):
    results_dir = os.path.join("results", now.strftime("%Y%m%d%H%M%S"), str(dataset)) + "_incomplete"
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

def experiment_tgnn(args, config):
    print('\n=== Europe ===\n')
    import preprocess_tgnn

    results_dir = make_results_dir(dataset_lst[0])

    config_tgnn = config['pandemic_tgnn_dataset']

    # 使用数据集对应文章的方法加载数据集 - pandemic tgnn
    # meta_labs, meta_graphs, meta_features, meta_y = \
    meta_dataset = preprocess_tgnn.read_meta_datasets(args.window, config_tgnn)
    
    nfeat = meta_dataset[2][0][0].shape[1] # `meta_dataset[2]` is features
    
    country_idx = dict(config_tgnn['country_idx'])
    for i, country in enumerate(country_idx.keys()):
        
        print("Training country", config_tgnn['countries'][i], '...')

        idx = country_idx[country]

        labels, gs_adj, features, y = (item[idx] for item in meta_dataset)

        n_samples= len(gs_adj)
        
        n_nodes = gs_adj[0].shape[0]
        
        test_sample, window, sep = 15, args.window, args.sep
        shift, batch_size, device = 1, 8, torch.device('cuda:0')
        
        idx_train = list(range(window - 1, test_sample - sep))
        
        idx_val = list(range(test_sample - sep, test_sample, 2)) 
                        
        idx_train = idx_train + list(range(test_sample - sep + 1, test_sample, 2))

        train_among_epochs, val_among_epochs, err_among_epochs, best_model_path, err = run_tgnn_model(
            gs_adj, features, y, idx_train, idx_val, args.graph_window, shift, batch_size, device, test_sample,
            results_dir, country)
        # graphs        : list,
        # features      : list,
        # y             : list,
        # idx_train     : list,
        # idx_val       : list,
        # graph_window  : int,
        # shift         : int,
        # batch_size    : int,
        # device        : torch.device,
        # test_sample   : int,
            
        # torch.save(model.state_dict(), os.path.join(results_dir, f'model_best_{country}.pth.tar'))
        # 存到文件中
        with open(os.path.join(results_dir, f'losses_{country}.bin'), 'wb') as f:
            pickle.dump((train_among_epochs, val_among_epochs), f)

        print('Training completed. Best model save in {}/:\n[best_val_loss {}({}), best_val_err {}({})]'.format(
            results_dir,
            min(val_among_epochs), val_among_epochs.index(min(val_among_epochs)),
            min(err_among_epochs), err_among_epochs.index(min(err_among_epochs))
        ))
    
    os.rename(results_dir, results_dir.split('_')[0])

def experiment_multiwave(args):
    print('\n=== Japan  ===\n')
    from preprocess_multiwave import read_data, device

    results_dir = make_results_dir(dataset_lst[1])

    print(f'results save in {results_dir}')

    #4.1
    #read the data
    # train_x_y, validate_x_y, test_x_y, all_mobility, all_infection, train_original, validate_original, test_original, train_list, validation_list =read_data()
    tokyo_datafile = 'dataset_tokyo.bin'
    if not os.path.exists(tokyo_datafile):
        res = read_data()
        with open(tokyo_datafile, 'wb') as f:
            print(f'Reading original data from dataset files...')
            pickle.dump(res, f)
            del res
    else:
        print(f'Reading processed data from bin file [{tokyo_datafile}]...')
    with open(tokyo_datafile, 'rb') as f:
        try:
            train_x_y, validate_x_y, test_x_y, all_mobility, all_infection, train_original, validate_original, test_original, train_list, validation_list = pickle.load(f)
        except Exception as e:
            msg = "- Failed: "
            if isinstance(e, EOFError): msg += 'File collapsed'
            elif isinstance(e, pickle.UnpicklingError): msg = 'File irregular'
            else: msg = e
            print(msg)
            os.remove(tokyo_datafile)
            return experiment_multiwave(args)

    with open(os.path.join(results_dir, 'args.txt'), 'w') as f:
        f.write(str(args))

    e_losses, trained_model = run_tokyo_model(train_x_y, validate_x_y, test_x_y, device, results_dir, args)



    os.rename(results_dir, results_dir.split('_')[0])




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate.')
    parser.add_argument('--hidden', type=int, default=64,
                        help='Number of hidden units.')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Size of batch.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate.')
    parser.add_argument('--window', type=int, default=7,
                        help='Size of window for features.')
    parser.add_argument('--graph-window', type=int, default=7,
                        help='Size of window for graphs in MPNN LSTM.')
    parser.add_argument('--recur',  default=False,
                        help='True or False.')
    parser.add_argument('--early-stop', type=int, default=100,
                        help='How many epochs to wait before stopping.')
    parser.add_argument('--start-exp', type=int, default=15,
                        help='The first day to start the predictions.')
    parser.add_argument('--ahead', type=int, default=14,
                        help='The number of days ahead of the train set the predictions should reach.')
    parser.add_argument('--sep', type=int, default=10,
                        help='Seperator for validation and train set.')
    parser.add_argument('--exp', type=str, choices=dataset_lst, required=True,
                        help='Which experiment to perform.')
    parser.add_argument('--use-graph-learner', default=False, help='是否使用图学习器替代原图结构')
    parser.add_argument('--graph-learner-decay', choices=['linear', 'square', 'step', 'exp'], default='square', help='设置使用图学习器评估loss的衰减')
    parser.add_argument('--graph-learner-decay-k', default=1e-2, help='lambda_T = k * lambda_0')
    parser.add_argument('--graph-learner-decay-start-epoch', default=10, help='衰减开始epoch')
    parser.add_argument('--graph-learner-decay-start-lambda', default=1., help='衰减开始值')

    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
    args = parser.parse_args()

    assert args.graph_learner_decay_k > 0 and args.graph_learner_decay_k < 1

    if args.exp in dataset_lst:
        idx = dataset_lst.index(args.exp)
        if idx == 0:
            experiment_tgnn(args, config)
        elif idx == 1:
            experiment_multiwave(args)
        else:
            print('Unimplemented exp: %s(%d)' % (args.exp, idx))
    else:
        print('Unknown exp:', args.exp)

if __name__ == '__main__':
    main()
    
    # from preprocess_multiwave import read_data
    # read_data()