import argparse, json, os, datetime
import preprocess_tgnn
from train import *
import torch

import random, numpy as np, time

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

def make_results_dirs():

    results_dir = "results"
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    results_dir = os.path.join(
        results_dir,
        datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)
    
    return results_dir

def experiment(args, config):

    results_dir = make_results_dirs()

    config_tgnn = config['pandemic_tgnn_dataset']

    # 使用数据集对应文章的方法加载数据集 - pandemic tgnn
    # meta_labs, meta_graphs, meta_features, meta_y = \
    meta_dataset = preprocess_tgnn.read_meta_datasets(
            args.window, config_tgnn)
    
    nfeat = meta_dataset[2][0][0].shape[1] # `meta_dataset[2]` is features
    
    country_idx = dict(config_tgnn['country_idx'])
    for i, country in enumerate(country_idx.keys()):
        
        print("Training country", config_tgnn['countries'][i], '...')

        idx = country_idx[country]

        labels, gs_adj, features, y = (item[idx] for item in meta_dataset)

        n_samples= len(gs_adj)
        
        n_nodes = gs_adj[0].shape[0]
        
        with open(os.path.join(results_dir, "results_"+country+".csv"), "w") as fw:
            
            fw.write(f"Epoch,val_loss,train_loss,time\n")
            
            print('Start training...')
            
            test_sample, window, sep = 15, 7, 10
            shift, batch_size, device = 1, 8, torch.device('cuda:0')
            
            idx_train = list(range(window-1, test_sample-sep))
            
            idx_val = list(range(test_sample-sep, test_sample, 2)) 
                            
            idx_train = idx_train+list(range(test_sample-sep+1, test_sample, 2))

            err = run_model(gs_adj, features, y, idx_train, idx_val, 1, shift, batch_size, device, test_sample, fw)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300,
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

    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
    args = parser.parse_args()
    experiment(args, config)

if __name__ == '__main__':
    main()
