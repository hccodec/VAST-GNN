
from utils.args import models_list

def select_model(args, train_loader):

    shape = [tuple(i.shape) for i in train_loader.dataset[0]]

    dynst_model_args = {
        "in_dim": args.window,
        "out_dim": 1,
        # "hidden": 32,
        "hidden_enc": 32,
        "hidden_dec": 64,
        "window_size": shape[0][-1],
        "num_heads": 4,
        "tcn_layers": 2,
        "lstm_layers": 2,
        "graph_layers": 2,
        "dropout": 0.5,
        "device": args.device,
        "no_graph_gt": args.no_graph_gt
    }
    lstm_model_args = {
        # 'in': train_loader.dataset[0][2].shape[-1],
        "lstm": {"hid": 128},
        "linear": {"hid": 64},
        "dropout": 0.5,
        # 'out': 32,
        "shape": shape,
    }
    mpnn_lstm_model_args = dict(nfeat=shape[0][-1], nhid=64, nout=1, # n_nodes=shape[0][1],
                              window=shape[0][0], dropout=0.5)

    assert args.model in models_list
    index = models_list.index(args.model)

    if index == 0:
        from models.LSTM_ONLY import LSTM_MODEL
        model_args = lstm_model_args
        model = LSTM_MODEL(args, model_args).to(args.device)
    elif index == 1:
        from models.dynst import dynst_extra_info, dynst
        model_args = dynst_model_args
        model = dynst(*model_args.values()).to(args.device)
    elif index == 2:
        from models.MPNN_LSTM import MPNN_LSTM
        model_args = mpnn_lstm_model_args
        model = MPNN_LSTM(*model_args.values()).to(args.device)
    else:
        raise ValueError("请选择模型：" + ", ".join(models_list))

    return model, model_args