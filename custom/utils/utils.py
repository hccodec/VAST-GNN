import os, torch, numpy as np, random
from models.SAB_GNN import Multiwave_SpecGCN_LSTM
from utils.logger import logger
import traceback, functools

font_stype = ['', 'bold', '']
color_code = ['red', 'green', 'yellow', 'blue', 'purple', 'cyan']
models = ['sabgnn', 'sab_gnn_case_only', 'lstm']

def _color(color, content):
    assert color in color_code
    _color = f"\033[1;{color_code.index(color) + 31}m"
    if isinstance(content, float): return f"{_color}{content:.3f}\033[0m"
    else: return f"{_color}{content}\033[0m"

def red(content): return _color('red', content)
def green(content): return _color('green', content)
def yellow(content): return _color('yellow', content)
def blue(content): return _color('blue', content)
def purple(content): return _color('purple', content)
def cyan(content): return _color('cyan', content)

def set_random_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)#让显卡产生的随机数一致
    torch.cuda.manual_seed_all(seed)#多卡模式下，让所有显卡生成的随机数一致？这个待验证
    np.random.seed(seed)#numpy产生的随机数一致
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    
def set_device(device_id):
    device_id = int(device_id)
    if device_id == -1:
        logger.info("使用 CPU 进行计算")
        return torch.device("cpu")
    elif torch.cuda.is_available():
        if 0 <= device_id < torch.cuda.device_count():
            logger.info(f"使用 GPU {device_id} 进行计算")
            return torch.device(f"cuda:{device_id}")
        else:
            logger.info(f"警告: 无效的 GPU 号 {device_id}, 切换到CPU")
            return torch.device("cpu")
    else:
        logger.info("警告: 未检测到可用的GPU, 使用CPU进行计算")
        return torch.device("cpu")
    

def catch(msg='出现错误，中断训练'):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except Exception as e:
                logger.info(msg)
                logger.info(str(e))
                for line in traceback.format_exc():
                    logger.info(str(line))
                    return None
        return wrapper
    return decorator
    
# models = ['sabgnn', 'sab_gnn_case_only', 'lstm']
def select_model(args, train_loader):
    specGCN_model_args = {
        'alpha': 0.2,
        'specGCN': {'hid': 6, 'out': 4, 'dropout': 0.5},
        'shape': list(train_loader.dataset[0][1].shape),
        'lstm': {'hid': 3}
    }

    assert args.model in models
    index = models.index(args.model)

    if index == 0:
        return Multiwave_SpecGCN_LSTM(args, specGCN_model_args)
    elif index == 1:
        specGCN_model_args['shape'] = train_loader.dataset[0][2].shape
        return Multiwave_SpecGCN_LSTM(args, specGCN_model_args)
    elif index == 2:
        return torch.nn.LSTM() # TODO