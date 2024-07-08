import torch, os
from train_test import compute, validate_test_process
from utils.data_process import split_dataset, load_data
from main import parse_args

pth_name = 'best_model_jp.pth'

def get_latest_model(args, result_dir='results'):
    print(args)
    latest = os.listdir(result_dir)[-1]
    print(f'读取{latest}')
    path = os.path.join(result_dir, latest, pth_name)
    return path if os.path.exists(path) else ''

if __name__ == '__main__':
    args = parse_args()
    latest_model = get_latest_model(args)
    trained_model = torch.load(latest_model)
    criterion = torch.nn.MSELoss()

    data_origin, date_all = load_data(args)
    train_loader, validation_loader, test_loader, train_origin, validation_origin, test_origin, train_indices, validation_indices, test_indices = split_dataset(args, data_origin, date_all)

    validation_result, validation_hat, validation_real = validate_test_process(trained_model, criterion, validation_loader)
    test_result, test_hat, test_real = validate_test_process(trained_model, criterion, test_loader)

    metrics = compute(
        validation_hat, validation_real,
        test_hat, test_real, args.case_normalize_ratio
    )
