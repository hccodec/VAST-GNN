import torch, os
from train_test import compute_metrics, validate_test_process
from utils.data_process.dataforgood import split_dataset, load_data

pth_name = 'best_model_jp.pth'
log_name = "log.txt"

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-d", "--dir", help='result dir')
    args = parser.parse_args()
    
    trained_model = torch.load()

    criterion = torch.nn.MSELoss()

    data_origin, date_all = load_data(args)
    train_loader, validation_loader, test_loader, train_origin, validation_origin, test_origin, train_indices, validation_indices, test_indices = split_dataset(args, data_origin, date_all)

    validation_result, validation_hat, validation_real = validate_test_process(trained_model, criterion, validation_loader, args.device)
    test_result, test_hat, test_real = validate_test_process(trained_model, criterion, test_loader, args.device)

    metrics = compute_metrics(
        validation_hat, validation_real,
        test_hat, test_real, args.case_normalize_ratio
    )
